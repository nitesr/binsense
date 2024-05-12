from .config import DataPrepConfig
from ..dataset_util import DataTag, Dataset as BinsenseDataset, Yolov8Deserializer
from ..embed_datastore import SafeTensorEmbeddingDatastore
from .dataset import BestBBoxDataset
from ..lightning.model import LitImageEmbedder
from ..lightning.spec import ImageEmbedder
from ..utils import get_default_on_none
from ..img_utils import convert_cxy_xy_and_scale

from torchvision.utils import draw_bounding_boxes
from matplotlib import pyplot as plt

from typing import Union
from torch.utils.data import DataLoader as TorchDataLoader

import lightning as L
import pandas as pd
import numpy as np
import logging, random, torch, os

logger = logging.getLogger(__name__)

class BBoxDatasetEmbedder:
    def __init__(
        self, 
        model: ImageEmbedder, 
        batch_size: int = None,
        config: DataPrepConfig = None,
        num_bbox_labels: int = None) -> None:
        self.cfg = get_default_on_none(config, DataPrepConfig())
        self.batch_size = get_default_on_none(batch_size, self.cfg.batch_size)
        self.model = model
        self.num_bbox_labels = num_bbox_labels
    
    def _get_best_bboxes(self, downloaded_ds: BinsenseDataset) -> pd.DataFrame:
        bbox_dict = {
            'bbox_label': [],
            'image_name': [],
            'image_path': [],
            'bbox_idx': [],
            'bbox_area': []
        }

        imgs_data = downloaded_ds.get_images(DataTag.TRAIN)
        for img_data in imgs_data:
            bboxes_data = downloaded_ds.get_bboxes(img_data.name)
            for bbox_idx, bbox_data in enumerate(bboxes_data):
                bbox_dict['bbox_label'].append(bbox_data.label)
                bbox_dict['image_name'].append(img_data.name)
                bbox_dict['image_path'].append(img_data.path)
                bbox_dict['bbox_idx'].append(bbox_idx)
                bbox_dict['bbox_area'].append(bbox_data.area)
        
        bbox_df = pd.DataFrame.from_dict(bbox_dict)
        best_bbox_idxs = bbox_df.groupby('bbox_label')['bbox_area'].idxmax()
        best_bbox_df = bbox_df.loc[best_bbox_idxs][['bbox_label', 'image_name', 'image_path', 'bbox_idx']]
        best_bbox_df.reset_index(drop=True, inplace=True)
        del bbox_df
        return best_bbox_df
    
    def _get_data_loader(self, num_workers: int = None) -> TorchDataLoader:
        downloaded_ds = Yolov8Deserializer(
            self.cfg.filtered_dataset_path,
            img_extns=['.jpg']).read()
        best_bbox_df = self._get_best_bboxes(downloaded_ds)
        logger.info(f"total bbox labels are {best_bbox_df.shape[0]}")
        
        torch_ds = BestBBoxDataset(best_bbox_df, downloaded_ds, processor=self.model.processor())
        torch_dl = TorchDataLoader(torch_ds, batch_size=self.batch_size, num_workers=num_workers)
        return torch_dl, best_bbox_df, torch_ds

    def _get_rw_embed_store(self, num_bbox_labels: int) -> SafeTensorEmbeddingDatastore:
        embedstore_size =  num_bbox_labels * self.model.get_embed_size() * 32
        embedstore_partition_size = 4 * 1024 * 1024 # 4MB per partition
        req_partitions = max(embedstore_size // embedstore_partition_size, 1)
        print(f"required partitions on embedding store are {req_partitions}")
        embed_store = SafeTensorEmbeddingDatastore(
            self.cfg.embed_store_dirpath, 
            req_partitions=max(embedstore_size // embedstore_partition_size, 1),
            read_only=False,
            clean_state=True
        )
        return embed_store
    
    def sample_crops(self, torch_ds: BestBBoxDataset):
        logger.info("sampling few crops for validation")
        
        n = len(torch_ds)
        r_idxs = random.sample(range(0, n), k=10)
        pixels = torch.stack([ torch_ds[i][2] for i in r_idxs ])
        query_bboxes = torch.stack([ torch_ds[i][1] for i in r_idxs ])
        
        pixels = self.model.processor().unnormalize_pixels(pixels)
        img_size = (pixels[0].shape[0], pixels[0].shape[1])
        
        pixels = torch.as_tensor(np.moveaxis(pixels, -1, 1))
        query_bboxes = convert_cxy_xy_and_scale(query_bboxes, img_size)
        
        annotated_imgs = []
        for i, img in enumerate(pixels):
            annotated_imgs.append(draw_bounding_boxes(
                image=img, 
                boxes=query_bboxes[i].unsqueeze(0), 
                labels=None, 
                colors=[(255,255,0)]))
        
        fig, axs = plt.subplots(10, 1, figsize=(8, 15))
        for i in range(0, 10):
            axs[i].imshow(annotated_imgs[i].permute(1, 2, 0))
        fig.suptitle("cropped images for embeddings")
        plt.savefig(
            fname=os.path.join(
                self.cfg.root_dir, 'sample_crops_images.png'), 
            format='png')
        return None


    def generate(self, **kwargs):
        num_workers = kwargs.pop('num_workers') if 'num_workers' in kwargs else 0
        num_workers = get_default_on_none(num_workers, 0)
        torch_dl, best_bbox_df, torch_ds = self._get_data_loader(num_workers)

        self.sample_crops(torch_ds)

        trainer = L.Trainer(**kwargs)
        embed_store = None
        if trainer.is_global_zero:
            num_bbox_labels = get_default_on_none(self.num_bbox_labels, len(best_bbox_df))
            embed_store = self._get_rw_embed_store(num_bbox_labels)
        
        best_bbox_labels = best_bbox_df['bbox_label'].to_list()
        lmodel = LitImageEmbedder(
            self.model,
            bbox_labels=best_bbox_labels,
            embed_ds=embed_store)
        trainer.predict(model=lmodel, dataloaders=torch_dl)
