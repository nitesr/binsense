from .config import DataPrepConfig
from ..dataset_util import DataTag, Dataset as BinsenseDataset, Yolov8Deserializer
from ..embed_datastore import SafeTensorEmbeddingDatastore
from .dataset import BestBBoxDataset
from ..lightning.model import LitImageEmbedder
from ..lightning.model_spec import ImageEmbedder
from ..utils import get_default_on_none

from typing import Union
from torch.utils.data import DataLoader as TorchDataLoader

import lightning as L
import pandas as pd
import logging

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
        bbox_df = pd.DataFrame(columns=['bbox_label', 'image_name', 'image_path', 'bbox_idx', 'bbox_area'])
        imgs_data = downloaded_ds.get_images(DataTag.TRAIN)
        for img_data in imgs_data:
            bboxes_data = downloaded_ds.get_bboxes(img_data.name)
            for bbox_idx, bbox_data in enumerate(bboxes_data):
                bbox_df.loc[len(bbox_df)] = [
                    bbox_data.label, img_data.name, 
                    img_data.path, bbox_idx,
                    bbox_data.width*bbox_data.height
                ]
        
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
        return torch_dl, best_bbox_df

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
    
    def generate(self, **kwargs):
        num_workers = kwargs.pop('num_workers') if 'num_workers' in kwargs else 0
        torch_dl, best_bbox_df = self._get_data_loader(num_workers)
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
        
        # progress_bar = tqdm(
        #     total=len(best_bbox_df), 
        #     desc="generating embeddings", file=open(os.devnull, 'w'))
        # logger.info(str(progress_bar))
        # progress_step = len(best_bbox_df) // 5
        # for batch_idx, x in enumerate(torch_dl):
        #     with torch.no_grad():
        #         bbox_embeddings = self.model(x)# B x 512
        #         start_idx = batch_idx * batch_size
        #         end_idx = batch_idx * batch_size + len(x)
        #         bbox_labels = df.iloc[start_idx:end_idx]['bbox_label'].to_list()
        #         bbox_embeddings = bbox_embeddings.detach().cpu()
        #         self.embedstore.put_many(bbox_labels, bbox_embeddings, device=device)
            
        #     progress_bar.update(len(x))
        #     if progress_bar.n >= progress_step:
        #         logger.info(str(progress_bar))
        #         progress_step += progress_bar.n
        # logger.info(str(progress_bar))
