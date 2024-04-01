from .config import DataPrepConfig
from ..dataset_util import DataTag, Dataset as BinsenseDataset
from .roboflow_util import RoboflowDatasetReader
from .model_util import BBoxEmbedder
from ..embed_datastore import EmbeddingDatastore
from .dataset import BestBBoxDataset

from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import pandas as pd
import logging, torch, os

logger = logging.getLogger(__name__)

class BBoxDatasetEmbedder:
    def __init__(
        self, 
        model: BBoxEmbedder, 
        embedstore: EmbeddingDatastore, 
        config: DataPrepConfig) -> None:
        
        self.cfg = config
        self.model = model
        self.embedstore = embedstore
    
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
    
    def generate(self, batch_size=None, test_run=False):
        if batch_size is None:
            batch_size = self.cfg.batch_size
        
        if test_run is None:
            test_run = False
        
        downloaded_ds = RoboflowDatasetReader(self.cfg.dataset_download_path).read()
        best_bbox_df = self._get_best_bboxes(downloaded_ds)
        logger.info(f"total bbox labels are {best_bbox_df.shape[0]}")
        df = best_bbox_df[0:10] if test_run else best_bbox_df
        
        torch_ds = BestBBoxDataset(df, downloaded_ds, processor=self.model.processor())
        torch_dl = TorchDataLoader(torch_ds, batch_size=batch_size)

        progress_bar = tqdm(
            total=len(torch_ds), 
            desc="generating embeddings", file=open(os.devnull, 'w'))
        logger.info(str(progress_bar))
        progress_step = len(torch_ds) // 5
        for batch_idx, x in enumerate(torch_dl):
            with torch.no_grad():
                bbox_embeddings = self.model(x)# B x 512
                start_idx = batch_idx * batch_size
                end_idx = batch_idx * batch_size + len(x)
                bbox_labels = df.iloc[start_idx:end_idx]['bbox_label'].to_list()
                self.embedstore.put_many(bbox_labels, bbox_embeddings.detach())
            
            progress_bar.update(len(x))
            if progress_bar.n >= progress_step:
                logger.info(str(progress_bar))
                progress_step += progress_bar.n
        logger.info(str(progress_bar))
