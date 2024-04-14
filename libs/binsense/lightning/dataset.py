
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch.utils
from ..dataset_util import Yolov8Deserializer
from ..dataset_util import Dataset as BinsenseDataset
from ..dataset_util import DataTag
from ..dataprep.config import DataPrepConfig
from ..embed_datastore import EmbeddingDatastore, SafeTensorEmbeddingDatastore
from ..utils import get_default_on_none, backup_file
from .. import torch_utils as tutls

from typing import List, Tuple, Any
from collections import Counter
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch import Tensor

import lightning as L
import pandas as pd
import numpy as np

import os, logging, PIL, torch, random

logger = logging.getLogger(__name__)

class InImageQueryDatasetBuilder:
    def __init__(
        self, 
        target_fpath: str = None,
        data_dir: str = None,
        embed_ds: EmbeddingDatastore = None,
        cfg: DataPrepConfig = None) -> None:
        
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.data_dir = get_default_on_none(data_dir, self.cfg.filtered_dataset_path)
        self.target_fpath = get_default_on_none(target_fpath, self.cfg.inimage_queries_csv)
        self.embed_ds = get_default_on_none(
            embed_ds, SafeTensorEmbeddingDatastore( \
                dir_path=self.cfg.embed_store_dirpath, read_only=True
            ))
        self.pos_neg_ratio=self.cfg.inimage_queries_pos_neg_ratio
    
    def _load_queries(self, ds: BinsenseDataset) -> List[str]:
        query_labels = set()
        def add_query_label(tag: DataTag):
            for img_data in ds.get_images(tag):
                for bboxes_data in ds.get_bboxes(img_data.name):
                    if self.embed_ds.has(bboxes_data.label):
                        query_labels.add(bboxes_data.label)
                    else:
                        logger.warn(f'{bboxes_data.label} is not embedded!')
        add_query_label(DataTag.TRAIN)
        add_query_label(DataTag.VALID)
        return list(query_labels)
    
    def _load_df(self, ds: BinsenseDataset) -> pd.DataFrame:
        df = pd.DataFrame(columns=[
            'image_relpath', 'bbox_label', 'count', 'tag'
        ])
        def read_ds(tag: DataTag):
            imgs_data = ds.get_images(tag)
            for img_data in imgs_data:
                bboxes_data = ds.get_bboxes(img_data.name)
                bbox_labels = [b.label for b in bboxes_data]
                for l, c in Counter(bbox_labels).items():
                    _, dir_name = os.path.split(self.data_dir)
                    img_rel_path = os.path.join(dir_name, tag.value, 'images', img_data.name)
                    df.loc[len(df)] = [img_rel_path, l, c, tag.value]
        
        read_ds(DataTag.TRAIN)
        read_ds(DataTag.VALID)
        read_ds(DataTag.TEST)
        return df
    
    def _prepare_queries(self, query_labels: List, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        prepares the queries frame for the query_labels passed
            with the data (data_df) available
        """
        # do a cross product, pick valid ones and drop invalids
        cross_df = data_df.join(pd.Series(query_labels, name="query_label"), how='cross')
        
        # create negative queries
        mask = cross_df.query('bbox_label != query_label')
        cross_df.loc[mask.index, 'count'] = 0
        
        # drop false negative queries created by cross join.
        #   even though query_label in image it marks the count as 0
        #   as above mask looks at row level, we need to look at image level
        #   for each group (image) check in query_label exists in image 
        #       and if count is 0 delete it
        drop_idxs = []
        for _, idx in cross_df.groupby('image_relpath').groups.items():
            mask = cross_df.loc[idx].query('query_label in bbox_label & count == 0')
            drop_idxs.extend(list(mask.index))
        cross_df.drop(drop_idxs, inplace=True)
        cross_df.reset_index(drop=True, inplace=True)
        return cross_df[['query_label', 'image_relpath', 'count', 'tag']]
    
    def _save_to_file(self, queries_df):
        # save the queries
        if os.path.exists(self.target_fpath):
            bkp_fpath = backup_file(self.target_fpath)
            logger.info(f'backing up from={self.target_fpath} to={bkp_fpath}')
        
        queries_df.to_csv(self.target_fpath, index=False)
    
    def _log_stats(self, queries_df):
        num_pos_queries = queries_df.query('count > 0').shape[0]
        num_neg_queries = queries_df.query('count == 0').shape[0]
        logger.info(f'''
                    num_pos_queries={num_pos_queries} 
                    num_neg_queries={num_neg_queries} 
                    neg_percent={num_neg_queries/queries_df.shape[0]}
                ''')
        logger.info(queries_df['tag'].value_counts())
    
    def _balance_the_dataset(self, queries_df: pd.DataFrame, inplace: bool = True):
        neg_queries = queries_df.query('count == 0')
        num_pos_queries = queries_df.query('count > 0').shape[0]
        num_neg_queries = neg_queries.shape[0]
        pos_neg_ratio = round(num_pos_queries / num_neg_queries, 1)
        if pos_neg_ratio >= self.pos_neg_ratio:
            logger.info(f"nothing to balance, pos_neg_queries_ratio={pos_neg_ratio}")
            return queries_df
        
        num_neg_queries_req = int(num_pos_queries / self.pos_neg_ratio)
        num_neg_queries_del = num_neg_queries - num_neg_queries_req
        logger.info(f"discarding. num_neg_queries_del={num_neg_queries_del}, num_neg_queries_req={num_neg_queries_req}")
        
        neg_queries_del_idx = neg_queries.sample(num_neg_queries_del).index
        if inplace:
            queries_df.drop(neg_queries_del_idx, inplace=inplace, axis=0)
            queries_df.reset_index(drop=True, inplace=inplace)
        else:
            queries_df = queries_df.drop(neg_queries_del_idx, axis=0).reset_index(drop=True)
        return queries_df
    
    def build(self):
        ds = Yolov8Deserializer(
            dir_path=self.data_dir, 
            img_extns=['.jpg']).read()
        
        query_labels = self._load_queries(ds)
        data_df = self._load_df(ds)
        queries_df = self._prepare_queries(query_labels, data_df)
        
        # TODO: check if label_path is passed instead ?
        #   and pass the dataset reader to the TorchDataset
        queries_df['bbox_coords'] = ''
        for i in queries_df.query('count > 0 & tag in ["train", "valid"]').index:
            img_rel_path = queries_df.loc[i, 'image_relpath']
            _, img_name = os.path.split(img_rel_path)
            query_label =  queries_df.loc[i, 'query_label']
            bboxes = []
            for bbox in ds.get_bboxes(img_name=img_name):
                if bbox.label != query_label:
                    continue
                bboxes.extend([str(v) for v in bbox.to_array()])
            queries_df.loc[i, 'bbox_coords'] = ' '.join(bboxes)
            
        text_index = queries_df.query('count > 0 & tag == "test"').index
        queries_df.loc[text_index, 'bbox_coords'] = ''
        
        self._balance_the_dataset(queries_df)
        self._log_stats(queries_df)
        self._save_to_file(queries_df)
        return self.target_fpath, queries_df

class _Dataset(TorchDataset):
    def __init__(
        self, 
        data_dir: str,
        df: pd.DataFrame, 
        tag: str,
        is_reorder: bool = False, 
        reorder_size: int = 8, 
        random_state: Any = None, 
        transform: Any = None) -> None:
        
        super().__init__()
        self.random_state = random_state
        
        self.data_dir = data_dir
        self.data_index = df[df["tag"] == tag].index
        self.data_df = df
        # if is_reorder and reorder_size > 1:
        #     data_df = self._reorder_df(data_df, reorder_size)
        # self.data_df = data_df
        self.transform = transform
        self.random_state = random_state
    
    def _reorder_df(self, df: pd.DataFrame, reorder_size: int) -> pd.DataFrame:
        """
        reorder the df so every df has atleast one query with count > 0.
        the limitation is from the loss function
        """
        reord_idx = []
        df = df.sample(
            frac=1, axis=0, 
            random_state=self.random_state, 
            ignore_index=True)
        
        nonzero_index = df.query('count > 0').index
        start_idx = 0
        while start_idx < len(df):
            end_idx = min(start_idx + reorder_size, len(df))
            max_count = df['count'][start_idx:end_idx].max(axis=0)
            if max_count == 0:
                # TODO: check if we need to change the random choice to use random_state
                reord_idx.append(random.choice(nonzero_index))
                end_idx -= 1
            reord_idx.extend(list(range(start_idx, end_idx)))
            start_idx = end_idx
        reord_df = df.iloc[reord_idx]
        reord_df.reset_index(drop=True, inplace=True)
        return reord_df
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def _format_bbox_coords(self, bbox_count: int, bbox_coords: str) -> Tensor:
        if bbox_count == 0 or not isinstance(bbox_coords, str):
            return tutls.empty_float_tensor()
        
        coords = torch.as_tensor(list(map(np.float32, bbox_coords.split())))
        return coords.reshape(bbox_count, 4)
        
    def __getitem__(self, index) -> Tuple:
        item = self.data_df.loc[self.data_index[index]]
        
        # inputs
        image_path = os.path.join(self.data_dir, item['image_relpath'])
        image = PIL.Image.open(image_path)
        query = item['query_label']
        inputs = {
            "image": image,
            "query": query,
            "idx": tutls.to_int_tensor(self.data_index[index])
        }
        input = self.transform(inputs) if self.transform else inputs
        
        # targets
        count = tutls.to_int_tensor(item['count'])
        bbox_coords = self._format_bbox_coords(count, item['bbox_coords'])
        labels = tutls.to_int_tensor([0] * count) if count > 0 else tutls.empty_int_tensor()
        target = {
            "count": count,
            "labels": labels,
            "boxes": bbox_coords
        }
        return input, target

def _collate_fn(batch):
    image_tensors = []
    query_tensors = []
    idx_tensors = []
    
    count_tensors = []
    boxes_tensors = []
    labels_tensors = []
    for (input, target) in batch:
        image_tensors.append(input['image'])
        query_tensors.append(input['query'])
        idx_tensors.append(input['idx'])
        
        count_tensors.append(target['count'])
        labels_tensors.append(target['labels'])
        boxes_tensors.append(target['boxes'])
    
    inputs = {
        "image": torch.stack(image_tensors, dim=0),
        "query": torch.stack(query_tensors, dim=0),
        "idx": torch.stack(idx_tensors, dim=0),
    }
    
    targets = {
        "count": count_tensors,
        "labels": labels_tensors,
        "boxes": boxes_tensors
    }
    return (inputs, targets)
    
class LitInImageQuerierDM(L.LightningDataModule):
    def __init__(
        self, data_dir, csv_filepath, batch_size = 8, num_workers=0, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str) -> None:
        self.data_df = pd.read_csv(self.csv_filepath, dtype={'bbox_coords': str})
        self.train_ds = _Dataset(data_dir=self.data_dir, df=self.data_df, tag='train', transform=self.transform)
        self.val_ds = _Dataset(data_dir=self.data_dir, df=self.data_df, tag='valid', transform=self.transform)
        self.test_ds = _Dataset(data_dir=self.data_dir, df=self.data_df, tag='test', transform=self.transform)
        
        train_len = len(self.train_ds)
        val_len = len(self.val_ds)
        test_len = len(self.test_ds)
        total = len(self.data_df)
        logger.info(f'''
                train/val/test split, 
                len={train_len}/{val_len}/{test_len}, 
                ratio={round(train_len/total, 1)}/{round(val_len/total, 1)}/{round(test_len/total, 1)}
            ''')
    
    def train_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(
            self.train_ds, 
            self.batch_size, 
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True)
    
    def val_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(self.val_ds, self.batch_size, collate_fn=_collate_fn, num_workers=self.num_workers)
    
    def test_dataloader(self) -> TorchDataLoader:
        return TorchDataLoader(self.test_ds, self.batch_size, collate_fn=_collate_fn)
    
    def teardown(self, stage: str) -> None:
        # nothing to teardown
        pass