from .config import DataPrepConfig
from .dataset import BinDataset
from .model_util import BboxPredictor
from ..utils import backup_file
from ..dataset_util import YoloDatasetBuilder, DataTag

from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import os, torch, logging, shutil

logger = logging.getLogger(__name__)

class TrainsetLoader:
    def __init__(self, config: DataPrepConfig) -> None:
        self.cfg = config
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        req_attrs = ['image_name', 'tag', 'bbox_label', 'bbox_count']
        for attr in req_attrs:
            if not attr in df.columns:
                raise ValueError(f'{attr} is not in dataset')
            
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        file_path = self.cfg.data_split_filepath
        item_df = pd.read_csv(file_path, dtype={'image_name': str})
        self._validate_dataset(item_df)
        
        # collect all labels before filtering
        labels = item_df["bbox_label"].sort_values(axis=0).unique()
        
        # filter for train/valid data and get image_name
        data_item_df = item_df[item_df["tag"] != "test"]
        data_bin_df = data_item_df[["image_name", "tag"]].drop_duplicates(ignore_index=True)
        return data_bin_df, data_item_df, labels

# TODO: 
#   This class loads all the data and then prepares
#   change it to process in chunks
class Preannotator:
    def __init__(
        self, 
        bbox_predictor: BboxPredictor,
        config: DataPrepConfig,
        test_run: False
        ) -> None:
        
        self.model = bbox_predictor
        self.cfg = config
        self.ts_loader = TrainsetLoader(config)
        self.test_run = test_run
    
    def _prepare_chkpt_files(self) -> None:
        if os.path.exists(self.cfg.label_chkpt_filepath):
            bkp_filepath = backup_file(self.cfg.label_chkpt_filepath)
            logger.info(f"backed up {self.cfg.label_chkpt_filepath} as {bkp_filepath}")
        
        if os.path.exists(self.cfg.bbox_chkpt_filepath):
            bkp_filepath = backup_file(self.cfg.bbox_chkpt_filepath)
            logger.info(f"backed up {self.cfg.bbox_chkpt_filepath} as {bkp_filepath}")

    def _save_labels(self, labels: List[str]) -> None:
        lblchkpoint_file = self.cfg.label_chkpt_filepath
        with open(lblchkpoint_file, 'w') as f:
            f.write('\n'.join(labels))
    
    def _predict_bboxes(self, x):
        with torch.no_grad():
            _, scores, bboxes = self.model(x)
            scores = scores.numpy()
            bboxes = bboxes.numpy()
        return scores, bboxes
    
    def _save_result(self, record_idx, bin_df, item_df, label_dict, scores, bboxes):
        image_name = bin_df.iloc[record_idx]["image_name"]
        
        with open(self.cfg.bbox_chkpt_filepath, 'a') as f:
            df = item_df[item_df['image_name'] == image_name][["image_name", "bbox_count", "bbox_label"]]
            bboxes_count = df["bbox_count"].sum()
            f.write(' '.join([image_name, str(bboxes_count)]))
            f.write(' ')
            
            bbox_idx = 0
            for bbox_count, bbox_label in zip(df["bbox_count"], df["bbox_label"]):
                tmp_label_id = label_dict[bbox_label]
                for _ in range(bbox_count):
                    f.write(' '.join([str(tmp_label_id), str(scores[bbox_idx]), *[str(b) for b in bboxes[bbox_idx, :]]]))
                    f.write(' ')
                    bbox_idx += 1
            f.write('\n')
    
    def preannotate(self, batch_size=None):
        
        if batch_size is None:
            batch_size = self.cfg.batch_size
        
        bin_df, item_df, labels =  self.ts_loader.load()
        self._prepare_chkpt_files()
        
        self._save_labels(labels)
        image_names = bin_df['image_name'].tolist()
        image_names = image_names[0:10] if self.test_run else image_names
        train_ds = BinDataset(image_names, images_dir=self.cfg.data_split_images_dir)
        train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True)
        
        
        label_dict = YoloDatasetBuilder().add_categories(labels)
        chkpoint_file = os.path.join(self.cfg.bbox_chkpt_filepath)
        with open(chkpoint_file, 'w') as f:
            f.write('# preannotated bounding boxes format - image_name bboxes_count *[bbox_label_id bbox_score center_x center_y width height]')
            f.write('\n')
        
        self.model.eval()
        record_idx = 0
        dl_progress_iter = tqdm(train_dl, desc="predicting bboxes", file=open(os.devnull, 'w'))
        for x in dl_progress_iter:
            scores, bboxes = self._predict_bboxes(x)
            for i in range(0, len(bboxes)):
                self._save_result(record_idx, bin_df, item_df, label_dict, scores[i], bboxes[i])
                record_idx += 1
            logger.info(str(dl_progress_iter))
        
        return self.cfg.label_chkpt_filepath, self.cfg.bbox_chkpt_filepath

class RoboflowUploadBuilder:
    def __init__(self, config=DataPrepConfig) -> None:
        self.cfg = config
        self.ts_loader = TrainsetLoader(config)
    
    def _load_labels_from_chkpoint(self):
        lblchkpoint_file = os.path.join(self.cfg.label_chkpt_filepath)
        
        with open(lblchkpoint_file, 'r') as f:
            labels = f.readlines()
            # to strip the new line at the end
            labels = [l[:-1] for l in labels]
            return labels

    def _interpret_preannotate_line(self, line: str, bin_df, ds_builder: YoloDatasetBuilder):
        tokens = [ t.strip() for t in line.split()]
        image_name = tokens[0]
        tag = bin_df[bin_df["image_name"] == image_name]["tag"].values[0]
        img_path = os.path.join(self.cfg.data_split_images_dir, image_name)
        img_id = ds_builder.add_image(img_path, DataTag(tag))
        
        #i = 1 is binqty
        i = 2
        while i < len(tokens):
            #i(label) i+1(score) i+2(center_x) i+3(center_y) i+4(width) i+5(height)
            ds_builder.add_bbox(
                img_id, int(tokens[i]),
                np.array([float(tokens[i+2]), float(tokens[i+3]), float(tokens[i+4]), float(tokens[i+5])])
            )
            i += 6
    
    def _prepare_uploaddir(self, target_dir: str):
        upload_dir = target_dir
        if os.path.exists(upload_dir):
            backup_file(upload_dir)
        os.makedirs(upload_dir)
        return upload_dir
    
    def build(self, dir_path: str = None) -> str:
        if dir_path is None:
            dir_path = self.cfg.robo_upload_dir
        
        _, item_df, _ = self.ts_loader.load()
        bin_df = item_df[["image_name", "tag"]].drop_duplicates(ignore_index=True)
        
        labels = self._load_labels_from_chkpoint()
        ds_builder = YoloDatasetBuilder()
        ds_builder.add_categories(labels)
        
        linecnt = 0
        chkpoint_file = os.path.join(self.cfg.bbox_chkpt_filepath)
        with open(chkpoint_file, 'r') as f:
            line = f.readline()
            while line:
                if not line.startswith('#'):
                    self._interpret_preannotate_line(line, bin_df, ds_builder)
                    linecnt += 1
                line = f.readline()
        
        logger.info(f'processed {linecnt} records and {len(labels)} labels.')
        upload_dir = self._prepare_uploaddir(dir_path)
        ds_builder.build().to_file(upload_dir, format='v8')
        return upload_dir