from binsense import config as cfg
from .dataset import BinDataset
from dataclasses import dataclass
import os

@dataclass
class DataPrepConfig:
    data_split_filepath = os.path.join(cfg.BIN_DATA_DIR, 'train_test_val_split_item.csv')
    data_split_images_dir = cfg.BIN_S3_DOWNLOAD_IMAGES_DIR
    root_dir = cfg.BIN_DATA_DIR
    label_chkpt_filepath = os.path.join(cfg.BIN_DATA_DIR, 'pre-annotate_labels.cpt')
    bbox_chkpt_filepath = os.path.join(cfg.BIN_DATA_DIR, 'pre-annotate_bboxes.cpt')
    robo_upload_dir = os.path.join(cfg.BIN_DATA_DIR, 'robo_upload')
    batch_size = 8
    dataset_class = BinDataset
    dataset_download_path = os.path.join(cfg.BIN_DATA_DIR, 'robo_download')
    rfmeta_file_path = os.path.join(cfg.BIN_DATA_DIR, 'robo_metadata.csv')