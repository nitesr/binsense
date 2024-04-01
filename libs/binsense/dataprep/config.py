from binsense import config as cfg
from .dataset import BinDataset
from dataclasses import dataclass
import os, re

@dataclass
class DataPrepConfig:
    
    raw_data_root_dir = cfg.BIN_S3_DOWNLOAD_DIR
    data_split_images_dir = os.path.join(cfg.BIN_S3_DOWNLOAD_DIR, 'images')
    data_split_labels_dir = os.path.join(cfg.BIN_S3_DOWNLOAD_DIR, 'metadata')
    raw_data_img_extn = '.jpg'
    
    data_split_filepath = os.path.join(cfg.BIN_DATA_DIR, 'train_test_val_split_item.csv')
    root_dir = cfg.BIN_DATA_DIR
    label_chkpt_filepath = os.path.join(cfg.BIN_DATA_DIR, 'pre-annotate_labels.cpt')
    bbox_chkpt_filepath = os.path.join(cfg.BIN_DATA_DIR, 'pre-annotate_bboxes.cpt')
    robo_upload_dir = os.path.join(cfg.BIN_DATA_DIR, 'robo_upload')
    batch_size = 8
    dataset_class = BinDataset
    dataset_download_path = os.path.join(cfg.BIN_DATA_DIR, 'robo_download')
    rfmeta_file_path = os.path.join(cfg.BIN_DATA_DIR, 'robo_metadata.csv')
    embed_store_dirpath = os.path.join(cfg.BIN_DATA_DIR, 'embed_store')
    
    roboql_dataset_url='https://app.roboflow.com/query/roboql/dataset'
    robo_workspace = "nitesh-c-eszzc"
    robo_project = "binsense_bbox_mini"
    robo_ann_group  ="bins"
    robo_dataset_version = 2
    robo_workspace_id = "eEYy11ONL8bl5ZlkZHqx7JE1SF92"
    robo_project_id = "PedxdvWMh6obbp6GmpGm"
    robo_meta_check_tags = [
        ('jithu', re.compile('jithu[-0-9]{0,}')),
        ('mythili', re.compile('mythili[-0-9]{0,}')),
        ('nitesh', re.compile('nitesh[-0-9]{0,}')),
        ('raghu', re.compile('nitesh[-0-9]{0,}')),
        ('adjusted', re.compile('adjusted')),
        ('assumed', re.compile('(assume[d]{0,1}|assumption)')),
        ('blurry', re.compile('(blurry|blurred|blur)')),
        ('done', re.compile('(done|dome|donne|completed|complete|finished|finish)')),
        ('hard', re.compile('(hard|hardy|very hard|hardest)'))
    ]