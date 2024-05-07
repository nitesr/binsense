import os
from pathlib import Path

# TODO: Make this configuration configurable by environment
ROOT_DIR_NAME = 'binsense'
cur_path = Path(os.path.abspath('.'))
dir_name = cur_path.name
while cur_path.parent != cur_path.parent.parent and cur_path.name != ROOT_DIR_NAME:
    cur_path = cur_path.parent

PROJ_DIR = str(cur_path) if ROOT_DIR_NAME in str(cur_path) else os.path.abspath('.')
MODEL_DIR = PROJ_DIR+'/_models'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
LOGS_DIR = PROJ_DIR+'/_logs'

BIN_DATA_DIR = os.path.join(DATA_DIR, 'bin')
BIN_S3_DOWNLOAD_DIR = os.path.join(BIN_DATA_DIR, 's3')

BIN_S3_DOWNLOAD_IMAGES_DIR = os.path.join(BIN_S3_DOWNLOAD_DIR, 'images')
BIN_S3_DOWNLOAD_META_DIR = os.path.join(BIN_S3_DOWNLOAD_DIR, 'metadata')

BIN_S3_BUCKET = 'aft-vbi-pds'
IK_DATA_INDEX_FILENAME = 'ik_data_index.csv'

BIN_ROBO_WORKSPACE = 'nitesh-c-eszzc'
BIN_ROBO_PROJECT = "binsense_segments"
BIN_ROBO_DOWNLOAD_DIR = os.path.join(DATA_DIR, 'robo')

COCO_DATA_DIR = os.path.join(DATA_DIR, 'coco_2017')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(BIN_DATA_DIR, exist_ok=True)
os.makedirs(BIN_S3_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(BIN_S3_DOWNLOAD_IMAGES_DIR, exist_ok=True)
os.makedirs(BIN_S3_DOWNLOAD_META_DIR, exist_ok=True)
os.makedirs(BIN_ROBO_DOWNLOAD_DIR, exist_ok=True)

