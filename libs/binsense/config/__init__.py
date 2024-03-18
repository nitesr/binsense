import os
from pathlib import Path

ROOT_DIR_NAME = 'binsense'
cur_path = Path(os.path.abspath('.'))
dir_name = cur_path.name
while cur_path.parent != cur_path.parent.parent and cur_path.name != ROOT_DIR_NAME:
    cur_path = cur_path.parent

PROJ_DIR = str(cur_path) if ROOT_DIR_NAME in str(cur_path) else os.path.abspath('.')
MODEL_DIR = PROJ_DIR+'/_models'
DATA_DIR = PROJ_DIR+'/_data'
LOGS_DIR = PROJ_DIR+'/_logs'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)