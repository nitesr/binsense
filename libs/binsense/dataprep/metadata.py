from ..utils import default_on_none, get_default_on_none
from .config import DataPrepConfig

from PIL import Image
from tqdm import tqdm 
from typing import Tuple, List, Any
from concurrent import futures

import json, traceback, os, logging
import pandas as pd

logger = logging.getLogger(__name__)

class BinMetadataLoader:
    """
    Loads the downloaded bin data to pandas. \
    Expects the images and metadata to be available locally. \
    check BinS3DataDownloader to download the data. \
    key method is `load`
    """
    
    def __init__(self, cfg: DataPrepConfig = None) -> None:
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.bin_csv_fname = os.path.split(self.cfg.rawdata_bin_csv_filepath)[1]
        self.item_csv_name = os.path.split(self.cfg.rawdata_items_csv_filepath)[1]
    
    def _dump_to_csv(self, img_dir: str, ann_dir: str, max_workers:int = 1):
        
        def load_json_file(ann_fname: str) -> None:
            meta_path = os.path.join(ann_dir, ann_fname)
            # each file is at max 2KB, load all file into buffer
            with open(meta_path, 'r', buffering=2*1024) as f:
                metadata_json = json.load(f)
            
            bin_id = ann_fname[0:ann_fname.rfind('.')]
            bin_image_name = f'{bin_id}{self.cfg.raw_data_img_extn}'
            bin_img_path = os.path.join(img_dir, bin_image_name)
            bin_img_exists = os.path.exists(bin_img_path)
            bin_img_width, bin_img_height = Image.open(bin_img_path).size if bin_img_exists else (0, 0)
            bin_obj = {
                "bin_id" : bin_id,
                "bin_qty" : metadata_json['EXPECTED_QUANTITY'],
                "bin_image_name" : bin_image_name,
                "bin_img_exists": bin_img_exists,
                "bin_image_kb": round(os.stat( bin_img_path).st_size / 1024, 1) if bin_img_exists else 0,
                "bin_image_width": bin_img_width,
                "bin_image_height": bin_img_height
            }
            bin_item_objs = []
            try:
                for item_id in metadata_json['BIN_FCSKU_DATA'].keys():
                    item_dict = metadata_json['BIN_FCSKU_DATA'][item_id]
                    item_name = default_on_none(item_dict, ['normalizedName'])
                    item_name = item_name if item_name else default_on_none(item_dict, ['name'])
                    bin_item_obj = {
                        "bin_id": bin_id,
                        "item_id": item_id,
                        "item_name": item_name,
                        "item_qty": item_dict['quantity'],
                        'item_length': default_on_none(item_dict, ['length', 'value'], float("nan")),
                        'item_length_unit': default_on_none(item_dict, ['length', 'unit']),
                        'item_width': default_on_none(item_dict, ['width','value'], float("nan")),
                        'item_width_unit': default_on_none(item_dict, ['width', 'unit']),
                        'item_height': default_on_none(item_dict, ['height', 'value'], float("nan")),
                        'item_height_unit': default_on_none(item_dict, ['height', 'unit']),
                        'item_weight': default_on_none(item_dict, ['weight','value'], float("nan")),
                        'item_weight_unit': default_on_none(item_dict, ['weight', 'unit'])
                    }
                    bin_item_objs.append(bin_item_obj)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"failed {bin_id}", exc_info=1)
            
            return bin_obj, bin_item_objs
        
        ann_files = os.listdir(ann_dir)
        task_progress_step = len(ann_files) // 10
        task_progress_bar = tqdm(
            total=len(ann_files), 
            desc="creating bin-metadata load tasks", file=open(os.devnull, 'w'))
        load_progress_step = len(ann_files) // 10
        load_progress_bar = tqdm(
            total=len(ann_files), 
            desc="loading bin-metadata", file=open(os.devnull, 'w'))
        
        logger.info(str(task_progress_bar))
        future_tasks = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for f_name in ann_files:
                meta_path = os.path.join(ann_dir, f_name)
                if not os.path.isfile(meta_path):
                    continue
                
                future_tasks.append(executor.submit(
                    load_json_file, 
                    ann_fname=f_name))
                task_progress_bar.update()
                if task_progress_bar.n >= task_progress_step:
                    task_progress_step += task_progress_bar.n
                    logger.info(str(task_progress_bar))
            logger.info(str(task_progress_bar))
            
            def to_csv_line(values: List) -> str:
                # escape csv special chars
                str_values = [v.replace('"', '""').replace('\\', '\\\\') if isinstance(v, str) else v for v in values]
                str_values = [f'"{v}"' if isinstance(v, str) else str(v) for v in str_values]
                return ','.join(str_values)
            
            bin_csv_values = []
            item_csv_values = []
            logger.info(str(load_progress_bar))
            for future_task in futures.as_completed(future_tasks):
                bin_obj, bin_item_objs = future_task.result()
                
                if len(bin_csv_values) == 0:
                    bin_csv_values.append(to_csv_line(bin_obj.keys()))
                bin_csv_values.append(to_csv_line(bin_obj.values()))
                
                for bin_item_obj in bin_item_objs:
                    if len(item_csv_values) == 0:
                        item_csv_values.append(to_csv_line(bin_item_obj.keys()))
                    item_csv_values.append(to_csv_line(bin_item_obj.values()))
                
                load_progress_bar.update()
                if load_progress_bar.n >= load_progress_step:
                    load_progress_step += load_progress_bar.n
                    logger.info(str(load_progress_bar))
            logger.info(str(load_progress_bar))
            
            parentdir, _ = os.path.split(ann_dir)
            bin_csv_path = os.path.join(parentdir, self.bin_csv_fname)
            items_csv_path = os.path.join(parentdir, self.item_csv_name)
            logger.info(f"writing csv files, bin_csv_path={bin_csv_path}, items_csv_path={items_csv_path}")
            with open(bin_csv_path, 'w') as f:
                f.writelines([f'{v}\n' for v in bin_csv_values])
            with open(items_csv_path, 'w') as f:
                f.writelines([f'{v}\n' for v in item_csv_values])
            return bin_csv_path, items_csv_path
    
    
    def load(self, source_dir : str = None, max_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the downloaded bin meta-data to pandas dataframes.
        
        Returns:
            `Tuple[pandas.DataFrame, pandas.DataFrame]`: bin & item metadata panda DataFrames.
            bin `DataFrame` columns = ['bin_id', 'bin_qty', 'bin_image_name', \
                'bin_img_exists', 'bin_image_kb', 'bin_image_width', 'bin_image_height']
            item `DataFrame` columns = [ 'bin_id', 'item_id', 'item_name', \
                'item_qty', 'item_length', 'item_length_unit', 'item_width', \
                'item_width_unit', 'item_height', 'item_height_unit', \
                'item_weight', 'item_weight_unit']
        """
        ann_dir = self.cfg.rawdata_labels_dir
        img_dir = self.cfg.rawdata_images_dir
        
        if source_dir is not None and source_dir != self.cfg.raw_data_root_dir:
            ann_dir = os.path.join(
                source_dir, os.path.split(self.cfg.rawdata_labels_dir)[1])
            img_dir = os.path.join(
                source_dir, os.path.split(self.cfg.rawdata_images_dir)[1])
        
        if source_dir is None:
            source_dir = self.cfg.raw_data_root_dir
        
        bin_csv_path = os.path.join(source_dir, self.bin_csv_fname)
        items_csv_path = os.path.join(source_dir, self.item_csv_name)
        if not os.path.exists(bin_csv_path) \
            or not os.path.exists(items_csv_path):
            logger.info('generating csv files')
            self._dump_to_csv(img_dir, ann_dir, max_workers)
        
        logger.info('reading csv files')
        bin_df = pd.read_csv(bin_csv_path, header=0, dtype={'bin_id': str})
        item_df = pd.read_csv(items_csv_path, header=0, dtype={'item_id': str, 'bin_id': str})
        
        return bin_df, item_df

def load(source_dir: str = None, max_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return BinMetadataLoader().load(source_dir, max_workers)