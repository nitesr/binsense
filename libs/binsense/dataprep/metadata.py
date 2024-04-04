from ..utils import default_on_none, get_default_on_none
from .config import DataPrepConfig

from PIL import Image
from tqdm import tqdm 
from typing import Tuple, Any
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

    def load(self, source_dir : str = None, max_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the downloaded bin meta-data to pandas dataframes.
        
        Returns:
            `Tuple[pandas.DataFrame, pandas.DataFrame]`: bin & item metadata panda DataFrames.
            bin `DataFrame` columns = ['bin_id', 'bin_qty', 'bin_image_name', \
                'bin_image_kb', 'bin_image_width', 'bin_image_height']
            item `DataFrame` columns = [ 'bin_id', 'item_id', 'item_name', \
                'item_qty', 'item_length', 'item_length_unit', 'item_width', \
                'item_width_unit', 'item_height', 'item_height_unit', \
                'item_weight', 'item_weight_unit']
        """
        ann_dir = self.cfg.data_split_labels_dir
        img_dir = self.cfg.data_split_images_dir
        
        if source_dir is not None and source_dir != self.cfg.raw_data_root_dir:
            ann_dir = os.path.join(
                source_dir, os.path.split(self.cfg.data_split_labels_dir)[1])
            img_dir = os.path.join(
                source_dir, os.path.split(self.cfg.data_split_images_dir)[1])
        
        bin_df = pd.DataFrame(columns=[
            'bin_id', 'bin_qty', 'bin_image_name', 
            'bin_image_kb', 'bin_image_width', 'bin_image_height'])
        
        item_df = pd.DataFrame(columns=[
            'bin_id', 'item_id', 'item_name', 'item_qty',
            'item_length', 'item_length_unit', 
            'item_width', 'item_width_unit', 
            'item_height', 'item_height_unit', 
            'item_weight', 'item_weight_unit'])

        ann_files = os.listdir(ann_dir)
        progress_step = len(ann_files) // 10
        progress_bar = tqdm(
            total=len(ann_files), 
            desc="loading bin-metadata", file=open(os.devnull, 'w'))
        logger.info(str(progress_bar))
        
        def load_json_file(ann_fname: str) -> None:
            meta_path = os.path.join(ann_dir, ann_fname)
            with open(meta_path, 'r') as f:
                metadata_json = json.load(f)
            
            bin_id = ann_fname[0:ann_fname.rfind('.')]
            bin_image_name = f'{bin_id}{self.cfg.raw_data_img_extn}'
            bin_img_path = os.path.join(img_dir, bin_image_name)
            bin_img_width, bin_img_height = Image.open(bin_img_path).size
            bin_obj = {
                "bin_id" : bin_id,
                "bin_qty" : metadata_json['EXPECTED_QUANTITY'],
                "bin_image_name" : bin_image_name,
                "bin_image_kb": round(os.stat( bin_img_path).st_size / 1024, 1),
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
        
        future_tasks = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for f_name in ann_files:
                meta_path = os.path.join(ann_dir, f_name)
                if not os.path.isfile(meta_path):
                    continue

                future_tasks.append(executor.submit(
                    load_json_file, 
                    ann_fname=f_name))
                
        for future_task in futures.as_completed(future_tasks):
            bin_obj, bin_item_objs = future_task.result()
            bin_df.loc[len(bin_df)] = bin_obj
            for bin_item_obj in bin_item_objs:
                item_df.loc[len(item_df)] = bin_item_obj
            progress_bar.update()
            if progress_bar.n >= progress_step:
                progress_step += progress_bar.n
                logger.info(str(progress_bar))
        logger.info(str(progress_bar))

        return bin_df, item_df

def load(source_dir: str = None, max_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return BinMetadataLoader().load(source_dir, max_workers)