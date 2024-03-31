from .config import BIN_S3_DOWNLOAD_IMAGES_DIR as IMAGES_DIR
from .config import BIN_S3_DOWNLOAD_META_DIR as ANN_DIR
from .utils import default_on_none

from PIL import Image
from tqdm import tqdm 
from typing import Tuple

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
    
    def __init__(self) -> None:
        pass

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        
        bin_df = pd.DataFrame(columns=[
            'bin_id', 'bin_qty', 'bin_image_name', 
            'bin_image_kb', 'bin_image_width', 'bin_image_height'])
        
        item_df = pd.DataFrame(columns=[
            'bin_id', 'item_id', 'item_name', 'item_qty',
            'item_length', 'item_length_unit', 
            'item_width', 'item_width_unit', 
            'item_height', 'item_height_unit', 
            'item_weight', 'item_weight_unit'])

        ann_files = os.listdir(ANN_DIR)
        progress_step = len(ann_files) // 10
        ann_progress_iter = tqdm(
            ann_files, 
            desc="loading bin-metadata", file=open(os.devnull, 'w'))
        for i, f_name in enumerate(ann_progress_iter):
            meta_path = os.path.join(ANN_DIR, f_name)
            if not os.path.isfile(meta_path):
                continue

            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            bin_id = f_name[0:f_name.rfind('.')]
            bin_qty = metadata['EXPECTED_QUANTITY']
            bin_image_name = f'{bin_id}.jpg'
            img_path = os.path.join(IMAGES_DIR, bin_image_name)
            bin_img_kb = round(os.stat( img_path).st_size / 1024, 1)
            bin_img_width, bin_img_height = Image.open(img_path).size
            
            bin_df.loc[len(bin_df)] = [
                bin_id, bin_qty, bin_image_name,
                bin_img_kb, bin_img_width, bin_img_height]
            try:
                for item_id in metadata['BIN_FCSKU_DATA'].keys():
                    item_dict = metadata['BIN_FCSKU_DATA'][item_id]
                    item_name = default_on_none(item_dict, ['normalizedName'])
                    item_name = item_name if item_name else default_on_none(item_dict, ['name'])
                    item_df.loc[len(item_df)] = [
                        bin_id,
                        item_id, item_name, item_dict['quantity'],
                        default_on_none(item_dict, ['length', 'value'], float("nan")), 
                        default_on_none(item_dict, ['length', 'unit']),
                        default_on_none(item_dict, ['width','value'], float("nan")), 
                        default_on_none(item_dict, ['width', 'unit']),
                        default_on_none(item_dict, ['height', 'value'], float("nan")), 
                        default_on_none(item_dict, ['height', 'unit']),
                        default_on_none(item_dict, ['weight','value'], float("nan")), 
                        default_on_none(item_dict, ['weight', 'unit'])]
                if i == progress_step:
                    progress_step += i
                    logger.info(str(ann_progress_iter))
            except Exception as e:
                traceback.print_exc()
                logger.error(f"failed {bin_id}", exc_info=1)
        logger.info(str(ann_progress_iter))

        return bin_df, item_df

def load() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return BinMetadataLoader().load()