from ..config import BIN_S3_BUCKET
from ..config import IK_DATA_INDEX_FILENAME
from .. import resources as data
from .config import DataPrepConfig
from ..utils import get_default_on_none, DirtyMarker

from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from importlib import resources
from typing import Optional
from concurrent import futures
import boto3, os, logging

logger = logging.getLogger(__name__)

class BinS3DataDownloader:
    """
    Downloads the bin data from the S3 based on the IK shared index file.
    key method is `download`
    """
    
    def __init__(self, cfg: DataPrepConfig = None) -> None:
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.dirty_marker = DirtyMarker('downloader', self.cfg.raw_data_root_dir)

    def _prepare(self, target_dir : str = None):
        """
        creates necessary directories to download to.
        """
        meta_dir = self.cfg.rawdata_labels_dir
        images_dir = self.cfg.rawdata_images_dir
        if target_dir is not None and target_dir != self.cfg.raw_data_root_dir:
            meta_dir = os.path.join(
                target_dir, os.path.split(self.cfg.rawdata_labels_dir)[1])
            images_dir = os.path.join(
                target_dir, os.path.split(self.cfg.rawdata_images_dir)[1])
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        return images_dir, meta_dir

    def _validate(self):
        """
        validates if the IK shared index file exists
        """
        if not resources.files(data)\
            .joinpath(IK_DATA_INDEX_FILENAME).is_file():
                raise ValueError(f'.resources.{IK_DATA_INDEX_FILENAME} missing!')

    def _index_mod_time(self):
        resource = resources.files(data).joinpath(IK_DATA_INDEX_FILENAME)
        with resources.as_file(resource) as data_file:
            file_modtime = int(os.path.getmtime(data_file))
        return file_modtime
    
    def download(self, force: Optional[bool] = False, max_workers: int = 1):
        """
        downloads the bin data from s3 to local directories.
        images: ./s3/images
        metadata: ./s3/meta
        
        Args:
            force (`bool`, *optional*): on True discards last download and reloads. 
        """
        self._validate()
        
        image_dir, meta_dir = self._prepare()
        
        if (not force) and (not self.dirty_marker.is_dirty(self._index_mod_time)):
            return
        
        resource = resources.files(data).joinpath(IK_DATA_INDEX_FILENAME)
        with resources.as_file(resource) as data_file:
            with data_file.open(mode='r') as f:
                image_names = f.readlines()[1:]
        image_names = [x.strip() for x in image_names]

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        def download(img_name: str, img_fpath: str, meta_fpath: str) -> None:
            if not os.path.exists(img_fpath):
                with open(img_fpath, 'wb') as f:
                    s3.download_fileobj(BIN_S3_BUCKET, f'bin-images/{img_name}{self.cfg.raw_data_img_extn}', f)
            
            if not os.path.exists(meta_fpath):
                with open(meta_fpath, 'wb') as f:
                    s3.download_fileobj(BIN_S3_BUCKET, f'metadata/{img_name}.json', f)

        progress_step = len(image_names) // 10
        progress_bar = tqdm(
            total=len(image_names), 
            desc="downloading bin data", file=open(os.devnull, 'w'))
                            
        future_tasks = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_name in image_names:
                image_name = image_name.strip()
                target_image_file = os.path.join(image_dir, f'{image_name}{self.cfg.raw_data_img_extn}')
                target_metadata_file = os.path.join(meta_dir, f'{image_name}.json')
                
                future_tasks.append(executor.submit(
                    download, 
                    img_name=image_name, 
                    img_fpath=target_image_file, 
                    meta_fpath=target_metadata_file))
            
            logger.info(str(progress_bar))
            for future_task in futures.as_completed(future_tasks):
                progress_bar.update()
                if progress_bar.n >= progress_step:
                    progress_step += progress_bar.n
                    logger.info(str(progress_bar))
            logger.info(str(progress_bar))
            
        self.dirty_marker.mark()
        return None

def download(force: bool =False, max_workers: int = 1) -> None:
    """
    delegates the call to `BinS3DataDownloader.download`.
    """
    BinS3DataDownloader().download(force, max_workers=max_workers)