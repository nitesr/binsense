from ..config import BIN_S3_BUCKET
from ..config import IK_DATA_INDEX_FILENAME
from .. import resources as data
from .config import DataPrepConfig
from ..utils import get_default_on_none

from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from importlib import resources
from typing import Optional
from concurrent import futures
import boto3, os, time

class BinS3DataDownloader:
    """
    Downloads the bin data from the S3 based on the IK shared index file.
    key method is `download`
    """
    
    def __init__(self, cfg: DataPrepConfig = None) -> None:
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.mark_fpath = os.path.join(self.cfg.raw_data_root_dir, 'downloader_mark.dat')

    def _prepare(self, target_dir : str = None):
        """
        creates necessary directories to download to.
        """
        meta_dir = self.cfg.data_split_labels_dir
        images_dir = self.cfg.data_split_images_dir
        if target_dir is not None and target_dir != self.cfg.raw_data_root_dir:
            meta_dir = os.path.join(
                target_dir, os.path.split(self.cfg.data_split_labels_dir)[1])
            images_dir = os.path.join(
                target_dir, os.path.split(self.cfg.data_split_images_dir)[1])
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

    # TODO: seperate the dirty marker into its own class
    #   can be resued when preparing subsequent data transformations
    def _mark(self):
        """
        marks the current downloaded time
        """
        with open(self.mark_fpath, 'w+') as f:
            f.write(str(int(time.time())))

    def _read_mark(self):
        """
        reads the last downloaded time
        """
        if not os.path.exists(self.mark_fpath):
            # return oldest epoch
            return 0 
        
        with open(self.mark_fpath, 'r') as f:
            dt = f.readline()
            return int(dt)

    def _is_dirty(self):
        """
        checks if last download is latest based on timestamp on IK shared index file.
        """
        downloaded_time = self._read_mark()
        resource = resources.files(data).joinpath(IK_DATA_INDEX_FILENAME)
        with resources.as_file(resource) as data_file:
            file_modtime = int(os.path.getmtime(data_file))
            return downloaded_time < file_modtime
        
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
        
        if (not force) and (not self._is_dirty()):
            return
        
        resource = resources.files(data).joinpath(IK_DATA_INDEX_FILENAME)
        with resources.as_file(resource) as data_file:
            with data_file.open(mode='r') as f:
                image_names = f.readlines()[1:]
        image_names = [x.strip() for x in image_names]

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        future_tasks = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_name in tqdm(image_names, desc="downloading bin data"):
                image_name = image_name.strip()
                target_image_file = os.path.join(image_dir, f'{image_name}{self.cfg.raw_data_img_extn}')
                target_metadata_file = os.path.join(meta_dir, f'{image_name}.json')
                
                def download(img_name: str, img_fpath: str, meta_fpath: str) -> None:
                    if not os.path.exists(img_fpath):
                        with open(img_fpath, 'wb') as f:
                            s3.download_fileobj(BIN_S3_BUCKET, f'bin-images/{img_name}{self.cfg.raw_data_img_extn}', f)
                    
                    if not os.path.exists(meta_fpath):
                        with open(meta_fpath, 'wb') as f:
                            s3.download_fileobj(BIN_S3_BUCKET, f'metadata/{img_name}.json', f)
                
                    future_tasks.append(executor.submit(
                        download, 
                        img_name=image_name, 
                        img_fpath=target_image_file, 
                        meta_fpath=target_metadata_file))
            futures.wait(future_tasks)
        self._mark()

def download(force: bool =False, max_workers: int = 1) -> None:
    """
    delegates the call to `BinS3DataDownloader.download`.
    """
    BinS3DataDownloader().download(force, max_workers=max_workers)