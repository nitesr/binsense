from .config import BIN_S3_DOWNLOAD_DIR
from .config import BIN_S3_DOWNLOAD_IMAGES_DIR
from .config import BIN_S3_DOWNLOAD_META_DIR
from .config import BIN_S3_BUCKET
from .config import IK_DATA_INDEX_FILENAME
from . import resources as data

from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from importlib import resources
from typing import Optional

import boto3, os, time

# TODO: seperate the dirty marker into its own class
#   can be resued when preparing subsequent data transformations
MARK_FILE_PATH = os.path.join(BIN_S3_DOWNLOAD_DIR, 'downloader_mark.dat')
class BinS3DataDownloader:
    """
    Downloads the bin data from the S3 based on the IK shared index file.
    key method is `download`
    """
    
    def __init__(self) -> None:
        pass

    def _prepare(self):
        """
        creates necessary directories to download to.
        """
        images_dir = BIN_S3_DOWNLOAD_IMAGES_DIR
        meta_dir = BIN_S3_DOWNLOAD_META_DIR
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

    def _validate(self):
        """
        validates if the IK shared index file exists
        """
        if not resources.files(data)\
            .joinpath(IK_DATA_INDEX_FILENAME).is_file():
                raise ValueError(f'.resources.{IK_DATA_INDEX_FILENAME} missing!')

    def _mark(self):
        """
        marks the current downloaded time
        """
        with open(MARK_FILE_PATH, 'w+') as f:
            f.write(str(int(time.time())))

    def _read_mark(self):
        """
        reads the last downloaded time
        """
        if not os.path.exists(MARK_FILE_PATH):
            # return oldest epoch
            return 0 
        
        with open(MARK_FILE_PATH, 'r') as f:
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
        
    def download(self, force: Optional[bool] = False):
        """
        downloads the bin data from s3 to local directories.
        images: ./s3/images
        metadata: ./s3/meta
        
        Args:
            force (`bool`, *optional*): on True discards last download and reloads. 
        """
        self._validate()
        
        self._prepare()
        
        if (not force) and (not self._is_dirty()):
            return
        
        resource = resources.files(data).joinpath(IK_DATA_INDEX_FILENAME)
        with resources.as_file(resource) as data_file:
            with data_file.open(mode='r') as f:
                image_names = f.readlines()[1:]
        image_names = [x.strip() for x in image_names]

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        for image_name in tqdm(image_names, desc="downloading bin data"):
            image_name = image_name.strip()
            target_image_file = os.path.join(BIN_S3_DOWNLOAD_IMAGES_DIR, f'{image_name}.jpg')
            target_metadata_file = os.path.join(BIN_S3_DOWNLOAD_META_DIR, f'{image_name}.json')
            
            if not os.path.exists(target_image_file):
                with open(target_image_file, 'wb') as f:
                    s3.download_fileobj(BIN_S3_BUCKET, f'bin-images/{image_name}.jpg', f)
            
            if not os.path.exists(target_metadata_file):
                with open(target_metadata_file, 'wb') as f:
                    s3.download_fileobj(BIN_S3_BUCKET, f'metadata/{image_name}.json', f)
        
        self._mark()

def download(force: bool =False) -> None:
    """
    delegates the call to `BinS3DataDownloader.download`.
    """
    BinS3DataDownloader().download(force)