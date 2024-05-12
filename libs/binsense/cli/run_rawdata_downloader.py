from ..dataprep.metadata import BinMetadataLoader
from ..dataprep.downloader import BinS3DataDownloader
from ..dataprep.config import DataPrepConfig
from ..utils import load_params, get_default_on_none

import logging, os, argparse, sys

logger = logging.getLogger(__name__)

def download(cfg: DataPrepConfig, force_download=False, num_workers: int = 1) -> bool:
    BinS3DataDownloader(cfg).download(force=force_download, max_workers=num_workers)
    _, _ = BinMetadataLoader(cfg).load(max_workers=num_workers)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    params = load_params('./params.yaml')

    parser.add_argument(
        "--force", help="donot consider the cache, download from the s3",
        action="store_true")

    parser.add_argument(
        "--num_workers", help="num of workers to prepare the data",
        type=int, default=None)
    
    args = parser.parse_args()
    download(
        cfg=cfg, 
        force_download=args.force, 
        num_workers=get_default_on_none(args.num_workers, params.data_download.num_workers)
    )
    
    sys.exit(0)


