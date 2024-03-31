from ..dataprep.roboflow_util import RoboflowCrawler
from ..dataprep.roboflow_util import RoboflowDownloader
from ..dataprep.config import DataPrepConfig
from ..config import BIN_DATA_DIR

import argparse
import logging

def run_metadata_crawler(cookie_str: str, workspace_id: str, project_id: str, ann_group: str):
    cfg = DataPrepConfig()
    crawler = RoboflowCrawler(
        project_id, workspace_id, cookie_str, 
        cfg.root_dir, ann_group)
    fpath = crawler.crawl()
    print('roboflow metadata downloaded at', fpath)

def run_download_dataset(api_key: str):
    downloader = RoboflowDownloader(api_key=api_key)
    dirpath = downloader.download(DataPrepConfig().root_dir)
    print('roboflow dataset downloaded at', dirpath)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="download metadata and dataset",
        action="store_true")
    parser.add_argument(
        "--cookie_str", help="cookie string from https://app.roboflow.com/ in browser after signing in.")
    parser.add_argument(
        "--workspace_id", help="workspace id from https://app.roboflow.com/ in browser after signing in.",
        default="eEYy11ONL8bl5ZlkZHqx7JE1SF92", required=False)
    parser.add_argument(
        "--project_id", help="project id from https://app.roboflow.com/ in browser after signing in.",
        default="PedxdvWMh6obbp6GmpGm", required=False)
    parser.add_argument(
        "--ann_group", help="annotationGroup from https://app.roboflow.com/ in browser after signing in.",
        default="bins", required=False)
    parser.add_argument(
        "--api_key", help="roboflow api key", required=False)
    
    args = parser.parse_args()
    
    if args.download:
        print("running meta crawler..")
        run_metadata_crawler(
            args.cookie_str, args.workspace_id, 
            args.project_id, args.ann_group)
        
        print("downloading dataset..")
        run_download_dataset(args.api_key)