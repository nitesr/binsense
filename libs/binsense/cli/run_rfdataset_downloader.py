from ..dataprep.roboflow_util import RoboflowCrawler
from ..dataprep.roboflow_util import RoboflowDownloader
from ..dataprep.config import DataPrepConfig
from ..config import BIN_DATA_DIR

import logging, os, argparse

def run_metadata_crawler(
    target_dir: str = None,
    cookie_str: str = None, 
    workspace_id: str = None, 
    project_id: str = None, 
    ann_group: str = None):
    
    cfg = DataPrepConfig()
    target_path = None
    if target_dir is not None and target_dir != cfg.root_dir:
        target_path = os.path.join(
            target_dir, os.path.split(cfg.rfmeta_file_path)[1])
    
    crawler = RoboflowCrawler(
        project_id, workspace_id, cookie_str, 
        target_path, ann_group, cfg=cfg)
    fpath = crawler.crawl()
    print('roboflow metadata downloaded at', fpath)

def run_download_dataset(
    target_dir: str = None,
    workspace: str = None, 
    project: str = None, 
    version: int = None, 
    api_key: str = None):
    
    cfg = DataPrepConfig()
    if target_dir == cfg.root_dir:
        target_dir = None # default to the ones in cfg
    elif target_dir is not None:
        target_dir = os.path.join(
            target_dir, os.path.split(cfg.dataset_download_path)[1])
    
    downloader = RoboflowDownloader(
        api_key=api_key, workspace=workspace, 
        project=project, version=version, 
        cfg=cfg)
    dirpath = downloader.download(target_dir=target_dir)
    print('roboflow dataset downloaded at', dirpath)

def run_validator():
    pass
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    rfmeta_fname = os.path.split(cfg.rfmeta_file_path)
    download_dirname = os.path.split(cfg.dataset_download_path)
    
    parser.add_argument(
        "--download", help="download metadata and dataset",
        action="store_true")
    parser.add_argument(
        "--cookie_str", help="cookie string from https://app.roboflow.com/ in browser after signing in.")
    parser.add_argument(
        "--workspace_id", help="workspace id from https://app.roboflow.com/ in browser after signing in.",
        default="eEYy11ONL8bl5ZlkZHqx7JE1SF92", required=False)
    parser.add_argument(
        "--workspace", help="workspace name from https://app.roboflow.com/",
        default=cfg.robo_workspace, required=False)
    parser.add_argument(
        "--project_id", help="project id from https://app.roboflow.com/ in browser after signing in.",
        default=cfg.robo_project_id, required=False)
    parser.add_argument(
        "--project", help="project name from https://app.roboflow.com/",
        default=cfg.robo_project, required=False)
    parser.add_argument(
        "--ann_group", help="annotationGroup from https://app.roboflow.com/ in browser after signing in.",
        default=cfg.robo_ann_group, required=False)
    parser.add_argument(
        "--dataset_version", help="dataset version generated from https://app.roboflow.com/.",
        default=cfg.robo_dataset_version, required=False, type=int)
    parser.add_argument(
        "--target_dir", help=f"target directory to download meta file ({rfmeta_fname}) & dataset ({download_dirname})",
        default=cfg.root_dir, required=False)
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