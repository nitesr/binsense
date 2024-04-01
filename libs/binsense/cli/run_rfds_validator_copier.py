from ..dataprep.roboflow_util import RoboflowDatasetValidator
from ..dataprep.roboflow_util import RoboflowDatasetCopier
from ..dataprep.config import DataPrepConfig
from ..dataprep.downloader import BinS3DataDownloader
from ..dataprep.metadata import BinMetadataLoader

import pandas as pd
import logging, os, argparse, sys

def get_orig_dataset(cfg: DataPrepConfig):
    BinS3DataDownloader(cfg).download()
    _, orig_df = BinMetadataLoader(cfg).load()
    orig_df.sort_values(by=['bin_id', 'item_id'], inplace=True)
    orig_df['image_name'] = orig_df['bin_id'] + '.jpg'
    orig_df.rename(columns={
        'item_id': 'bbox_label',
        'item_qty': 'bbox_count'
    }, inplace=True)
    orig_df = orig_df[['image_name', 'bbox_label', 'bbox_count']].reset_index(drop=True)
    
    uploaded_df = pd.read_csv(cfg.data_split_filepath, dtype={'image_name': str, 'bbox_label': str})
    uploaded_df = uploaded_df[["image_name", "tag", "bbox_label", "bbox_count"]]
    uploaded_df["bbox_label"] = uploaded_df["bbox_label"].str.split('|').str[0]

    df = orig_df.merge(uploaded_df[['image_name', 'bbox_label', 'tag']], on=["image_name", "bbox_label"], how="left")
    df.fillna({'tag': 'test'}, inplace=True)
    # TODO: populate df['uploaded'] = 1 or 0 
    del uploaded_df
    return df

def _default_rfmeta_path(source_dir: str, cfg: DataPrepConfig) -> str:
    rfmeta_path = None
    if source_dir is not None and source_dir != cfg.root_dir:
        rfmeta_path = os.path.join(
            source_dir, os.path.split(cfg.rfmeta_file_path)[1])
    return rfmeta_path

def _default_rfds_path(source_dir: str, cfg: DataPrepConfig) -> str:
    rfds_dir = None
    if source_dir is not None and source_dir != cfg.root_dir:
        rfds_dir = os.path.join(
            source_dir, os.path.split(cfg.dataset_download_path)[1])
    return rfds_dir

def run_copier(
    orig_df: pd.DataFrame,
    source_dir: str = None,
    target_dir: str = None):
    
    cfg = DataPrepConfig()
    target_path = None
    if target_dir is not None and target_dir != cfg.root_dir:
        target_path = os.path.join(
            target_dir, os.path.split(cfg.filtered_dataset_path)[1])
        
    rfmeta_path = _default_rfmeta_path(source_dir, cfg)
    rfds_dir = _default_rfds_path(source_dir, cfg)
    
    copier = RoboflowDatasetCopier(
        target_path, crawler_filepath=rfmeta_path, 
        dataset_dirpath=rfds_dir, cfg=cfg)
    copiedto_path = copier.filter_and_copy(orig_df)
    print(f"copied @ {copiedto_path}")
    return copiedto_path

def run_validator(
    orig_df: pd.DataFrame,
    source_dir: str = None):
    
    cfg = DataPrepConfig()
    rfmeta_path = None
    rfds_dir = None
    
    if source_dir is not None and source_dir != cfg.root_dir:
        rfmeta_path = os.path.join(
            source_dir, os.path.split(cfg.rfmeta_file_path)[1])
        rfds_dir = os.path.join(
            source_dir, os.path.split(cfg.dataset_download_path)[1])
    
    validator = RoboflowDatasetValidator(
    crawler_filepath=rfmeta_path, 
    dataset_dirpath=rfds_dir)

    valid, checks = validator.validate(orig_df)
    print(f'validation successful: {valid}')
    print('\t\n'.join([ f'{chk[0]}: {chk[1]}' for chk in checks]))
    return valid
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    rfmeta_fname = os.path.split(cfg.rfmeta_file_path)[1]
    download_dirname = os.path.split(cfg.dataset_download_path)[1]
    dssplit_fname = os.path.split(cfg.data_split_filepath)[1]
    filteredds_dirname = os.path.split(cfg.filtered_dataset_path)[1]
    
    parser.add_argument(
        "--validate", help="validate metadata and dataset with orig dataset",
        action="store_true")
    parser.add_argument(
        "--source_dir", help=f"source directory for \
            train split file({dssplit_fname}), \
            meta file ({rfmeta_fname}), and downloded dataset ({download_dirname})",
        default=cfg.root_dir, required=False)
    parser.add_argument(
        "--copy", help="copy the downloaded & orig dataset",
        action="store_true")
    parser.add_argument(
        "--target_dir", help=f"target directory for \
            filtered dataset ({filteredds_dirname})",
        default=cfg.root_dir, required=False)
    
    args = parser.parse_args()
    
    if args.validate or args.copy:
        orig_df = get_orig_dataset(cfg)
        valid = None
        
        def run_v():
            print("running validator..")
            return run_validator(orig_df, args.source_dir)
        
        if args.validate:
            run_v()
        
        if args.copy:
            if valid is None:
                valid = run_v()
            if valid:
                print("running copier..")
                run_copier(orig_df, args.source_dir, args.target_dir)
            else:
                print("validation failed!")
                sys.exit(1)
    
    sys.exit(0)