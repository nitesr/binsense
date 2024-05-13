from ...dataprep.roboflow_util import RoboflowDatasetValidator
from ...dataprep.roboflow_util import RoboflowDatasetReader

from ...dataprep.config import DataPrepConfig
from ...dataprep.downloader import BinS3DataDownloader
from ...dataprep.metadata import BinMetadataLoader
from ...dataset_util import YoloDatasetCopier
from ...utils import load_params

import pandas as pd
import logging, os, argparse, sys

def get_orig_dataset(cfg: DataPrepConfig, force_download=False, num_workers: int = 1):
    BinS3DataDownloader(cfg).download(force=force_download, max_workers=num_workers)
    _, orig_df = BinMetadataLoader(cfg).load(max_workers=num_workers)
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
    uploaded_df['uploaded'] = 1

    df = orig_df.merge(uploaded_df[['image_name', 'bbox_label', 'tag', 'uploaded']], on=["image_name", "bbox_label"], how="left")
    
    uploaded_corrections_df = pd.read_csv(cfg.data_split_corrections_filepath, dtype={'image_name': str, 'bbox_label': str})
    uploaded_corrections_df['uploaded'] = 1

    df = df.merge(uploaded_corrections_df, suffixes=[None, '_corr'], on=['image_name', 'bbox_label'], how="outer")
    df.loc[~df.bbox_count_corr.isna(), 'bbox_count'] = df['bbox_count_corr']
    df.loc[~df.bbox_count_corr.isna(), 'uploaded'] = df['uploaded_corr']
    df = df[['image_name', 'tag', 'uploaded', 'bbox_label', 'bbox_count']]

    df.fillna({'tag': 'test', 'uploaded': 0}, inplace=True)
    del uploaded_df
    return df

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
    else:
        target_path = cfg.filtered_dataset_path
    

    rfds_dir = _default_rfds_path(source_dir, cfg)
    source_ds = RoboflowDatasetReader(
        dataset_dirpath=rfds_dir, cfg=cfg).read()
    img_tags_df = orig_df.query('uploaded == 1').groupby(by='image_name').agg(first_tag=('tag', 'first')).reset_index()
    img_tag_pairs = dict(zip(img_tags_df.image_name, img_tags_df.first_tag))
    dst_categories = orig_df['bbox_label'].unique()

    dst_ds = YoloDatasetCopier(
        source_ds, 
        categories=dst_categories, 
        img_tag_pairs=img_tag_pairs).copy()

    dst_ds.to_file(target_path, format='yolov8')
    print(f"copied @ {target_path}")

    valid, checks = validate(orig_df, target_path)
    if not valid:
        print(f'validation failed after copy')
        print('\t\n'.join([ f'{chk[0]}: {chk[1]}' for chk in checks]))
        raise ValueError(f'validation failed after copy to {target_path}')

    return target_path

def validate(orig_df: pd.DataFrame, rds_dirpath: str) -> bool:
    valid, checks =  RoboflowDatasetValidator(
        crawler_filepath=None,
        dataset_dirpath=rds_dirpath, 
        cfg=cfg, 
        validate_crawled_dataset=False).validate(orig_df)
    
    return valid, checks    

def run_validator(
    orig_df: pd.DataFrame,
    source_dir: str = None):
    
    cfg = DataPrepConfig()
    rfds_dir = None
    
    if source_dir is not None and source_dir != cfg.root_dir:
        rfds_dir = os.path.join(
            source_dir, os.path.split(cfg.dataset_download_path)[1])
        
    valid, checks = validate(orig_df, rfds_dir)
    valresults_fp = os.path.join(cfg.root_dir, 'downloaded_data_valid_results.txt')
    with open(valresults_fp, 'w') as f:
        f.write(f'validation successful: {valid}\n')
        f.write('\t\n'.join([ f'{chk[0]}: {chk[1]}' for chk in checks]))

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
    params = load_params('./params.yaml')

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
    parser.add_argument(
        "--num_workers", help=f"number of workers to download orig dataset",
        default=params.data_download.num_workers, required=False, type=int)
    
    args = parser.parse_args()
    
    if args.validate or args.copy:
        orig_df = get_orig_dataset(
            cfg=cfg, 
            num_workers=args.num_workers)
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
                print(f"running copier..{args.source_dir} -> {args.target_dir}")
                run_copier(orig_df, args.source_dir, args.target_dir)
            else:
                print("validation failed!")
                sys.exit(1)
    
    sys.exit(0)