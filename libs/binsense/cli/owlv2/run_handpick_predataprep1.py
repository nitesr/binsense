from ...dataprep.metadata import BinMetadataLoader
from ...dataprep.downloader import BinS3DataDownloader
from ...dataprep.config import DataPrepConfig
from ...utils import load_params, get_default_on_none

from matplotlib import pyplot as plt
from PIL import Image
from typing import List, Tuple

import pandas as pd
import seaborn as sb
import numpy as np
import logging, os, argparse, sys, random

logger = logging.getLogger(__name__)

def to_bin_image(bin_id):
    """
    given a bin_id gets the bin's image in PIL
    """
    return Image.open(os.path.join(cfg.rawdata_images_dir, f'{bin_id}.jpg'))


def read_handpicked_bins():
    fp = os.path.join(cfg.root_dir, 'handpicked_bins.txt')
    with open(fp, 'r') as f:
        bin_ids = f.readlines()
    bin_ids = [id.strip() for id in bin_ids]
    return bin_ids

def download(num_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    BinS3DataDownloader(cfg).download(max_workers=num_workers)
    bin_df, item_df = BinMetadataLoader(cfg).load(max_workers=num_workers)
    return (bin_df, item_df)

def filter_data(bin_df, item_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    item_df.sort_values(by="item_id", inplace=True)
    # filter for items with more than one bin
    # df = item_df.groupby(['item_id'])['bin_id'].nunique().reset_index(name="bins")
    # df = df[df['bins'] > 1]
    # item_df = item_df[item_df.item_id.isin(df['item_id'])]
    # bin_df = bin_df[bin_df.bin_id.isin(item_df.bin_id)]

    # filter for handpicked bins only
    handpicked_bins = read_handpicked_bins()
    bin_df = bin_df[bin_df.bin_id.isin(handpicked_bins)]
    item_df = item_df[item_df.bin_id.isin(bin_df.bin_id)]

    logger.info(f"#bins={len(bin_df)}, #item_ids={item_df['item_id'].nunique()}")
    return bin_df, item_df

def validate_dataset(train_bin_ids: List[str], test_bin_ids: List[str], item_df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    Compares if the item in test_bin_ids are available in train_bin_ids
    Args:
        train_bin_ids (`List[str]`):
        test_bin_ids (`List[str]`):
        item_df (`pd.DataFrame`):
    Returns:
        missing item class ratio (`float`)
        missing item ids (`List[str]`)
    """
    train_item_df = item_df[item_df.bin_id.isin(train_bin_ids)]
    test_item_df = item_df[item_df.bin_id.isin(test_bin_ids)]
    
    test_item_ids = set(test_item_df['item_id'].unique().tolist())
    train_item_ids = set(train_item_df['item_id'].unique().tolist())
    missing_items_ids = test_item_ids - train_item_ids
    
    return len(missing_items_ids) / len(test_item_ids), missing_items_ids

def train_test_split(bin_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rnd = random.Random(seed)
    bin_ids = bin_df['bin_id'].tolist()
    rnd.shuffle(bin_ids)
    train_len = int(.8*len(bin_ids))
    train_bin_ids = bin_ids[0:train_len]
    test_bin_ids = bin_ids[train_len:]
    
    train_bin_df = bin_df[bin_df.bin_id.isin(train_bin_ids)].reset_index(drop=True)
    test_bin_df = bin_df[bin_df.bin_id.isin(test_bin_ids)].reset_index(drop=True)
    return train_bin_df, test_bin_df

def test_val_split(bin_df: pd.DataFrame, test_bin_df: pd.DataFrame, train_bin_df: pd.DataFrame, seed: int):
    # seperate test and valid sets
    test_df=test_bin_df.sample(frac=0.67,random_state=seed)

    bin_df['tag'] = bin_df['bin_id'].apply(lambda x: \
        'train' if x in train_bin_df['bin_id'].tolist() \
            else 'test' if x in test_df['bin_id'].tolist() \
                else 'valid'
    )
    logger.info(bin_df['tag'].value_counts())

def add_itemclasses(bin_df: pd.DataFrame, item_df: pd.DataFrame):
    x_item_df = item_df[item_df.bin_id.isin(bin_df['bin_id'])]
    df = x_item_df.groupby(['bin_id'])['item_id'].count().reset_index(name="item_classes")
    return bin_df.merge(df, how="inner", validate="1:1")


def plot_bin_attrs(train_bin_df: pd.DataFrame, test_bin_df: pd.DataFrame):
    bin_attrs = ['bin_qty', 'bin_image_kb', 'bin_image_width', 'bin_image_height', "item_classes"]
    
    rows = 2
    cols = len(bin_attrs)
    fig, axs = plt.subplots(rows, cols, figsize=(12, rows*3))
    for i, attr in enumerate(bin_attrs):
        # train
        g = sb.histplot(data=train_bin_df, x=attr, ax=axs[0][i])
        g.set(xlabel=None)
        if i > 0:
            axs[0][i].get_yaxis().set_visible(False)
        else:
            g.set_ylabel('train: count')
    
    for i, attr in enumerate(bin_attrs):
        # test
        g = sb.histplot(data=test_bin_df, x=attr, ax=axs[1][i])
        if i > 0:
            axs[1][i].get_yaxis().set_visible(False)
        else:
            g.set_ylabel('test: count')
            
    fig.suptitle("train vs test (bin attrs)")
    plt.savefig(
        fname=os.path.join(
            cfg.root_dir, 'handpick_train_test_dist.png'), 
        format='png')

def plot_sampled_bins(item_df: pd.DataFrame, test_bin_df: pd.DataFrame, train_bin_df: pd.DataFrame):
    test_item_ids = item_df[item_df.bin_id.isin(test_bin_df['bin_id'])]['item_id'].unique()
    train_item_ids = item_df[item_df.bin_id.isin(train_bin_df['bin_id'])]['item_id'].unique()

    sample_item_ids = list(filter(lambda id: id in train_item_ids, np.random.choice(test_item_ids, 4)))
    sample_item_df = item_df[item_df.item_id.isin(sample_item_ids)]

    def sample_bin(bin_df, item_df, item_id):
        the_item_df = item_df[item_df['item_id'] == item_id]
        df = bin_df[bin_df.bin_id.isin(the_item_df['bin_id'])].sample(1)
        return df['bin_id'].tolist()[0], df['bin_image_name'].tolist()[0]

    fig, axs = plt.subplots(len(sample_item_ids), 2, figsize=(8, 8))
    for i, item_id in enumerate(sample_item_ids):
        _, train_bin = sample_bin(train_bin_df, sample_item_df, item_id)
        _, test_bin = sample_bin(test_bin_df, sample_item_df, item_id)
        axs[i][0].imshow(Image.open(os.path.join(cfg.rawdata_images_dir, train_bin)))
        axs[i][1].imshow(Image.open(os.path.join(cfg.rawdata_images_dir, test_bin)))
    fig.suptitle("train & test samples")
    plt.savefig(
        fname=os.path.join(
            cfg.root_dir, 'handpick_train_test_samples.png'),
        format='png')

def presave_split(item_df: pd.DataFrame, bin_df: pd.DataFrame):
    df = item_df[['item_id', 'bin_id', 'item_qty']]
    df = df.merge(bin_df[["bin_id", "tag"]], how="left", on="bin_id")
    df.sort_values(by=["bin_id"], inplace=True)
    df['image_name'] = df["bin_id"] + '.jpg'
    df.rename(columns={
        'item_id': 'bbox_label',
        'item_qty': 'bbox_count'
    }, inplace=True)

    df = df[['image_name', 'tag', 'bbox_label', 'bbox_count']]
    return df

def validate_output(orig_item_df: pd.DataFrame, out_df: pd.DataFrame):
    out_bins = out_df['image_name'].str.split('.').str[0]
    df = orig_item_df[orig_item_df.bin_id.isin(out_bins)].copy()

    df['image_name'] = df['bin_id'] + '.jpg'
    df = df.merge(out_df, left_on=['image_name', 'item_id'], right_on=['image_name', 'bbox_label'])
    df['valid'] = df['item_qty'] == df['bbox_count']
    logger.info(f"#Number of invalid entries - {len(df[~df['valid']])}")
    assert len(df[~df['valid']]) == 0

def prep(num_workers: int, manual_seed: int):
    # download and filter the data
    obin_df, oitem_df  = download(num_workers=num_workers)
    bin_df, item_df  = filter_data(obin_df, oitem_df)
    
    train_bin_df, test_bin_df = train_test_split(bin_df, manual_seed)
    miss_ratio, missing_item_ids = validate_dataset(train_bin_df['bin_id'].tolist(), test_bin_df['bin_id'].tolist(), item_df)
    logger.info(f'#trrain={train_bin_df.shape[0]}, #test={test_bin_df.shape[0]}')
    logger.info(f'miss%={round(miss_ratio*100, 1)}, #miss={len(missing_item_ids)}')

    # plot the distributions
    plt_train_df = add_itemclasses(train_bin_df, item_df)
    plt_test_df = add_itemclasses(test_bin_df, item_df)
    plot_bin_attrs(plt_train_df, plt_test_df)

    # plot few samples
    plot_sampled_bins(item_df, test_bin_df, train_bin_df)

    # split test:val & then save train:test:val splits
    test_val_split(bin_df, test_bin_df, train_bin_df, manual_seed)
    out_df = presave_split(item_df, bin_df)
    validate_output(oitem_df, out_df)

    out_df.to_csv(cfg.data_split_filepath, index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    params = load_params('./params.yaml')

    parser.add_argument(
        "--num_workers", help="num of workers to prepare the data",
        type=int, default=params.data_download.num_workers)
    
    parser.add_argument(
        "--manual_seed", help="manual seed for random split",
        type=int, default=params.data_split.manual_seed)
    
    args = parser.parse_args()
    prep(
        num_workers=args.num_workers,
        manual_seed=args.manual_seed
    )
    
    sys.exit(0)


