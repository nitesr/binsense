from ...dataprep.metadata import BinMetadataLoader
from ...dataprep.downloader import BinS3DataDownloader
from ...dataprep.config import DataPrepConfig
from ...embed_datastore import SafeTensorEmbeddingDatastore

import pandas as pd
import logging, os, argparse, sys

logger = logging.getLogger(__name__)

def extract_and_save(cfg: DataPrepConfig) -> bool:
    item_df = pd.read_csv(cfg.rawdata_items_csv_filepath, header=0, dtype={'item_id': str, 'bin_id': str})
    products_df = item_df.groupby(by='item_id').agg(item_name=('item_name', 'first')).reset_index()

    # filter only embedded items
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True).to_read_only_store()
    embedded_items = list(embed_ds.get_keys())
    products_df = products_df[products_df.item_id.isin(embedded_items)]

    logger.info(f'writing {products_df.shape[0]} products to file_path={cfg.embedded_products_csv_filepath}')
    products_df.to_csv(cfg.embedded_products_csv_filepath, index=False)
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    
    args = parser.parse_args()
    extract_and_save(
        cfg=cfg
    )
    
    sys.exit(0)


