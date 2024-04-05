from ..dataprep.config import DataPrepConfig
from ..embed_datastore import SafeTensorEmbeddingDatastore
from ..dataset_util import Dataset as BinsenseDataset
from ..dataset_util import DataTag, Yolov8Deserializer


import pandas as pd
import logging, random, torch, os, argparse, sys

def _get_best_bboxes(downloaded_ds: BinsenseDataset) -> pd.DataFrame:
    bbox_df = pd.DataFrame(columns=['bbox_label', 'image_name', 'image_path', 'bbox_idx', 'bbox_area'])
    imgs_data = downloaded_ds.get_images(DataTag.TRAIN)
    for img_data in imgs_data:
        bboxes_data = downloaded_ds.get_bboxes(img_data.name)
        for bbox_idx, bbox_data in enumerate(bboxes_data):
            bbox_df.loc[len(bbox_df)] = [
                bbox_data.label, img_data.name, 
                img_data.path, bbox_idx,
                bbox_data.width*bbox_data.height
            ]
    
    best_bbox_idxs = bbox_df.groupby('bbox_label')['bbox_area'].idxmax()
    best_bbox_df = bbox_df.loc[best_bbox_idxs][['bbox_label', 'image_name', 'image_path', 'bbox_idx']]
    best_bbox_df.reset_index(drop=True, inplace=True)
    del bbox_df
    return best_bbox_df

def validate(test_size: int = None) -> bool:
    cfg = DataPrepConfig()
    downloaded_ds = Yolov8Deserializer(
            cfg.filtered_dataset_path,
            img_extns=['.jpg']).read()
    best_bbox_df =_get_best_bboxes(downloaded_ds)
    print(f"total bbox labels are {best_bbox_df.shape[0]}")
    
    embedding_store = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    labels = best_bbox_df['bbox_label'].tolist()
    
    if test_size is not None:
        labels = labels[0:test_size]
    
    keys = list(embedding_store.get_keys())
    
    assert len(keys) == len(labels)
    assert embedding_store.has(random.choice(labels))
    assert embedding_store.has(random.choice(labels))
    assert embedding_store.get(random.choice(labels)).shape == torch.Size([512])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--validate", help="validate metadata and dataset with orig dataset",
        action="store_true")

    parser.add_argument(
        "--test_size", help="limit the test size to this one",
        type=int, default=None)
    
    args = parser.parse_args()
    if args.validate:
        validate(args.test_size)
