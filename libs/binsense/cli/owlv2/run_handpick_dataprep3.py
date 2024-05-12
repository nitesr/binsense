from ...lightning.owlv2_model import Owlv2ImageEmbedder
from ...dataprep.config import DataPrepConfig
from ...dataprep.embedding_util import BBoxDatasetEmbedder, SafeTensorEmbeddingDatastore

from ...dataset_util import Yolov8Deserializer
from ...utils import load_params

from ...owlv2 import Owlv2ForObjectDetection, Owlv2Config
from ...owlv2 import hugg_loader as hloader

import argparse, logging, sys, random, torch

logger = logging.getLogger(__name__)

"""
usage on shell:
```
nohup python -m binsense.cli.owlv2.run_handpick_dataprep3 --generate_embeds > ./_logs/run_bbbox_embedder.log &
```
"""
def _get_bbox_embedder(cfg: DataPrepConfig) -> Owlv2ImageEmbedder:
    owl_model_cfg = Owlv2Config(**hloader.load_owlv2model_config())
    model = Owlv2ForObjectDetection(owl_model_cfg)
    model.load_state_dict(hloader.load_owlv2model_statedict())
    bbox_embedder = Owlv2ImageEmbedder(model=model)
    return bbox_embedder

def run_embedder(
    batch_size: int = None, 
    num_bbox_labels: int = None, **kwargs):
    cfg = DataPrepConfig()
    embedder = BBoxDatasetEmbedder(
        model=_get_bbox_embedder(cfg),
        batch_size=batch_size,
        num_bbox_labels = num_bbox_labels
    )
    fast_dev_run = kwargs.pop('test_run') if 'test_run' in kwargs else False
    embedder.generate(fast_dev_run=fast_dev_run, **kwargs)
    logger.info('generated embeddings, validating..')

    validate()


def validate():
    cfg = DataPrepConfig()
    downloaded_ds = Yolov8Deserializer(
            cfg.filtered_dataset_path,
            img_extns=['.jpg']).read()
    
    best_bbox_df = BBoxDatasetEmbedder(model=None)._get_best_bboxes(downloaded_ds)
    print(f"total bbox labels are {best_bbox_df.shape[0]}")
    
    embedding_store = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    labels = best_bbox_df['bbox_label'].tolist()
    
    keys = list(embedding_store.get_keys())
    
    results = [
        ('len(labels) == len(keys)', len(keys) == len(labels)),
        ('try a random label', embedding_store.has(random.choice(labels))),
        ('try a random label', embedding_store.has(random.choice(labels))),
        ('try a random label', embedding_store.has(random.choice(labels)))
    ]
    valid = all([t[1] for t in results])
    print(f'validation successful: {valid}')
    print('\t\n'.join([ f'{chk[0]}: {chk[1]}' for chk in results]))
    assert valid



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    params = load_params('./params.yaml')

    parser.add_argument(
        "--generate_embeds", help="preannotate using owlv2 model",
        action="store_true")
    parser.add_argument(
        "--batch_size", help="batch size for the model", default=params.generate_embeds.batch_size, type=int)
    parser.add_argument(
        "--num_workers", help="num of dataloader workers", default=params.generate_embeds.dl_workers, type=int)
    parser.add_argument(
        "--devices", help="num of devices available", default="auto")
    parser.add_argument(
        "--accelerator", help="lightining accelerator e.g. cpu, gpu, mps", default="auto", type=str)
    parser.add_argument(
        "--strategy", help="lightining strategy e.g. ddp", default="auto", type=str)
    parser.add_argument(
        "--num_bbox_labels", help="number of bbox labels", default=params.generate_embeds.bbox_labels_estimate, type=int)
    parser.add_argument(
        "--test_run", help="do a test run",
        action="store_true")
    
    args = parser.parse_args()
    if args.generate_embeds:
        run_embedder(
            batch_size=args.batch_size, 
            num_bbox_labels=args.num_bbox_labels,
            test_run=args.test_run,
            devices=args.devices,
            accelerator=args.accelerator,
            strategy=args.strategy,
            num_workers=args.num_workers
        )
    sys.exit(0)
