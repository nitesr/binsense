from ...lightning.owlv2_model import Owlv2ImageEmbedder
from ...dataprep.config import DataPrepConfig
from ...dataprep.embedding_util import BBoxDatasetEmbedder

from ...owlv2 import Owlv2ForObjectDetection, Owlv2Config
from ...owlv2 import hugg_loader as hloader

import argparse, logging, sys

logger = logging.getLogger(__name__)

"""
usage on shell:
```
nohup python -m binsense.cli.owlv2.run_bbox_embedder --generate > ./_logs/run_bbbox_embedder.log &
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
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate", help="preannotate using owlv2 model",
        action="store_true")
    parser.add_argument(
        "--batch_size", help="batch size for the model", default=8, type=int)
    parser.add_argument(
        "--num_workers", help="num of dataloader workers", default=None, type=int)
    parser.add_argument(
        "--devices", help="num of devices available", default="auto")
    parser.add_argument(
        "--accelerator", help="lightining accelerator e.g. cpu, gpu", default="auto", type=str)
    parser.add_argument(
        "--strategy", help="lightining strategy e.g. ddp", default="auto", type=str)
    parser.add_argument(
        "--num_bbox_labels", help="number of bbox labels", default=2000, type=int)
    parser.add_argument(
        "--test_run", help="do a test run",
        action="store_true")
    
    args = parser.parse_args()
    if args.generate:
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
