from ...owlv2 import hugg_loader as hloader
from ...owlv2.model import Owlv2ForObjectDetection
from ...owlv2.config import Owlv2Config
from ...owlv2.processor import Owlv2ImageProcessor

from ...dataprep.config import coco_cfg, DataPrepConfig
from ...lightning.config import coco_train_cfg, Config as TrainConfig
from ...lightning.dataset import InImageQueryDatasetBuilder, LitInImageQuerierDM
from ...lightning.owlv2_model import OwlV2InImageQuerier
from ...lightning.model import LitInImageQuerier
from ...embed_datastore import SafeTensorEmbeddingDatastore
from ...utils import get_default_on_none, default_on_none

from lightning.pytorch.loggers import TensorBoardLogger
from typing import Dict, Tuple

import lightning as L
import logging, argparse, os, sys

def _get_baseline_model():
    owl_model_cfg = Owlv2Config(**hloader.load_owlv2model_config())
    model = Owlv2ForObjectDetection(owl_model_cfg)
    model.load_state_dict(hloader.load_owlv2model_statedict())
    return OwlV2InImageQuerier(model)

def _get_transform_fn(embed_ds):
    processor = Owlv2ImageProcessor()
    def transform(inputs):
        inputs['image'] = processor.preprocess(inputs['image'])['pixel_values'][0]
        inputs['query'] = embed_ds.get(inputs['query']).reshape((1, -1))
        return inputs
    return transform

def build_dataset(pos_neg_ratio: float = None):
    tcfg = coco_train_cfg()
    dcfg = coco_cfg()
    dcfg.inimage_queries_pos_neg_ratio = get_default_on_none(pos_neg_ratio, dcfg.inimage_queries_pos_neg_ratio)
    dcfg.inimage_queries_csv = tcfg.data_csv_filepath
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    csv_path, _  = InImageQueryDatasetBuilder(embed_ds=embed_ds, cfg=dcfg).build()
    print(f"dataset built @ {csv_path}")

def _sync_config(
    batch_size: int = None, 
    num_workers: int = 0, 
    learning_rate: float= None, 
    **kwargs) -> Tuple[TrainConfig, Dict]:
    cfg = coco_train_cfg()
    cfg.batch_size = get_default_on_none(batch_size, cfg.batch_size)
    cfg.num_workers = get_default_on_none(num_workers, cfg.num_workers)
    cfg.learning_rate = get_default_on_none(learning_rate, cfg.learning_rate)
    
    cfg_attrs = list(filter(lambda x: x[0] != '_', [n for n in cfg.__dir__()]))
    for k in cfg_attrs:
        cfg.__setattr__(k, kwargs.pop(k, cfg.__getattribute__(k)))
    
    print('Config: ', cfg)
    return cfg, kwargs

def train(
    batch_size: int = None, 
    learning_rate: float= None, 
    num_workers: int = 0, 
    ckpt_fname: str = None,
    **kwargs):
    
    print('kwargs: ', kwargs)
    cfg, kwargs = _sync_config(batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate, **kwargs)
    print('kwargs: ', kwargs)
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, transform=_get_transform_fn(embed_ds))
    
    model = _get_baseline_model()
    lmodel = LitInImageQuerier(model, cfg=cfg)
    
    trainer = L.Trainer(
        min_epochs=cfg.min_epochs, 
        max_epochs=cfg.max_epochs, 
        **kwargs)
    ckpt_fpath = os.path.join(cfg.chkpt_dirpath, ckpt_fname) if ckpt_fname else None
    trainer.fit(lmodel, datamodule=data_module, ckpt_path=ckpt_fpath)

def test(
    batch_size: int = None, 
    num_workers: int = 0, 
    ckpt_fname: str = None,
    experiment_version: str = None,
    **kwargs):
    cfg, kwargs = _sync_config(batch_size=batch_size, num_workers=num_workers, **kwargs)
    print('kwargs: ', kwargs)
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, transform=_get_transform_fn(embed_ds))
    
    model = _get_baseline_model()
    lmodel = LitInImageQuerier(
        model, cfg=cfg,
        results_csvpath=os.path.join(cfg.data_dirpath, f'testresults_{experiment_version}.csv')
    )
    trainer = L.Trainer(**kwargs)
    ckpt_fpath = os.path.join(cfg.chkpt_dirpath, ckpt_fname) if ckpt_fname else None
    trainer.test(lmodel, datamodule=data_module, ckpt_path=ckpt_fpath)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = coco_train_cfg()
    dcfg = coco_cfg()

    parser.add_argument(
        "--learning_rate", help="learning rate", type=float,
        default=cfg.learning_rate)
    
    parser.add_argument(
        "--min_epochs", help="minimum number of epochs", type=int,
        default=cfg.min_epochs)
    
    parser.add_argument(
        "--max_epochs", help="maximum number of epochs", type=int,
        default=cfg.max_epochs)
    
    parser.add_argument(
        "--batch_size", help="batch size", type=int,
        default=cfg.batch_size)
    
    parser.add_argument(
        "--score_threshold", help="score_threshold value", type=float,
        default=cfg.score_threshold)
    
    parser.add_argument(
        "--nms_threshold", help="nms_threshold value", type=float,
        default=cfg.nms_threshold)
    
    parser.add_argument(
        "--iou_threshold", help="iou_threshold value", type=float,
        default=cfg.iou_threshold)
    
    parser.add_argument(
        "--eos_coef", help="no object weight coef value", type=float,
        default=cfg.eos_coef)
    
    parser.add_argument(
        "--num_workers", help="Number of dataloader workers", type=int,
        default=cfg.num_workers)
    
    parser.add_argument(
        "--ckpt_fname", help="checkpoint of filename to resume from", type=str)
    
    parser.add_argument(
        "--fast_dev_run", help="few runs to test", 
        default=False)
    
    parser.add_argument(
        "--devices", help="num of devices or processes", 
        default=1)
    
    parser.add_argument(
        "--num_nodes", help="num of nodes", type=int,
        default=1)
    
    parser.add_argument(
        "--accelerator", help="trainer accelerator (cpu, gpu)", type=str,
        default='auto')
    
    parser.add_argument(
        "--strategy", help="trainer strategy (ddp, etc.)", type=str,
        default='auto')
    
    parser.add_argument(
        "--experiment_version", help="experiment version", type=str)
    
    parser.add_argument(
        "--profiler", help="profiling", type=str,
        default=None)
    
    parser.add_argument(
        "--build_dataset", help="build only dataset", action="store_true")
    
    parser.add_argument(
        "--pos_neg_dataset_ratio", help="pos:neg data ratio",
        default=dcfg.inimage_queries_pos_neg_ratio,
        type=float
    )
    
    parser.add_argument(
        "--train", help="train the model", action="store_true")
    
    parser.add_argument(
        "--test", help="test the model", action="store_true")
    
    args = parser.parse_args()
    if args.build_dataset:
        build_dataset(args.pos_neg_dataset_ratio)
    
    if args.train:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        train(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ckpt_fname=args.ckpt_fname,
            logger=tlogger,
            fast_dev_run=args.fast_dev_run,
            devices=args.devices,
            strategy=args.strategy,
            num_nodes=args.num_nodes,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            profiler=args.profiler,
            experiment_version=args.experiment_version,
            learning_rate=args.learning_rate,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            nms_threshold=args.nms_threshold,
            eos_coef=args.eos_coef
        )
    elif args.test:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        test(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ckpt_fname=args.ckpt_fname,
            logger=tlogger,
            fast_dev_run=args.fast_dev_run,
            devices=args.devices,
            strategy=args.strategy,
            num_nodes=args.num_nodes,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            profiler=args.profiler,
            experiment_version=args.experiment_version,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            nms_threshold=args.nms_threshold,
        )
        sys.exit(0)