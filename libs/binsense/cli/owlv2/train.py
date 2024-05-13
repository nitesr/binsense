from ...owlv2 import hugg_loader as hloader
from ...owlv2.model import Owlv2ForObjectDetection
from ...owlv2.config import Owlv2Config
from ...owlv2.processor import Owlv2ImageProcessor

from ...dataprep.config import DataPrepConfig
from ...lightning.config import Config as TrainConfig
from ...lightning.dataset import InImageQueryDatasetBuilder, LitInImageQuerierDM
from ...lightning.owlv2_model import OwlV2InImageQuerier
from ...lightning.model import LitInImageQuerier
from ...embed_datastore import SafeTensorEmbeddingDatastore, EmbeddingDatastore
from ...utils import get_default_on_none, load_params

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from typing import Dict, Tuple, Any

import pandas as pd
import lightning as L
import logging, argparse, os, sys, torch

def _get_baseline_model():
    owl_model_cfg = Owlv2Config(**hloader.load_owlv2model_config())
    model = Owlv2ForObjectDetection(owl_model_cfg)
    model.load_state_dict(hloader.load_owlv2model_statedict())
    return OwlV2InImageQuerier(model)

class TransformFn:
    def __init__(self, embed_ds: EmbeddingDatastore) -> None:
        self.embed_ds = embed_ds
        self.processor = Owlv2ImageProcessor()
        
    def transform(self, record):
        inputs, target = record
        orig_width, orig_height = inputs['image'].width, inputs['image'].height
        max_length = max(orig_width, orig_height)
        inputs['image'] = self.processor.preprocess(inputs['image'])['pixel_values'][0]
        inputs['query'] = self.embed_ds.get(inputs['query']).reshape((1, -1))
        if target['q_boxes'].shape[0] > 0:
            target['q_boxes'][:,0] = target['q_boxes'][:,0] * orig_width / max_length
            target['q_boxes'][:,1] = target['q_boxes'][:,1] * orig_height / max_length

        if target['boxes'].shape[0] > 0:
            target['boxes'][:,0] = target['boxes'][:,0] * orig_width / max_length
            target['boxes'][:,1] = target['boxes'][:,1] * orig_height / max_length
        return inputs, target

    def __call__(self, record: Any) -> Any:
        return self.transform(record)

def print_dataset_stats(cfg: DataPrepConfig, csv_path: str) -> None:
    df = pd.read_csv(csv_path)[["query_label", "image_relpath", "count", "tag"]]
    
    # TODO: change it to presentable format
    ds_stats_fp = os.path.join(cfg.root_dir, 'train_dataset_stats.txt')
    with open(ds_stats_fp, 'w') as f:
        txt = "count by tag --> \n"
        metric = df.groupby("tag").aggregate("count")["count"]
        txt += f'{metric}'
        f.write(f"{txt}" + '\n')
        print(txt)

        txt = "count by tag & is_pos_query --> \n"
        df["query_type"] = df["count"] > 0
        metric = df.groupby(by=["tag", "query_type"]).aggregate("count")["count"]
        txt += f'{metric}'
        f.write(f"{txt}" + '\n')
        print(txt)

        txt = "quantiles by class counts --> \n"
        txt += "train:\n"
        metric = df.query("tag == 'train'").groupby(by=["query_label"]).aggregate("count")["count"].describe()
        txt += f'{metric}' + '\n'
        txt += "test:\n"
        metric = df.query("tag == 'test'").groupby(by=["query_label"]).aggregate("count")["count"].describe()
        txt += f'{metric}' + '\n'
        f.write(f"{txt}" + '\n')
        print(txt)

        txt = "quantiles by item counts --> \n"
        txt += "train:\n"
        metric = df.query("tag == 'train'").groupby(by=["count"]).aggregate("count")["tag"].describe()
        txt += f'{metric}' + '\n'
        txt += "test:\n"
        metric = df.query("tag == 'test'").groupby(by=["count"]).aggregate("count")["tag"].describe()
        txt += f'{metric}' + '\n'
        f.write(f"{txt}" + '\n')
        print(txt)


def build_dataset(pos_neg_ratio: float = None, manual_seed: int = None):
    tcfg = TrainConfig()
    dcfg = DataPrepConfig()
    dcfg.inimage_queries_pos_neg_ratio = get_default_on_none(pos_neg_ratio, dcfg.inimage_queries_pos_neg_ratio)
    dcfg.inimage_queries_csv = tcfg.data_csv_filepath
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True).to_read_only_store()
    csv_path, _  = InImageQueryDatasetBuilder(embed_ds=embed_ds, cfg=dcfg, manual_seed=manual_seed).build()
    print(f"dataset built @ {csv_path}")

    print_dataset_stats(dcfg, csv_path)

def _sync_config(
    batch_size: int = None, 
    num_workers: int = 0, 
    learning_rate: float= None, 
    **kwargs) -> Tuple[TrainConfig, Dict]:
    cfg = TrainConfig()
    cfg.batch_size = get_default_on_none(batch_size, cfg.batch_size)
    cfg.num_workers = get_default_on_none(num_workers, cfg.num_workers)
    cfg.learning_rate = get_default_on_none(learning_rate, cfg.learning_rate)
    
    for k in cfg.__dict__:
        cfg.__dict__[k] = kwargs.pop(k, cfg.__dict__[k])
    
    print('Config: ', cfg)
    return cfg, kwargs

def train(
    batch_size: int = None, 
    learning_rate: float= None, 
    num_workers: int = 0, 
    ckpt_fname: str = None,
    manual_seed: int = None,
    early_stopping: bool = False,
    **kwargs):
    
    print('kwargs: ', kwargs)
    cfg, kwargs = _sync_config(batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate, **kwargs)
    print('kwargs: ', kwargs)
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True).to_read_only_store()
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        transform=TransformFn(embed_ds),
        random_state=manual_seed)
    
    model = _get_baseline_model()
    lmodel = LitInImageQuerier(model, cfg=cfg)
    
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=3))

    trainer = L.Trainer(
        min_epochs=cfg.min_epochs, 
        max_epochs=cfg.max_epochs, 
        callbacks=callbacks,
        **kwargs)
    ckpt_fpath = os.path.join(cfg.chkpt_dirpath, ckpt_fname) if ckpt_fname else None
    trainer.fit(lmodel, datamodule=data_module, ckpt_path=ckpt_fpath)

def test(
    batch_size: int = None, 
    num_workers: int = 0, 
    ckpt_fname: str = None,
    experiment_version: str = None,
    manual_seed: int = None,
    **kwargs):
    cfg, kwargs = _sync_config(batch_size=batch_size, num_workers=num_workers, **kwargs)
    print('kwargs: ', kwargs)
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True).to_read_only_store()
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        transform=TransformFn(embed_ds),
        random_state=manual_seed)
    
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
    cfg = TrainConfig()
    dcfg = DataPrepConfig()
    params = load_params('./params.yaml')

    parser.add_argument(
        "--learning_rate", help="learning rate", type=float,
        default=cfg.learning_rate)
    
    parser.add_argument(
        "--min_epochs", help="minimum number of epochs", type=int,
        default=params.train.min_epochs)
    
    parser.add_argument(
        "--max_epochs", help="maximum number of epochs", type=int,
        default=params.train.max_epochs)
    
    parser.add_argument(
        "--batch_size", help="batch size", type=int,
        default=params.train.batch_size)
    
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
        default=params.train.dl_workers)
    
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
        default=params.train.profiler)
    
    parser.add_argument(
        "--build_dataset", help="build only dataset", action="store_true")
    
    parser.add_argument(
        "--pos_neg_dataset_ratio", help="pos:neg data ratio",
        default=params.train.pos_neg_dataset_ratio,
        type=float
    )
    parser.add_argument(
        "--manual_seed", help="manual seed for randomness",
        default=params.train.manual_seed, type=int
    )
    parser.add_argument(
        "--early_stopping", help="consider early stopping for training",
        default=params.train.early_stopping, type=bool
    )
    
    parser.add_argument(
        "--train", help="train the model", action="store_true")
    
    parser.add_argument(
        "--test", help="test the model", action="store_true")
    
    
    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    if args.build_dataset:
        build_dataset(args.pos_neg_dataset_ratio, args.manual_seed)
    
    if args.train:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        train(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ckpt_fname=args.ckpt_fname,
            logger=tlogger,
            manual_seed=args.manual_seed,
            early_stopping=args.early_stopping,
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
            eos_coef=args.eos_coef,
            log_every_n_steps=params.train.log_every_n_steps
        )
    elif args.test:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        test(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ckpt_fname=args.ckpt_fname,
            logger=tlogger,
            manual_seed=args.manual_seed,
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