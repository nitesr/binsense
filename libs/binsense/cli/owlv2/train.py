from ...owlv2 import hugg_loader as hloader
from ...owlv2.model import Owlv2ForObjectDetection
from ...owlv2.config import Owlv2Config
from ...owlv2.processor import Owlv2ImageProcessor

from ...dataprep.config import DataPrepConfig
from ...lightning.config import Config as TrainConfig
from ...lightning.dataset import InImageQueryDatasetBuilder, LitInImageQuerierDM
from ...lightning.owlv2_model import OwlV2InImageQuerier
from ...lightning.model import LitInImageQuerier
from ...embed_datastore import SafeTensorEmbeddingDatastore
from lightning.pytorch.loggers import TensorBoardLogger

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

def build_dataset():
    tcfg = TrainConfig()
    dcfg = DataPrepConfig()
    dcfg.inimage_queries_csv = tcfg.data_csv_filepath
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    csv_path, _  = InImageQueryDatasetBuilder(embed_ds=embed_ds, cfg=dcfg).build()
    print(f"dataset built @ {csv_path}")

def train(baseline_model: bool = True, epochs: int = None, batch_size: int = None, num_workers: int = 0, ckpt_fname: str = None, **kwargs):
    cfg = TrainConfig()
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, transform=_get_transform_fn(embed_ds))
    
    model = _get_baseline_model() if baseline_model else None
    lmodel = LitInImageQuerier(model)
    
    trainer = L.Trainer(**kwargs)
    ckpt_fpath = os.path.join(cfg.chkpt_dirpath, ckpt_fname) if ckpt_fname else None
    trainer.fit(lmodel, datamodule=data_module, ckpt_path=ckpt_fpath)

def test(baseline_model: bool = True, batch_size: int = None, num_workers: int = 0, ckpt_fname: str = None, **kwargs):
    cfg = TrainConfig()
    
    # TODO: change it to get directly from TrainConfig
    embed_ds = SafeTensorEmbeddingDatastore(cfg.embed_store_dirpath, read_only=True)
    data_module = LitInImageQuerierDM(
        data_dir=cfg.data_dirpath,
        csv_filepath=cfg.data_csv_filepath, 
        batch_size=batch_size, 
        num_workers=num_workers, transform=_get_transform_fn(embed_ds))
    
    model = _get_baseline_model() if baseline_model else None
    lmodel = LitInImageQuerier(model)
    trainer = L.Trainer(**kwargs)
    ckpt_fpath = os.path.join(cfg.chkpt_dirpath, ckpt_fname) if ckpt_fname else None
    trainer.test(lmodel, datamodule=data_module, ckpt_path=ckpt_fpath)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = TrainConfig()

    parser.add_argument(
        "--epochs", help="number of epochs", type=int,
        default=cfg.epochs)
    
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
        "--num_workers", help="Number of dataloader workers", type=int,
        default=cfg.num_workers)
    
    parser.add_argument(
        "--baseline_model", help="use baseline model",
        action="store_true")
    
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
        "--experiment_version", help="experiment version", type=str,
        default='v0')
    
    parser.add_argument(
        "--profiler", help="profiline", type=str,
        default=None)
    
    parser.add_argument(
        "--build_dataset", help="build only dataset", action="store_true")
    
    parser.add_argument(
        "--train", help="train the model", action="store_true")
    
    parser.add_argument(
        "--test", help="test the model", action="store_true")
    
    args = parser.parse_args()
    if args.build_dataset:
        build_dataset()
    
    if args.train:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        train(
            baseline_model=args.baseline_model,
            epochs=args.epochs,
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
            profiler=args.profiler
        )
    elif args.test:
        tlogger = TensorBoardLogger(cfg.tb_logs_dir, version=args.experiment_version)
        test(
            baseline_model=args.baseline_model,
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
            profiler=args.profiler
        )
        sys.exit(0)