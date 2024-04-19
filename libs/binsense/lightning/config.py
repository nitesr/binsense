from .. import config as cfg

from dataclasses import dataclass
import os

@dataclass
class Config:
    data_dirpath: str = cfg.BIN_DATA_DIR
    data_csv_filepath: str = os.path.join(cfg.BIN_DATA_DIR, 'inimage_queries.csv')
    embed_store_dirpath: str = os.path.join(cfg.BIN_DATA_DIR, 'embed_store')
    chkpt_dirpath: str = os.path.join(cfg.BIN_DATA_DIR, 'chkpts')
    results_csv_filepath: str = os.path.join(cfg.BIN_DATA_DIR, 'test_results.csv')
    results_topk_bboxes: int = 15
    tb_logs_dir = os.path.join(cfg.LOGS_DIR, 'tb')
    experiment_version: str = None
    learning_rate: float = 1e-5
    lr_decay_rate: float = 0.95
    min_epochs: int = 10
    max_epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    reg_loss_coef: float = 0.5
    giou_loss_coef: float = 1.0
    label_loss_coef: float = 1.5
    eos_coef: float = 1.0
    use_focal_loss: bool = True
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    match_cost_label: float = 0.25
    match_cost_bbox: float = 0.75
    match_cost_giou: float = 2.0
    iou_threshold: float = 0.98
    nms_threshold: float = 1
    score_threshold: float = 0.98
    use_no_object_class: bool = False
