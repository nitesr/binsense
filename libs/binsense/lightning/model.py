from .metrics import QueryAccuracy
from .spec import InImageQuerier, ImageEmbedder
from .spec import MultiBoxLoss
from ..embed_datastore import EmbeddingDatastore, SafeTensorEmbeddingDatastore
from .config import Config
from ..utils import get_default_on_none, backup_file
from ..img_utils import corner_to_centers
from .losses import DETRMultiBoxLoss
from .. import torch_utils as tutls

from collections import OrderedDict
from torchmetrics.classification import MulticlassConfusionMatrix
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.ops import box_iou
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Any, List, Dict, Tuple, Mapping
from matplotlib import pyplot as plt

import lightning as L

import torch, logging, os

logger = logging.getLogger(__name__)

class LitImageEmbedder(L.LightningModule):
    def __init__(
        self,
        model: ImageEmbedder,
        bbox_labels: List,
        embed_ds: EmbeddingDatastore
    ) -> None:
        super(LitImageEmbedder, self).__init__()
        self.model = model
        self.embed_ds = embed_ds
        self.bbox_labels = bbox_labels
    
    def predict_step(self, batch):
        idx = batch[0]
        x = batch[1]
        _, embeddings = self.model(x)
        return (idx, embeddings)
    
    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        idx = self.all_gather(outputs[0])
        bbox_embeddings = self.all_gather(outputs[1])
        if self.trainer.is_global_zero:
            idx = idx.flatten(0, 1) if len(idx.shape) > 2 else idx
            bbox_embeddings = bbox_embeddings.flatten(0, 1) if len(bbox_embeddings.shape) > 2 else bbox_embeddings
            labels = [self.bbox_labels[i] for i in idx]
            self.embed_ds.put_many(labels, bbox_embeddings)

class LitInImageQuerier(L.LightningModule):
    def __init__(
        self, 
        model: InImageQuerier, 
        cfg: Config = None,
        results_csvpath: str = None,
    ) -> None:
        super(LitInImageQuerier, self).__init__()
        
        self.model = model
        self.cfg = get_default_on_none(cfg, Config())
        
        self.save_hyperparameters("cfg")
        
        self.loss = DETRMultiBoxLoss(
                self.cfg.reg_loss_coef, self.cfg.giou_loss_coef, 
                self.cfg.label_loss_coef, self.cfg.eos_coef)
        self.iou_threshold = self.cfg.iou_threshold
        self.lr = self.cfg.learning_rate
        self.lr_decay_rate = self.cfg.lr_decay_rate
        self.results_csvpath = get_default_on_none(results_csvpath, self.cfg.results_csv_filepath)
        
        # Lightning expects the metrics to be defined as module variables
        self.train_exists_acc = QueryAccuracy(criteria="exists")
        self.val_exists_acc = QueryAccuracy(criteria="exists")
        self.test_exists_acc = QueryAccuracy(criteria="exists")
        
        #conf matrix by count (0, 1, 2, 3, 4, >4)
        self.train_conf_matrix = MulticlassConfusionMatrix(num_classes=6)
        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes=6)
        self.test_conf_matrix = MulticlassConfusionMatrix(num_classes=6)
        
        self.conf_matrix = {
            'train': self.train_conf_matrix,
            'val': self.val_conf_matrix,
            'test': self.test_conf_matrix
        }
        
        self.exists_acc = {
            'train': self.train_exists_acc,
            'val': self.val_exists_acc,
            'test': self.test_exists_acc
        }
        
        self.train_matches_acc = QueryAccuracy(criteria="matches")
        self.val_matches_acc = QueryAccuracy(criteria="matches")
        self.test_matches_acc = QueryAccuracy(criteria="matches")
        
        self.matches_acc = {
            'train': self.train_matches_acc,
            'val': self.val_matches_acc,
            'test': self.test_matches_acc
        }
        
        self.train_meets_acc = QueryAccuracy(criteria="meets")
        self.val_meets_acc = QueryAccuracy(criteria="meets")
        self.test_meets_acc = QueryAccuracy(criteria="meets")
        
        self.meets_acc = {
            'train': self.train_meets_acc,
            'val': self.val_meets_acc,
            'test': self.test_meets_acc
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=self.lr_decay_rate)
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def select_preds(self, outputs: Dict[str, Tensor]) -> Dict[str, List]:
        """
        filter out the crowd
        """
        pred_logits = outputs['pred_logits'].detach()
        pred_boxes = outputs['pred_boxes'].detach()
        
        # expecting only one query class
        probs = torch.max(pred_logits[...,:-1], dim=-1)
        scores = torch.sigmoid(probs.values)
        bboxes_xy = corner_to_centers(pred_boxes)
        
        # Apply non-maximum suppression (NMS)
        if self.cfg.nms_threshold < 1.0:
            for idx in range(bboxes_xy.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue

                    ious = box_iou(bboxes_xy[idx][i, :].unsqueeze(0), bboxes_xy[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > self.cfg.nms_threshold] = 0.0
        
        selected_pred_boxes = []
        selected_pred_scores = []
        alphas = torch.zeros_like(scores, device=pred_logits.device)
        for idx in range(pred_boxes.shape[0]):
            # Select scores for boxes matching the current query:
            query_scores = scores[idx]
            if not query_scores.nonzero().numel():
                selected_pred_boxes.append(tutls.empty_float_tensor().to(pred_boxes.device))
                selected_pred_scores.append(tutls.empty_float_tensor().to(pred_boxes.device))
                continue

            # Apply threshold on scores before scaling
            query_scores[query_scores < self.cfg.score_threshold] = 0.0

            # Scale box alpha such that the best box for each query has alpha 1.0 
            #   and the worst box has alpha 0.1 range(10%, 100%) of max_score.
            max_score = torch.max(query_scores) + 1e-6
            query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
            query_alphas = torch.clip(query_alphas, 0.0, 1.0)
            alphas[idx] = query_alphas

            mask = alphas[idx] > 0
            box_scores = alphas[idx][mask]
            boxes = pred_boxes[idx][mask]
            selected_pred_boxes.append(boxes)
            selected_pred_scores.append(box_scores)
        
        return {"pred_boxes": selected_pred_boxes, "pred_scores": selected_pred_scores}
    
    def _compute_metrics(self, preds, targets, step: str, input_idx):
        assert preds is not None
        assert targets is not None
        
        device = targets["count"][0].device
        if len(preds['pred_boxes']) != len(targets["count"]):
            logger.error(f'''
                idxs={input_idx}, 
                preds_len={len(preds['pred_boxes'])}, 
                targets_len={len(targets["count"])}, targets={targets}''')
            assert False
        
        pred_counts = []
        target_counts = []
        for i, pred_boxes in enumerate(preds['pred_boxes']):
            pred_counts.append(tutls.to_int_tensor(pred_boxes.shape[0]).to(device))
            target_counts.append(targets["count"][i].to(device))
        
        pred = torch.stack(pred_counts, dim=0)
        tgt = torch.stack(target_counts, dim=0)
        
        metrics = OrderedDict()
        # accuracy by meeting the target (pred > count)
        if step in self.meets_acc:
            self.meets_acc[step](pred, tgt)
            metrics[f'{step}_meets_acc'] = self.meets_acc[step]
            
        # accuracy by count
        if step in self.matches_acc:
            self.matches_acc[step](pred, tgt)
            metrics[f'{step}_matches_acc'] = self.matches_acc[step]
        
        # accuracy by presence
        if step in self.exists_acc:
            self.exists_acc[step](pred, tgt)
            metrics[f'{step}_exists_acc'] = self.exists_acc[step]
        
        pred_classes = pred.clone()
        tgt_classes = tgt.clone()
        pred_classes[pred_classes > 4] = 5
        tgt_classes[tgt_classes > 4] = 5
        self.conf_matrix[step](pred_classes, tgt_classes)
        
        return metrics
    
    def _log_losses_metrics(self, losses: Dict, metrics: Dict, step: str, **kwargs):
        for i, (name, metric) in enumerate(metrics.items()):
            self.log(name=name, value=metric, prog_bar=(i == 0), **kwargs)
        
        for i, (name, loss) in enumerate(losses.items()):
            self.log(name=f'{step}_{name}', value=loss, prog_bar=(i == 0), **kwargs)
    
    def _log_conf_matrix(self, step: str):
        if not isinstance(self.logger, TensorBoardLogger):
            return
        
        tblogger = self.logger.experiment
        fig, ax = plt.subplots(1, 1, figsize = (10,7))
        fig, _ = self.conf_matrix[step].plot(ax=ax)
        plt.close(fig)
        tblogger.experiment.add_figure(f"{step}_confusion_matrix", fig, self.current_epoch)
    
    def _common_step(self, batch) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        inputs = batch[0]
        targets = batch[1]
        outputs = self.model(inputs)
        losses = self.loss(outputs, targets)
        return losses, outputs
    
    def training_step(self, batch, batch_idx) -> Tensor:
        losses, outputs = self._common_step(batch)
        return {'loss': losses['loss'], 'losses': losses, 'outputs': outputs, "input_idx": batch[0]["idx"]}
    
    def on_train_batch_end(self, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        batch_outputs = outputs['outputs']
        batch_losses = outputs['losses']
        preds = self.select_preds(batch_outputs)
        accs = self._compute_metrics(preds=preds, targets=batch[1], step='train', input_idx=batch[0]["idx"])
        self._log_losses_metrics(batch_losses, accs, 'train', on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))
    
    def validation_step(self, batch, batch_idx):
        losses, outputs = self._common_step(batch)
        return {'loss': losses['loss'], 'losses': losses, 'outputs': outputs, "input_idx": batch[0]["idx"] }
    
    def on_validation_batch_end(self, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_outputs = outputs['outputs']
        batch_losses = outputs['losses']
        preds = self.select_preds(batch_outputs)
        accs = self._compute_metrics(preds, batch[1], 'val', batch[0]["idx"])
        self._log_losses_metrics(batch_losses, accs, 'val', on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))
    
    def on_validation_epoch_end(self) -> None:
        self._log_conf_matrix('val')
    
    def _initialize_pred_results_file(self) -> None:
        if self.trainer.is_global_zero:
            if os.path.exists(self.results_csvpath):
                bkp_fp = backup_file(self.results_csvpath)
                logger.info(f'backing up {self.results_csvpath} to {bkp_fp}')
            with open(self.results_csvpath, 'w') as f:
                f.write('input_idx,pred_boxes_count,pred_boxes_coords')
                f.write('\n')
    
    def _log_pred_results(self, outputs: Tensor | Mapping[str, Any] | None, fpath: str) -> None:
        batch_input_idx = self.all_gather(outputs['input_idx'])
        batch_pred_boxes = self.all_gather(outputs['outputs']['pred_boxes'])
        batch_pred_logits = self.all_gather(outputs['outputs']['pred_logits'])
        
        if self.trainer.is_global_zero:
            # flatten the device index, in case the shape is more than expected.
            batch_input_idx = batch_input_idx.flatten(0, 1) if len(batch_input_idx.shape) > 1 else batch_input_idx
            batch_pred_boxes = batch_pred_boxes.flatten(0, 1) if len(batch_pred_boxes.shape) > 3 else batch_pred_boxes
            batch_pred_logits = batch_pred_logits.flatten(0, 1) if len(batch_pred_logits.shape) > 3 else batch_pred_logits
            preds = self.select_preds({"pred_boxes": batch_pred_boxes, 'pred_logits': batch_pred_logits})
            with open(fpath, 'a') as f:
                for i, input_idx in enumerate(batch_input_idx):
                    pred_boxes = preds['pred_boxes'][i]
                    pred_boxes_coords = ' '.join([str(t.item()) for t in pred_boxes.flatten()])
                    f.write(f'{input_idx},{pred_boxes.shape[0]},{pred_boxes_coords}')
                    f.write('\n')
    
    def on_test_start(self) -> None:
        self._initialize_pred_results_file()
    
    def on_train_epoch_end(self) -> None:
        self._log_conf_matrix('test')
    
    def on_test_batch_end(self, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch_outputs = outputs['outputs']
        preds = self.select_preds(batch_outputs)
        accs = self._compute_metrics(preds, batch[1], 'test', batch[0]["idx"])
        for name, metric in accs.items():
            self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(batch[0]))
        self._log_pred_results(outputs, self.results_csvpath)
    
    def test_step(self, batch, batch_idx):
        outputs = self.model(batch[0])
        return {'outputs': outputs, "input_idx": batch[0]["idx"]}
    
    def predict_step(self, batch) -> Any:
        inputs = batch[0]
        outputs = self.model(inputs)
        preds = self.select_preds(outputs)
        return preds

