from .model_spec import InImageQuerier
from .model_spec import MultiBoxLoss

from torchmetrics.classification import Accuracy
from torch import Tensor
from typing import Any
import lightning as L
import torch

class LitInImageQuerier(L.LightningModule):
    def __init__(
        self, 
        model: InImageQuerier, 
        loss: MultiBoxLoss) -> None:
        super(LitInImageQuerier, self).__init__()
        self.model = model
        self.loss = loss
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.iou_threshold = 0.98
    
    def transform_preds(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes) -> Tensor:
        """
        filter out the crowd
        """
        pass
    
    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        logits, pred_bboxes = self.model(x)
        gt_labels, gt_boxes = y
        train_loss = self.loss(pred_bboxes, logits, gt_labels, gt_boxes)
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, pred_bboxes = self.model(x)
        gt_labels, gt_boxes = y
        test_loss = self.loss(pred_bboxes, logits, gt_labels, gt_boxes)
        self.log("val_loss", test_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, pred_bboxes = self.model(x)
        gt_labels, gt_boxes = y
        test_loss = self.loss(pred_bboxes, logits, gt_labels, gt_boxes)
        self.log("test_loss", test_loss)