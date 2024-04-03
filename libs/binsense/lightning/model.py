from .model_spec import InImageQuerier, ImageEmbedder
from .model_spec import MultiBoxLoss
from ..embed_datastore import EmbeddingDatastore

from torchmetrics.classification import Accuracy
from torch import Tensor
from typing import Any, List
import lightning as L
import torch

class LitImageEmbedder(L.LightningModule):
    def __init__(
        self,
        model: ImageEmbedder,
        bbox_labels: List,
        batch_size: int,
        embed_ds: EmbeddingDatastore
    ) -> None:
        super(LitImageEmbedder, self).__init__()
        self.model = model
        self.embed_ds = embed_ds
        self.bbox_labels = bbox_labels
        self.batch_size = batch_size
    
    def predict_step(self, batch, batch_idx):
        self.trainer.device_ids
        x = batch[0]
        _, bbox_embeddings = self.model(x)
        start_idx = batch_idx * self.batch_size
        end_idx = batch_idx * self.batch_size + len(x)
        self.embed_ds.put_many(self.bbox_labels[start_idx:end_idx], bbox_embeddings)

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