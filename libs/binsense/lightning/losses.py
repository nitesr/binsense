from binsense.img_utils import center_to_corners
from binsense.lightning.spec import MultiBoxLoss
from binsense.matcher import HungarianMatcher

from torch.nn import functional as F
from torchvision.ops import generalized_box_iou
from torchvision.ops import sigmoid_focal_loss
from collections import OrderedDict

from typing import Dict

import torch


class DETRMultiBoxLoss(MultiBoxLoss):

    def __init__(
        self,
        reg_loss_coef: float = 1.0,
        giou_loss_coef: float = 1.0,
        label_loss_coef: float = 1.0,
        eos_coef: float = 0.1,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        use_focal_loss: bool = False,
        has_no_object_class: bool = False,
        match_cost_label: float = 1,
        match_cost_bbox: float = 1,
        match_cost_giou: float = 1):
        super(DETRMultiBoxLoss, self).__init__()
        """
        creates the object detection loss used in DETR 
        Args:
            bbox_loss_coef (`float`): coefficient for l1 bbox loss
            giou_loss_coef (`float`): coefficient for giou bbox loss
            label_loss_coef (`float`): classfication loss - bce or mce
        """
        self.label_loss_coef = label_loss_coef
        self.reg_loss_coef = reg_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.eos_coef = eos_coef
        
        self.use_focal = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        self.has_no_object_class = has_no_object_class

         # TODO: do we need same weights
        self.matcher = HungarianMatcher(
            cost_class=match_cost_label,
            cost_bbox=match_cost_bbox,
            cost_giou=match_cost_giou)
    
    def _run_matcher(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        # lets not pass no-object logits to matcher 
        #   as there is nothing to match in ground truth
        match_pred_logits = pred_logits[...,:-1] if self.has_no_object_class else pred_logits
        matching_indices = self.matcher(
                outputs={'pred_boxes': pred_boxes, 'pred_logits': match_pred_logits},
                targets=[ \
                    {
                        'labels': gt_labels[i],
                        'boxes': gt_box
                    } for i, gt_box in enumerate(gt_boxes) ]
            )
        return matching_indices
    
    def _calc_bbox_loss(self, pred_boxes, gt_boxes, matching_indices):
        """
            calculates both the regression and iou loss.
            
            gets matching boxes based on matching indices
                on both pred and ground truth and then computes the loss
                for matching bboxes.
            note: the indices can be out of order in col i.e. on ground truths
        """
        src_batch_idx = torch.cat([torch.full_like(predi, i) for i, (predi, _) in enumerate(matching_indices)])
        src_pred_idx = torch.cat([predi for (predi, _) in matching_indices])
        
        src_boxes = pred_boxes[(src_batch_idx, src_pred_idx)]
        tgt_boxes = torch.cat([gt_bbox[gt_idx] for gt_bbox, (_, gt_idx) in zip(gt_boxes, matching_indices)], dim=0)

        num_boxes = sum([ len(x) for x in gt_boxes])
        loss_reg = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_reg = loss_reg.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            center_to_corners(src_boxes),
            center_to_corners(tgt_boxes)))
        loss_giou = loss_giou.sum() / num_boxes
        
        return loss_reg, loss_giou
    
    def _calc_ce_label_loss(self, pred_logits, gt_labels, matching_indices=None):
        # for label loss we will do it on entire prediction logits
        #   build the canvas of size prediction logits with label as no-object
        #   and update it with ground truth labels for the matching ones.
        # note: the indices can be out of order in col i.e. on ground truths)
        
        num_classes = pred_logits.shape[-1]
        src_logits = pred_logits
        
        # (num_classes-1) is no-object and maps to src_logits[..., -1]
        tgt_classes = torch.full(
            src_logits.shape[:2], (num_classes-1),
            dtype=torch.int64, device=src_logits.device)
        
        # if there are matching indices, update target label on canvas
        if matching_indices:
            src_batch_idx = torch.cat([torch.full_like(predi, i) for i, (predi, _) in enumerate(matching_indices)])
            src_pred_idx = torch.cat([predi for (predi, _) in matching_indices])
            tgt_classes_temp = torch.cat([gt_l[gt_idx] for gt_l, (_, gt_idx) in zip(gt_labels, matching_indices)])
            tgt_classes[(src_batch_idx, src_pred_idx)] = tgt_classes_temp
        
        empty_weight = torch.ones(num_classes, device=pred_logits.device)
        empty_weight[-1] = self.eos_coef
        return F.cross_entropy(src_logits.transpose(1, 2), tgt_classes, empty_weight, reduction='mean')
    
    def _calc_focal_label_loss(self, pred_logits, gt_labels, matching_indices=None):
        # for label loss we will do it on entire prediction logits
        #   build the canvas of size prediction logits with label as no-object
        #   and update it with ground truth labels for the matching ones.
        # note: the indices can be out of order in col i.e. on ground truths)
        
        num_classes = pred_logits.shape[-1]
        src_logits = pred_logits
        
        # (num_classes-1) is no-object and maps to src_logits[..., -1]
        ohe = torch.as_tensor([0.0]*(num_classes), device=src_logits.device)
        if self.has_no_object_class:
            ohe[-1] = 1.0
        
        tgt_probs = torch.full(
            src_logits.shape, 0.0,
            dtype=torch.float32, device=src_logits.device) + ohe
        
        # if there are matching indices, update target label on canvas
        if matching_indices:
            src_batch_idx = torch.cat([torch.full_like(predi, i) for i, (predi, _) in enumerate(matching_indices)])
            src_pred_idx = torch.cat([predi for (predi, _) in matching_indices])
            tgt_classes_temp = torch.cat([gt_l[gt_idx] for gt_l, (_, gt_idx) in zip(gt_labels, matching_indices)])
            classes = torch.arange(0, num_classes, dtype=torch.float32).to(src_logits.device)
            tgt_classes_temp = (tgt_classes_temp.unsqueeze(-1) == classes).float()
            tgt_probs[(src_batch_idx, src_pred_idx)] = tgt_classes_temp
        
        focal_loss = sigmoid_focal_loss(
            src_logits, tgt_probs, 
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, 
            reduction='none')
        
        empty_weight = torch.ones(num_classes, device=pred_logits.device)
        if self.has_no_object_class:
            empty_weight[-1] = self.eos_coef
        focal_loss = focal_loss * empty_weight
        
        return focal_loss.mean()
    
    def _calc_label_loss(self, pred_logits, gt_labels, matching_indices=None):
        if self.use_focal or not self.has_no_object_class:
            return self._calc_focal_label_loss(pred_logits, gt_labels, matching_indices)
        else:
            # can't cross entropy does softmax across the classes
            #  no-object needs to be considered for that.
            return self._calc_ce_label_loss(pred_logits, gt_labels, matching_indices)
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        assert "boxes" in targets
        assert "labels" in targets
        assert "pred_boxes" in outputs
        assert "pred_logits" in outputs

        pred_boxes = outputs["pred_boxes"] # B x NUM_PATCHES x 4
        pred_logits = outputs["pred_logits"] # B x NUM_PATCHES x NUM_QUERIES
        gt_boxes = targets["boxes"] # B x [ M x 4 ]
        
        assert pred_logits.shape[:2] == pred_boxes.shape[:2]
        assert pred_boxes.shape[0] == len(gt_boxes)
        
        # run hungrian matcher
        #   match is required only when the batch's ground truth has atleast one object
        gt_labels = targets["labels"] # B x [M x 1]
        match_required = max([ len(gt_l) for gt_l in gt_labels ]) > 0
        if match_required:
            matching_indices = self._run_matcher(pred_logits, pred_boxes, gt_labels, gt_boxes)
            loss_reg, loss_giou = self._calc_bbox_loss(pred_boxes, gt_boxes, matching_indices)
            loss_label = self._calc_label_loss(pred_logits, gt_labels, matching_indices)
        else:
            loss_reg, loss_giou = torch.as_tensor(0.0, device=pred_boxes.device), torch.as_tensor(0.0, device=pred_boxes.device)
            loss_label = self._calc_label_loss(pred_logits, gt_labels)
        
        loss = self.reg_loss_coef * loss_reg \
            + self.giou_loss_coef * loss_giou \
            + self.label_loss_coef * loss_label
        
        return OrderedDict({
            'loss': loss, 'loss_reg': loss_reg.detach(), 
            'loss_giou': loss_giou.detach(), 'loss_label': loss_label.detach()
        })