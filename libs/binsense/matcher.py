from .img_utils import center_to_corners, scale_bboxes

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from torch import nn
from typing import Dict, List, Tuple, Optional

import torch, logging

logger = logging.getLogger(__name__)


def hungarian_matcher(
    outputs: Dict[str, torch.Tensor], 
    targets: List[Dict[str, torch.Tensor]], 
    cost_class: Optional[float] = 1, 
    cost_bbox: Optional[float] = 1, 
    cost_giou: Optional[float] = 1) -> List[ Tuple[int, int]]:
    """
    same as `HungarianMatcher`(..)(..)
    """
    return HungarianMatcher(cost_class, cost_bbox, cost_giou)(outputs, targets)

"""
Hungarian matcher
https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py
"""
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets 
        and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. 
        Because of this, in general, there are more predictions than targets. 
        In this case, we do a 1-to-1 matching of the best predictions,
        while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(
        self, 
        cost_class: Optional[float] = 1, 
        cost_bbox: Optional[float] = 1, 
        cost_giou: Optional[float] = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error 
                in the matching cost
            cost_bbox: This is the relative weight of the L1 error 
                of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss 
                of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: List[Dict[str, torch.Tensor]]) -> List[ Tuple[int, int]]:
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] 
                    with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] 
                    with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, num_classes = outputs["pred_logits"].shape
        

        # flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1)
        if num_classes == 1:
            # in case of num_classes = 1, apply sigmoid to get probability
            out_prob = torch.sigmoid(out_prob)
            matched_out_prob = out_prob
        else:
            out_prob = out_prob.softmax(-1)  # [batch_size * num_queries, num_classes]
            # concat the target labels to match the flattened preds
            tgt_ids = torch.cat([v["labels"] for v in targets])
            matched_out_prob = out_prob[:, tgt_ids]
            
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -matched_out_prob
        
        # flattened preds and concat the target boxes to match
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(center_to_corners(out_bbox), center_to_corners(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
        # Unflatten, B x patches x sum(tgt[i]_size)
        C = C.view(bs, num_queries, -1).cpu()

        # Unflatten,  B x [ B x patches x tgt[i]_size ]
        sizes = [len(v["boxes"]) for v in targets]
        C_tgts = [c for c in C.split(sizes, -1)]
        
        # complete assignment of anchors to ground truth boxes of minimal cost.
        # indices: selected indices (row x col) of B x [ B x patches x tgt[i]_size ][i]
        #   B x [ row array of patch idx, col array of gt idx ]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_tgts)]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



class SSDMatcher(nn.Module):
    def __init__(self, iou_threshold: Optional[float] = 0.5) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
    
    @torch.no_grad()
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]) -> List[ Tuple[int, int]]:
        pred_boxes = outputs["pred_boxes"]
        pred_scores = torch.sigmoid(outputs["pred_logits"])
        assert pred_scores.size(1) == pred_boxes.size(1)
        
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]
        assert gt_labels.size(1) == gt_boxes.size(1)
        
        device = pred_boxes.get_device()
        batch_size = pred_boxes.size(0)
        for i in range(batch_size):
            n_objects = gt_boxes[i].size(0)
            
            gt_boxes_corners = center_to_corners(gt_boxes[i])
            pred_boxes_corners = center_to_corners(pred_boxes[i])
            
            overlap_gt_x_anchor = generalized_box_iou(gt_boxes_corners, pred_boxes_corners)
            
            # match anchor to gt
            best_gt_by_anchor, best_gt_idx_by_anchor = overlap_gt_x_anchor.max(dim=0)
            _, best_anchor_idx_by_gt = overlap_gt_x_anchor.max(dim=1)
            
            # to handle if all anchors are assigned to other objects
            best_gt_idx_by_anchor[best_anchor_idx_by_gt] = torch.LongTensor(range(n_objects)).to(device)
            # to handle if all the anchors with object are assigned to background
            best_gt_by_anchor[best_anchor_idx_by_gt] = 1 
            
            # TODO: return pair of selected pred indices by gt indices
            return None
