from ..dataprep.dataset import ImageProcessor, OwlImageProcessor
from ..owlv2 import Owlv2ForObjectDetection
from .model_spec import ImageEmbedder, ObjectDetector, MultiBoxLoss
from ..img_utils import center_to_corners
from ..matcher import HungarianMatcher

from torchvision.ops import generalized_box_iou
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict

import torch, logging

logger = logging.getLogger(__name__)

class Owlv2BboxPredictor(ObjectDetector):
    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super(Owlv2BboxPredictor, self).__init__()
        self.model = model
        self._processor = OwlImageProcessor()

    def processor(self) -> ImageProcessor:
        return self._processor

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images(`torch.Tensor`): images in BxCxWxH format
        Return:
            bbox_logits(`torch.Tensor`):
            bboxes(`torch.Tensor`):
            scores(`torch.Tensor`):
        """
        image_embeds, _ = self.model.image_embedder(images)
        bbox_logits  = self.model.objectness_predictor(image_embeds)
        bboxes = self.model.box_predictor(image_embeds)

        probs = torch.sort(bbox_logits, descending=True)
        bboxes = bboxes[torch.arange(bboxes.size(0)).unsqueeze(1), probs.indices]
        scores = torch.sigmoid(probs.values)
        return (probs.values, bboxes, scores)


class Owlv2ImageEmbedder(ImageEmbedder):
    def __init__(self, model: Owlv2ForObjectDetection, *args, **kwargs) -> None:
        super(ImageEmbedder, self).__init__(*args, **kwargs)
        self.model = model
        self._processor = OwlImageProcessor()

    def get_embed_size(self) -> int:
        return 512
    
    def processor(self) -> ImageProcessor:
        return self._processor

    def forward(self, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # try with 
        #   highest IOU with actual image size
        #   normalized over pre-processed image size.
        #   i.e. (0, 0, im_w/960, im_h/960)
        vit_embeddings = self.model.image_embedder(bboxes)[0] # B x C x 960 x 960 -> B x 3600 x 768
        bbox_embeddings = self.model.embed_image_query(vit_embeddings)[0]
        
        # class_embeddings = self.model.class_predictor(vit_embeddings)[1] # B x 3600 x 768 -> B x 3600 x 512
        # object_logits = self.model.objectness_predictor(vit_embeddings) # B x 3600 x 768 -> B x 3600
        # del vit_embeddings
        # object_scores = torch.sigmoid(object_logits)
        # _, top_idxs = torch.max(object_scores, dim=1, keepdim=True)
        # bbox_embeddings = torch.take_along_dim(class_embeddings, top_idxs[...,None], dim=1)
        return (None, bbox_embeddings.squeeze(dim=1))

class DETRMultiBoxLossWithoutLabel(MultiBoxLoss):
    
    def __init__(
        self, 
        bbox_loss_coef: float = 1.0, 
        giou_loss_coef: float = 1.0, 
        label_loss_coef: float = 1.0):
        super(DETRMultiBoxLossWithoutLabel, self).__init__()
        """
        creates the object detection loss used in DETR 
        Args:
            bbox_loss_coef (`float`): coefficient for l1 bbox loss
            giou_loss_coef (`float`): coefficient for giou bbox loss
            label_loss_coef (`float`): classfication loss - bce or mce
        """
        self.label_loss_coef = label_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.matcher = HungarianMatcher(label_loss_coef, bbox_loss_coef, giou_loss_coef)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        pred_boxes = outputs["pred_boxes"] # B x NUM_PATCHES x 4
        pred_logits = outputs["pred_logits"] # B x NUM_PATCHES x 1
        gt_boxes = targets["boxes"] # B x [ M x 4 ]
        
        # 1 for object presence
        # B x [ M x 1 ]
        gt_labels = [
            torch.full_like(
                (gb.shape[0], 1), 1, 
                dtype=torch.int16, 
                device=pred_boxes.device) \
            for gb in gt_boxes ] 
        
        # run hungrian matcher
        matching_indices = self.matcher(
            {'pred_boxes': pred_boxes, 'pred_logits': pred_logits},
            [{'labels': gt_l, 'boxes': gt_boxes[i]} for i, gt_l in enumerate(gt_labels) ]
        )
        
        # get matching boxes based on matching indices
        batch_idx = torch.cat([torch.full_like(pred, i) for i, (pred, _) in enumerate(matching_indices)])
        pred_idx = torch.cat([pred for (pred, _) in matching_indices])
        src_boxes = pred_boxes[(batch_idx, pred_idx)]
        tgt_boxes = torch.cat([gt_bbox[gt_idx] for gt_bbox, (_, gt_idx) in zip(gt_boxes, matching_indices)], dim=0)
        
        num_boxes = sum([ len(x) for x in gt_labels])
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            center_to_corners(src_boxes),
            center_to_corners(tgt_boxes)))
        loss_giou = loss_giou.sum() / num_boxes
        
        tgt_labels_tmp = torch.cat([gt_l[gt_idx] \
            for gt_l, (_, gt_idx) in zip(gt_labels, matching_indices)]
        )
        # considering 0 as crowd
        tgt_labels = torch.zeros(
                        pred_logits.shape[:2], 
                        dtype=torch.int64,
                        device=pred_logits.device
                    )
        tgt_labels[(batch_idx, pred_idx)] = tgt_labels_tmp
        loss_labels = F.binary_cross_entropy_with_logits(pred_logits, tgt_labels)
        
        loss = self.bbox_loss_coef * loss_bbox \
            + self.giou_loss_coef * loss_giou \
            + self.label_loss_coef * loss_labels
        
        return loss