import torch.utils
import torch.utils.data
from ..dataprep.dataset import ImageProcessor

from typing import Tuple, Dict, Any
from torch import nn, Tensor

import logging, torch

logger = logging.getLogger(__name__)

class ObjectDetector(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ObjectDetector, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (`Tensor`):  batch of transformed item images
        Returns:
            bbox_logits(`Tensor`):
            bboxes(`Tensor`):
        """
        # implement in subclass
        pass

class ImageEmbedder(nn.Module):
    def __init__(self, encoder: nn.Module, *args, **kwargs) -> None:
        super(ImageEmbedder, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def get_embed_size(self) -> int:
        pass
    
    def forward(x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (`Tensor`): batch of transformed item images
        Returns:
            logits (`Tensor`): classification logits
            embeddings (`Tensor`): embeddings
        """
        # implement in subclass
        pass

class InImageQuerier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(InImageQuerier, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            inputs['image (`Tensor`)']: 
                batch of transformed bin images to be queried upon
                size: B x C x W x H
            inputs['query (`Tensor`)']:
                batch of query embeddings
                size: B x NUM_QUERIES x EMBED_SIZE
        Returns:
            outputs['pred_logits (`Tensor`)']: 
                predicted class logits
                size: B x NUM_PATCHES x (NUM_QUERIES+1)
            outputs['pred_bboxes (`Tensor`)']: 
                predicted bounding boxes
                size: B x NUM_PATCHES x 4
        """
        # implement in subclass
        pass

class MultiBoxLoss(nn.Module):
    def forward(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        calculate loss as a function of localization and classification loss
        Args:
            outputs['pred_boxes (`Tensor`)']: predicted boxes based on priors/anchors
                e.g. for owlv2 with 16x16 patch size and 960x960 image
                    there are 3600 priors/anchors. so pred_boxes size 
                    will be B x 3600 x 4
            outputs['pred_logits (`Tensor`)']: logits from classification head based
                on priors/anchors & num of classes
                e.g. for owlv2 the class_logits size is B x 3600 x C
                    where C is num of classes (all zeros can be 
                        considered as backgroup).
                e.g. for an InImageQuerier we are finding if the object
                    in the image exists or not, so the size is B x 3600
            outputs['boxes (`Tensor`)']: ground truth bounding boxes
            outputs['labels (`Tensor`)']: ground truth classes/labels 
                for each bounding box.
        Returns:
            losses['loss (`Tensor`)']: weighted loss (a scalar value)
            losses['loss_label (`Tensor`)']: classification loss (a scalar value)
            losses['loss_reg (`Tensor`)']: bbox coords regression loss (a scalar value)
            losses['loss_giou (`Tensor`)']: bbox iou loss (a scalar value)
        """
        pass
    