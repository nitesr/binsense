from typing import Tuple, Dict, Any
from torch import nn, Tensor

class ImageEmbedder(nn.Module):
    def __init__(self, encoder: nn.Module, *args, **kwargs) -> None:
        super(ImageEmbedder, self).__init__(*args, **kwargs)
    
    def forward(x: Tensor) -> Tuple[Tensor]:
        """
        Args:
            x (`Tensor`): 
            batch of transformed item images
        Returns:
            logits (`Tensor`): classification logits
            embeddings (`Tensor`): embeddings
        """
        # implement in subclass
        pass

class InImageQuerier(nn.Module):
    def __init__(self, encoder: nn.Module, *args, **kwargs) -> None:
        super(InImageQuerier, self).__init__(*args, **kwargs)
    
    def forward(x: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            x (`Tuple[Tensor]`): 
                0: batch of transformed bin images to be queried upon
                1: batch of query embeddings
        Returns:
            logits (`Tensor`): classification logits
            bboxes (`Tensor`): bounding boxes
        """
        # implement in subclass
        pass

class MultiBoxLoss(nn.Module):
    def forward(self, pred_boxes, class_logits, gt_boxes, gt_labels) -> Tensor:
        """
        calculate loss as a function of localization and classification loss
        Args:
            pred_boxes (`Tensor`): predicted boxes based on priors/anchors
                e.g. for owlv2 with 16x16 patch size and 960x960 image
                    there are 3600 priors/anchors. so pred_boxes size 
                    will be B x 3600 x 4
            class_logits (`Tensor`): logits from classification head based
                on priors/anchors & num of classes
                e.g. for owlv2 the class_logits size is B x 3600 x C
                    where C is num of classes (all zeros can be 
                        considered as backgroup).
                e.g. for an InImageQuerier we are finding if the object
                    in the image exists or not, so the size is B x 3600
            gt_boxes (`Tensor`): ground truth bounding boxes
            gt_labels (`Tensor`): ground truth classes/labels 
                for each bounding box.
        Returns:
            loss (`Tensor`): a scalar value
        """
        pass
    