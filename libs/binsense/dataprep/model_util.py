from ..owlv2 import Owlv2ForObjectDetection
from torch import nn

import logging, torch

class BboxPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BboxPredictor, self).__init__(*args, **kwargs)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

class Owlv2BboxPredictor(BboxPredictor):
    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super(Owlv2BboxPredictor, self).__init__()
        self.model = model
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images(`torch.Tensor`): images in BxCxWxH format
        Return:
            bbox_logits(`torch.Tensor`):
            scores(`torch.Tensor`):
            bboxes(`torch.Tensor`):
        """
        image_embeds, _ = self.model.image_embedder(images)
        bbox_logits  = self.model.objectness_predictor(image_embeds)
        bboxes = self.model.box_predictor(image_embeds)
        
        probs = torch.sort(bbox_logits.detach(), descending=True)
        bboxes = bboxes[torch.arange(bboxes.size(0)).unsqueeze(1), probs.indices]
        scores = torch.sigmoid(probs.values)
        return bbox_logits, scores, bboxes