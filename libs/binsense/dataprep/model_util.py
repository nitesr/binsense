from .dataset import ImageProcessor, OwlImageProcessor

from ..owlv2 import Owlv2ForObjectDetection
from torch import nn

import logging, torch

logger = logging.getLogger(__name__)

class BboxPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BboxPredictor, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass

class Owlv2BboxPredictor(BboxPredictor):
    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super(Owlv2BboxPredictor, self).__init__()
        self.model = model
        self._processor = OwlImageProcessor()
    
    def processor(self) -> ImageProcessor:
        return self._processor
    
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

class BBoxEmbedder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BBoxEmbedder, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        pass
    
class Owlv2BBoxEmbedder(BBoxEmbedder):
    def __init__(self, model: Owlv2ForObjectDetection, *args, **kwargs) -> None:
        super(Owlv2BBoxEmbedder, self).__init__(*args, **kwargs)
        self.model = model
        self._processor = OwlImageProcessor()
    
    def processor(self) -> ImageProcessor:
        return self._processor
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        vit_embeddings = self.model.image_embedder(bboxes)[0] # B x C x 960 x 960 -> B x 3600 x 768
        class_embeddings = self.model.class_predictor(vit_embeddings)[1] # B x 3600 x 768 -> B x 3600 x 512
        object_logits = self.model.objectness_predictor(vit_embeddings) # B x 3600 x 768 -> B x 3600
        del vit_embeddings
        
        object_scores = torch.sigmoid(object_logits)
        _, top_idxs = torch.max(object_scores, dim=1, keepdim=True)
        bbox_embeddings = torch.take_along_dim(class_embeddings, top_idxs[...,None], dim=1)
        return bbox_embeddings.squeeze(dim=1)