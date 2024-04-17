from ..dataprep.dataset import ImageProcessor, OwlImageProcessor
from ..owlv2 import Owlv2ForObjectDetection
from .spec import ImageEmbedder, ObjectDetector, InImageQuerier

from torch import Tensor
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

class OwlV2InImageQuerier(InImageQuerier):
    def __init__(self, model: Owlv2ForObjectDetection, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        
        # freeze the ViT
        for p in self.model.vision_model.parameters():
            p.requires_grad = False
        
        #freeze the class embed parameters
        for p in self.model.class_head.dense0.parameters():
            p.requires_grad = False
        
        self._processor = OwlImageProcessor()
        
    def processor(self) -> ImageProcessor:
        return self._processor
    
    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        o = self.model(inputs['image'], inputs['query'])
        return {'pred_logits': o.pred_logits, 'pred_boxes': o.pred_boxes}
