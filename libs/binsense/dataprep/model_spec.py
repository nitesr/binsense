from .dataset import ImageProcessor

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

class BBoxEmbedder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BBoxEmbedder, self).__init__(*args, **kwargs)
    
    def processor(self) -> ImageProcessor:
        pass
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        pass
    
