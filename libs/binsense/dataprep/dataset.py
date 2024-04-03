from binsense.owlv2 import Owlv2ImageProcessor
from binsense.config import BIN_S3_DOWNLOAD_IMAGES_DIR as IMG_DIR
from binsense.owlv2 import hugg_loader as hloader
from binsense.img_utils import center_to_corners, scale_bboxes
from binsense.dataset_util import Dataset as BinsenseDataset
from binsense.owlv2 import Owlv2ImageProcessor

from torch.utils.data import Dataset as TorchDataset
from typing import Union, List, Tuple, Any, Dict
from PIL.Image import Image as PILImage

import numpy as np
import pandas as pd
import torch, os, PIL, types

# TODO: do the pre-processing on a batch instead of each image
class ImageProcessor:
    def __init__(self) -> None:
        pass
    
    def preprocess(
        self, 
        images: Union[ List[PILImage], np.ndarray], 
        *args, **kwargs) -> Tuple[torch.Tensor]:
        pass
    
    def postprocess(self, *args, **kwargs) -> Any:
        pass

class OwlImageProcessor(ImageProcessor):
    def __init__(self, owl_config: Dict[str, Any] = None) -> None:
        super(OwlImageProcessor, self).__init__()
        if owl_config is None:
            owl_config = hloader.load_owlv2processor_config()
        self.processor = Owlv2ImageProcessor(**owl_config)
        
    def preprocess(self, images: List[PILImage] | np.ndarray, *args, **kwargs) -> Tuple[torch.Tensor]:
        return self.processor.preprocess(images)["pixel_values"]
        
class BinDataset(TorchDataset):
    def __init__(
        self, 
        image_names: Union[List[str], np.array],
        preprocessor: ImageProcessor = None,
        images_dir: str = IMG_DIR
    ) -> None:
        
        super(BinDataset, self).__init__()
        self.image_names = image_names
        self.file_paths = [os.path.join(images_dir, name) for name in image_names]
        if preprocessor is None:
            preprocessor = OwlImageProcessor()
        self.processor = preprocessor
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        fp = self.file_paths[index]
        img = PIL.Image.open(fp)
        return (self.processor.preprocess(img))

class BestBBoxDataset(TorchDataset):
    def __init__(
        self, 
        best_bbox_df: pd.DataFrame, 
        data_ds: BinsenseDataset,
        processor: ImageProcessor = None
    ) -> None:
        
        super(BestBBoxDataset, self).__init__()
        self.best_bbox_df = best_bbox_df
        self.data_ds = data_ds
        if processor is None:
            processor = OwlImageProcessor()
        self.processor = processor
    
    def __len__(self):
        return self.best_bbox_df.shape[0]
    
    def _get_bbox_center(self, index) -> np.array:
        bbox_rec = self.best_bbox_df.iloc[index].to_dict()
        bbox_data = self.data_ds.get_bboxes(bbox_rec['image_name'])[bbox_rec['bbox_idx']]
        #util functions need array of bboxes
        bbox_center = np.array([[
            bbox_data.center_x, bbox_data.center_y, 
            bbox_data.width, bbox_data.height
        ]])
        return bbox_center, bbox_rec['image_path']
    
    def _crop_bbox(self, bbox_center, img_path) -> np.array:
        img_pil = PIL.Image.open(img_path)
        img_size = (img_pil.width, img_pil.height)
        bbox_corners = center_to_corners(bbox_center)
        bbox_corners = scale_bboxes(bbox_corners, img_size)
        img_bbox = img_pil.crop(tuple(bbox_corners[0]))
        return img_bbox
    
    def __getitem__(self, index) -> torch.FloatTensor:
        bbox_center, img_path = self._get_bbox_center(index)
        img_bbox = self._crop_bbox(bbox_center, img_path)
        bbox_pixels = self.processor.preprocess(img_bbox)
        return (bbox_pixels)
