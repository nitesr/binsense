from binsense.owlv2 import Owlv2ImageProcessor
from binsense.config import BIN_S3_DOWNLOAD_IMAGES_DIR as IMG_DIR
from binsense.owlv2 import hugg_loader as hloader
from binsense.img_utils import center_to_corners, scale_bboxes, create_polygon_mask
from binsense.dataset_util import Dataset as BinsenseDataset
from binsense.dataset_util import BoundingBox
from binsense.owlv2 import Owlv2ImageProcessor

from torchvision.tv_tensors import Mask
from torchvision.ops import box_convert
import torchvision.transforms.v2  as transforms

from torch.utils.data import Dataset as TorchDataset
from typing import Union, List, Tuple, Any, Dict
from PIL.Image import Image as PILImage

import numpy as np
import pandas as pd
import torch, os, PIL, cv2

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

    def resize_boxes_to_original_size(
        self, 
        boxes: torch.Tensor,
        orig_sizes: torch.Tensor) -> torch.Tensor:
        pass

class OwlImageProcessor(ImageProcessor):
    def __init__(self, owl_config: Dict[str, Any] = None) -> None:
        super(OwlImageProcessor, self).__init__()
        if owl_config is None:
            owl_config = hloader.load_owlv2processor_config()
        self.processor = Owlv2ImageProcessor(**owl_config)
        
    def preprocess(self, images: List[PILImage] | np.ndarray, *args, **kwargs) -> Tuple[torch.Tensor]:
        return self.processor.preprocess(images)["pixel_values"][0]
    
    def resize_boxes_to_original_size(
            self, boxes: torch.Tensor, 
            orig_sizes: torch.Tensor) -> torch.Tensor:
        return self.processor.resize_boxes_to_original_size(boxes, orig_sizes)
    
    def unnormalize_pixels(
        self, pixels: torch.Tensor
    ):
        return self.processor.unnormalize_pixels(pixels)
        
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
        return (
            torch.tensor([index], dtype=torch.int32), 
            torch.tensor([img.width, img.height], dtype=torch.int32),
            self.processor.preprocess(img))

class BestBBoxDataset(TorchDataset):
    def __init__(
        self, 
        best_bbox_df: pd.DataFrame, 
        data_ds: BinsenseDataset,
        processor: ImageProcessor = None,
        crop_by_segment: bool = True,
    ) -> None:
        
        super(BestBBoxDataset, self).__init__()
        self.best_bbox_df = best_bbox_df
        self.data_ds = data_ds
        if processor is None:
            processor = OwlImageProcessor()
        self.processor = processor
        self.crop_by_segment = crop_by_segment
    
    def __len__(self):
        return self.best_bbox_df.shape[0]
    
    def _get_bbox(self, index) -> Tuple[BoundingBox, str]:
        bbox_rec = self.best_bbox_df.iloc[index].to_dict()
        bbox_data = self.data_ds.get_bboxes(bbox_rec['image_name'])[bbox_rec['bbox_idx']]
        return bbox_data, bbox_rec['image_path']
    
    def _crop_bbox(self, bbox_data: BoundingBox, img_path) -> np.array:
        img_pil = PIL.Image.open(img_path)
        img_size = (img_pil.width, img_pil.height)
        bbox_corners = center_to_corners(np.array([bbox_data.to_array()]))
        bbox_corners = scale_bboxes(bbox_corners, img_size)
        img_bbox = img_pil.crop(tuple(bbox_corners[0]))
        return img_bbox
    
    def _crop_segment(self, bbox_data: BoundingBox, img_path) -> np.array:
        cv_img = cv2.imread(img_path, 1)
        img_size = (cv_img.shape[1], cv_img.shape[0])
        bbox_corners = center_to_corners(np.array([bbox_data.to_array()]))
        bbox_corners = scale_bboxes(bbox_corners, img_size)

        segment = bbox_data.segmentation[0]
        segment[:,0] = segment[:,0] * cv_img.shape[1]
        segment[:,1] = segment[:,1] * cv_img.shape[0]
        segment = segment.astype("int")

        # create mask based on polygon (segment)
        # cv_img = np.array(query_image)
        mask = np.zeros_like(cv_img, dtype = "uint8")
        cv2.fillConvexPoly(mask, segment, (1, 1, 1))
        mask = mask > 0

        # apply mask on the image to extract segment
        # owl uses (255/2, 255/2, 255/2) for padding
        out_img = np.full_like(cv_img, 255/2, dtype = "uint8")
        out_img[mask] = cv_img[mask]

        # crop the image
        out_img = cv2.cvtColor(out_img, code=cv2.COLOR_BGR2RGB)
        img_bbox = PIL.Image.fromarray(out_img)
        img_bbox = img_bbox.crop(tuple(bbox_corners[0]))
        return img_bbox
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bbox_data, img_path = self._get_bbox(index)
        img_bbox = self._crop_segment(bbox_data, img_path) if self.crop_by_segment else self._crop_bbox(bbox_data, img_path)
        bbox_pixels = self.processor.preprocess(img_bbox)
        
        # query_bbox = self.renorm_cropped_box(
        #     img_size, 
        #     (bbox_pixels.shape[1], bbox_pixels.shape[2]))

        max_len = max(img_bbox.size)
        query_bbox = np.array([img_bbox.size[0]/2, img_bbox.size[1]/2, img_bbox.size[0], img_bbox.size[1]], dtype=np.float32)
        query_bbox /= max_len

        return (
            torch.tensor([index], dtype=torch.int32), 
            torch.as_tensor(query_bbox, dtype=torch.float32), 
            bbox_pixels
        )
