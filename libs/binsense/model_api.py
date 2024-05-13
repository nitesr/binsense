import PIL.Image
from .embed_datastore import EmbeddingDatastore
from .owlv2 import Owlv2ImageProcessor, Owlv2ForObjectDetection
from typing import List
from PIL.Image import Image as PILImage
from .img_utils import annotate_image

import numpy as np

import torch, os, PIL


class BinImageFinder:
    def __init__(self, images_dir: str) -> None:
        if not os.path.exists(images_dir):
            raise ValueError(f"can't find path {images_dir}")
        self.images_dir = images_dir
    
    def _resolve(self, bin_id: str) -> str:
        return os.path.join(self.images_dir, f'{bin_id}.jpg')
    
    def has(self, bin_id: str) -> bool:
        return os.path.exists(self._resolve(bin_id))
    
    def to_PIL(self, bin_id: str) -> PILImage:
        if not self.has(bin_id):
            raise ValueError(f"can't find the image for {bin_id}")
        return PIL.Image.open(self._resolve(bin_id))

class BinPreprocessor:
    def __init__(self) -> None:
        pass
    
class InBinQuerier:
    def __init__(self) -> None:
        pass

class OwlBinPreprocessor(BinPreprocessor):
    def __init__(self, preprocessor: Owlv2ImageProcessor) -> None:
        super(OwlBinPreprocessor, self).__init__()
        self.processor = preprocessor
        
    def __call__(self, image) -> torch.Tensor:
        return self.processor.preprocess(image)["pixel_values"]

class OwlImageQuerier(InBinQuerier):
    def __init__(
        self, 
        model: Owlv2ForObjectDetection, 
        threshold: float = 0.995,
        device: str = 'cpu') -> None:
        super(OwlImageQuerier, self).__init__()
        self.threshold = threshold
        self.nms_threshold = 1
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.processor = Owlv2ImageProcessor()
    
    def _get_image_size(self, target_pixels: torch.Tensor) -> torch.Tensor:
        img_size = [x for x in target_pixels.shape[2:]]
        return torch.tensor(img_size, dtype=torch.int64)
    
    def _annotate(self, target_pixels: torch.Tensor, pred_boxes: List) -> PILImage:
        img_pixels = self.processor.unnormalize_pixels(target_pixels)[0]
        img = PIL.Image.fromarray(img_pixels)
        boxes_cxy=np.vstack(pred_boxes)

        if len(boxes_cxy) > 0:
            return annotate_image(
                img, 
                labels=None, 
                bboxes_cxy=boxes_cxy, 
                seg_coords=None, 
                colors=[(0,255,255)]*len(boxes_cxy),
                normalize=False,
                convert_cxy_xy=False)
        else:
            return img
        
    def __call__(self, target_pixels: torch.Tensor, query_embeds: np.ndarray) -> np.array:
        if target_pixels.shape[0] != query_embeds.shape[0]:
            raise ValueError("num of queries doesn't match keys is allowed!")
        
        n = len(target_pixels)
        query_embeds = torch.tensor(query_embeds, dtype=torch.float32)
        img_sizes = self._get_image_size(target_pixels).expand(n, -1)
        with torch.no_grad():
            target_pixels = target_pixels.to(self.device)
            query_embeds = query_embeds.to(self.device)
            outputs = self.model(target_pixels, query_embeds)

            results = self.processor.post_process_image_guided_detection(
                outputs, self.threshold, self.nms_threshold, 
                target_sizes=img_sizes)

            pred_boxes = [ r["boxes"].cpu().numpy() for r in results ]
            ann_image = self._annotate(target_pixels, pred_boxes)
            return pred_boxes, ann_image

class ModelApi:
    def __init__(
        self, 
        model: InBinQuerier, 
        preprocessor: BinPreprocessor,
        embed_ds: EmbeddingDatastore, 
        bin_images_dir: str = None) -> None:
        
        self.model = model
        self.preprocessor = preprocessor
        self.embed_ds = embed_ds
        self.img_finder = BinImageFinder(bin_images_dir) if bin_images_dir is not None else None
        
    def find_item_qts_in_binimage(self, item_ids: List[str], bin_image: PILImage) -> List[int]:
        qts = []
        qs = []
        for i_i in item_ids:
            if self.embed_ds.has(i_i):
                qts.append(-1)
                qs.append(self.embed_ds.get(i_i))
            else:
                qts.append(0)

        if len(qs) == 0:
            return qts

        pred_boxes = []
        qs = np.array(qs)
        k = self.preprocessor(bin_image)
        ks = k.expand(len(qs), *[s for s in k.shape[1:]])
        vs, ann_image = self.model(ks, qs)
        vi = 0
        for i, qt in enumerate(qts):
            if qt == -1:
                qts[i] = len(vs[vi])
                pred_boxes.append(vs[vi])
                vi += 1
            else:
                pred_boxes.append([])
        return qts, pred_boxes, ann_image

    def find_item_qts_in_bin(self, item_ids: List[str], bin_id: str) -> List[int]:
        img = self.img_finder.to_PIL(bin_id)
        return self.find_item_qts_in_binimage(item_ids, img)
    
    def find_item_qty_in_bin(self, item_id: str, bin_id: str) -> int:
        qts, _, _ = self.find_item_qts_in_bin([item_id], bin_id)
        return qts[0]
    
    def is_item_qty_exist_in_bin(self, item_id: str, item_qty: int, bin_id: str) -> bool:
        return self.find_item_qty_in_bin(item_id, item_qty, bin_id) >= item_qty
    
    def is_item_exist_in_bin(self, item_id: str, bin_id: str) -> bool:
        return self.is_item_qty_exist_in_bin(item_id, 1, bin_id)
    
    