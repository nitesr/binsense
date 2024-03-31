from binsense.owlv2 import Owlv2ImageProcessor
from binsense.config import BIN_S3_DOWNLOAD_IMAGES_DIR as IMG_DIR
from binsense.owlv2 import hugg_loader as hloader

from torch.utils.data import Dataset
from typing import Union, List, Tuple

import numpy as np
import torch, os, PIL

class BinDataset(Dataset):
    def __init__(self, image_names: Union[List[str], np.array], images_dir: str = IMG_DIR) -> None:
        super().__init__()
        self.image_names = image_names
        self.file_paths = [os.path.join(images_dir, name) for name in image_names]
        self.processor = Owlv2ImageProcessor(**hloader.load_owlv2processor_config())
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        fp = self.file_paths[index]
        img = PIL.Image.open(fp)
        return self.processor.preprocess(img)["pixel_values"][0]
