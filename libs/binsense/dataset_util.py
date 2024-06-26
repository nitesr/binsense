from .utils import FileIterator, backup_file
from .img_utils import convert_xy_cxy_and_unscale
from .img_utils import corner_to_centers

from pathlib import Path

from xml.etree.ElementTree import indent, parse, ElementTree, Element, SubElement, tostring
from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

import numpy as np
import json, os, shutil, yaml, logging, cv2

from typing import List, Dict, Tuple, Union, Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    label: str = None
    center_x: float = None
    center_y: float = None
    width: float = None
    height: float = None
    area: float = None
    segmentation: List[np.ndarray] = None
    normalized: bool = True
    _label_id: int = None
    _img_id: int = None
    
    def to_array(self):
        return np.array([self.center_x, self.center_y, self.width, self.height])

class DataTag(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'

@dataclass
class ImageData:
    id: int = None
    name: str = None
    path: str = None
    tag: DataTag = DataTag.TRAIN
    width: float = 0
    height: float = 1
    normalized: bool = True

@dataclass
class LabelData:
    id: int = None
    name: str = None

class Dataset:
    """
    Marker class to declare the methods across all Dataset(s).
    """
    def __init__(self) -> None:
        pass
    
    def get_images(self, tag: DataTag = DataTag.TRAIN) -> List[ImageData]:
        pass
    
    def get_bboxes(self, img_name: str) -> List[BoundingBox]:
        pass
    
    def get_labels(self, img_name: str) -> List[LabelData]:
        pass

    def get_all_labels(self) -> List[LabelData]:
        pass

    def get_all_images(self) -> List[ImageData]:
        pass
    
    def to_file(self, file_path: str, format: str = None, exclude_tags: List[DataTag] = []) -> None:
        pass


class DatasetBuilder:
    """
    Marker class to declare the methods across all the DatasetBuilder(s).
    Usage:
        builder = Yolov8DatasetBuilder()
        cat_dict = build.add_categories(['cat', 'dog', 'wolf'])
        img_id = builder.add_image('img_path', 'train')
        builder.add_bboxes(img_id, cat_ids, bboxes)
        
    """
    def __init__(self) -> None:
        pass

    def image_exists(self, img_path: str, tag: DataTag = DataTag.TRAIN) -> bool:
        """
        should check if the image exists in the dataset

        Args:
            img_path (`str`): complete file path to the image.
            tag(`str`): train/test/valid tag. default is train
        Returns:
            exists (`bool`):
        """
        pass
    def add_image(self, img_path: str, tag: DataTag = DataTag.TRAIN) -> int:
        """
        should add the image to the dataset

        Args:
            img_path (`str`): complete file path to the image.
            tag(`str`): train/test/valid tag. default is train
        Returns:
            img_id (`int`):
        """
        pass
    
    def add_category(self, category_name: str) -> int:
        """
        adds the category/label to the dataset, 
        in case of duplicates returns the existing id.

        Args:
            category_name (`str`):
        Returns:
            catetory_id (`int`):
        """
        pass
    
    def add_categories(self, category_names: List[str]) -> Dict[str, int]:
        """
        adds the categories/labels to the dataset.
        Args:
            category_names (`List[str]`):
        Returns:
            catetory_ids (`Dict[str, int]`):
        """
        pass
    
    def add_bbox(self, img_id: int, category_id: int, bbox: np.ndarray=None):
        """
        adds the bounding box annotation to the dataset,
        doesn't check for duplicates
        Args:
            img_id (`int`): image id
            catetory_id (`int`): category id
            bbox (`np.ndarray`): bbox coordinates,
                should be cx, xy, w, h and normalized
        """
        pass
    
    def add_bboxes(self, img_id: int, category_ids: List[int], bboxes: np.ndarray=None):
        """
        adds the bounding box annotation to the dataset.
        raises ValueError if len(category_ids) != len(bboxes)
        doesn't check for duplicates
        Args:
        """
        pass
    
    def build(self) -> Dataset:
        pass

class YoloDataset(Dataset):
    def __init__(
        self, 
        images: Dict[str, ImageData], 
        categories: Dict[str, LabelData], 
        bboxes: Dict[int, List[BoundingBox]], 
    ) -> None:
        super(YoloDataset, self).__init__()
        self.images = images
        self.categories = categories
        self.bboxes = bboxes
    
    def get_images(self, tag: DataTag = DataTag.TRAIN) -> List[ImageData]:
        filtered_images = []
        for key in self.images.keys():
            if self.images[key].tag == tag:
                filtered_images.append(self.images[key])
        return filtered_images
    
    def _get_image(self, image_name: str) -> ImageData:
        if image_name not in self.images.keys():
            raise ValueError(f'image {image_name} is not found!')
        return self.images[image_name]

    def get_bboxes(self, img_name: str) -> List[BoundingBox]:
        image_data = self._get_image(img_name)
        return self.bboxes[image_data.id]
    
    def get_labels(self, img_name: str) -> List[LabelData]:
        bboxes = self.get_bboxes(img_name)
        filtered_labels  = []
        for bbox in bboxes:
            filtered_labels.append(self.categories[bbox.label])
        return filtered_labels
    
    def get_all_labels(self) -> List[LabelData]:
        return [v for _, v in self.categories.items()]
    
    def get_all_images(self) -> List[ImageData]:
        return [v for _, v in self.images.items()]
    
    def to_file(self, file_path: str, format: str = 'yolov8', exclude_tags: List[DataTag] = []) -> None:
        if format == 'yolov8':
            Yolov8Serializer(self).to_file(file_path, exclude_tags)
        elif format == 'yolov1':
            Yolov1Serializer(self).to_file(file_path, exclude_tags)
        else:
            raise ValueError(f'{format} is not supported!')

class YoloDeserializer:
    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
    
    def read(self) -> YoloDataset:
        pass

class Yolov8Deserializer(YoloDeserializer):
    def __init__(
        self, 
        dir_path: str, 
        image_name_extractor: Callable[[str], str] = None,
        label_name_extractor: Callable[[str], str] = None,
        img_extns: List[str] = None) -> None:
        super(Yolov8Deserializer, self).__init__(dir_path)
        
        if img_extns is None:
            img_extns = ['.jpg', '.jpeg', '.png', '.tif']
        if image_name_extractor is None:
            image_name_extractor = lambda x: x
        if label_name_extractor is None:
            label_name_extractor = lambda x: x
        
        self.builder = YoloDatasetBuilder()
        self.img_extns = img_extns
        self.image_name_extractor = image_name_extractor
        self.label_name_extractor = label_name_extractor
        

    def _read_index(self, index_file: str) -> Tuple[str, str, str]:
        with open(index_file, 'r') as f:
            data = yaml.safe_load(f)
        # label_count = data['nc']
        labels =  data['names'].values() if isinstance(data['names'], dict) else data['names']
        clean_labels = []
        for l in labels:
            clean_l = self.label_name_extractor(l)
            if clean_l is None or len(clean_l) == 0:
                logger.error(f'extracted label({clean_l}) is empty for {l}')
                continue
            clean_labels.append(clean_l)
        
        self.builder.add_categories(clean_labels)
        
        def _to_abs_path(ds_path: str):
            return ds_path.replace('..', self.dir_path)
            
        return {\
            DataTag.TRAIN: _to_abs_path(data['train']) if 'train' in data.keys() else None, 
            DataTag.VALID: _to_abs_path(data['val']) if 'val' in data.keys() else None, 
            DataTag.TEST: _to_abs_path(data['test']) if 'test' in data.keys() else None
        }
    
    def _interpret_ann_line(self, line: str, img_id: int) -> None:
        tokens = [ t.strip() for t in line.split()]
        label_id = tokens[0]

        self.builder.add_bbox(
            img_id, int(label_id),
            np.array([np.float32(t) for t in tokens[1:]])
        )
        
    def _read_ann_file(self, img_id: int, ann_path: str) -> None:
        with open(ann_path, 'r') as f:
            line = f.readline()
            while line:
                if not line.startswith('#'):
                    self._interpret_ann_line(line, img_id)
                line = f.readline()
        
    def _read_data(self, ds_path: str, tag: DataTag) -> None:
        ds_dir, imgdir_name = os.path.split(ds_path)
        if imgdir_name != 'images':
            return
        
        chld_ds_dirs = os.listdir(ds_path)
        progress_step = len(chld_ds_dirs) // 10
        progress_bar = tqdm(
            total=len(chld_ds_dirs),
            file=open(os.devnull, 'w'),
            desc=f'reading {tag} files')
        
        for i, imgfile_name in enumerate(chld_ds_dirs):
            name, extn = os.path.splitext(imgfile_name)
            if not extn in self.img_extns:
                continue
            img_path = os.path.join(ds_path, imgfile_name)
            img_name = self.image_name_extractor(imgfile_name)
            if img_name is None or len(img_name) == 0:
                logger.error(f'extracted image name({img_name}) is empty for {imgfile_name}')
                continue
            if self.builder.image_exists(img_path, tag, img_name):
                logger.error(f'duplicate image name({img_name}) with path({img_path})')
                continue

            img_id = self.builder.add_image(img_path, tag, img_name)
            
            lbl_path = os.path.join(ds_dir, 'labels', f'{name}.txt')
            if not os.path.exists(lbl_path):
                lbl_path = os.path.join(ds_dir, 'images', f'{name}.txt')
            if not os.path.exists(lbl_path):
                logger.info(f'label not found for {imgfile_name} @ {ds_dir}')
                continue
            
            self._read_ann_file(img_id, lbl_path)
            progress_bar.update(1)
            if i >= progress_step:
                progress_step += i
                logger.info(str(progress_bar))
        logger.info(str(progress_bar))
    
    def read(self) -> YoloDataset:
        ds_dict = self._read_index(os.path.join(self.dir_path, 'data.yaml'))
        
        for tag, ds_path in ds_dict.items():
            if ds_path and os.path.exists(ds_path):
                self._read_data(ds_path, tag)
            elif ds_path:
                logger.warn(f'missing dataset path - {ds_path}')
            else:
                logger.info(f'missing dataset for {tag}')
        
        return self.builder.build()

class YoloSerializer:
    def __init__(self, dataset: YoloDataset) -> None:
        self.dataset = dataset
        
    def _get_images(self) -> List[ImageData]:
        return self.dataset.images.values()
    
    def _get_labels(self) -> List[LabelData]:
        return self.dataset.categories.values()
    
    def _get_bboxes(self, img_id: int) -> List[BoundingBox]:
        return self.dataset.bboxes[img_id]
    
    def _prepare_root_path(self, root_path: str) -> None:
        if os.path.exists(root_path):
            bkp_file = backup_file(root_path)
            logger.info(f"backing up {root_path} to {bkp_file}")
            os.makedirs(root_path)
        
    def _make_dir(self, root_path, tag):
        tag_path = os.path.join(root_path, tag)
        tag_images_path = os.path.join(tag_path, 'images')
        os.makedirs(tag_path, exist_ok=True)
        os.makedirs(tag_images_path, exist_ok=True)
        return tag_images_path
    
    def _make_tag_dirs(self, root_path):
        return ( \
            self._make_dir(root_path, "train"),
            self._make_dir(root_path, "valid"),
            self._make_dir(root_path, "test")
        )
    
    def _images_path(self, root_path, tag: DataTag):
        tag_name = 'train' if tag == DataTag.TRAIN \
            else 'valid' if tag == DataTag.VALID \
                else 'test'
        return os.path.join(root_path, tag_name, 'images')
    
    def _move_image(self, root_path, img_data: ImageData) -> str:
        target_dir = self._images_path(root_path, img_data.tag)
        target_path = os.path.join(target_dir, img_data.name)
        shutil.copyfile(img_data.path, target_path)
        return target_path
    
    def _create_ann(self, root_path, img_data: ImageData, bboxes: List[BoundingBox]) -> str:
        target_path = self._images_path(root_path, img_data.tag)
        pfx = os.path.splitext(img_data.name)[0]
        ann_file = os.path.join(target_path, f'{pfx}.txt')
        with open(ann_file, 'w') as f:
            for bbox in bboxes:
                f.write(' '.join([str(x) \
                    for x in [ 
                            bbox._label_id, 
                            *(bbox.to_array() if bbox.segmentation is None else bbox.segmentation[0].flatten())
                    ]
                ]))
                f.write('\n')
        return ann_file
    
    def to_file(self, file_path: str, exclude_tags: List[DataTag] = []) -> None:
        pass

class Yolov1Serializer(YoloSerializer):
    def __init__(self, dataset: YoloDataset) -> None:
        super(Yolov1Serializer, self).__init__(dataset)
        
    def _create_index(self, root_path: str, labels: List[LabelData], img_path_dict: Dict[DataTag, List[str]]):
        labels = sorted(labels, key=lambda l: l.id)
        
        label_file = os.path.join(root_path, 'obj.names')
        with open(label_file, 'w') as f:
            for l in labels:
                f.write(l)
                f.write('\n')
        
        data_file = os.path.join(root_path, 'obj.data')
        with open(data_file, 'w') as f:
            f.write(f'classes = {len(labels)}\n')
            f.write('train = data/train.txt\n')
            f.write('valid = data/valid.txt\n')
            f.write('names = data/obj.names\n')
        
        train_file = os.path.join(root_path, 'train.txt')
        train_img_paths = img_path_dict[DataTag.TRAIN]
        with open(train_file, 'w') as f:
            for img_path in train_img_paths:
                _, tail = os.path.split(img_path)
                f.write(f'data/train/images/{tail}')
                f.write('\n')
        
        val_file = os.path.join(root_path, 'valid.txt')
        val_img_paths = img_path_dict[DataTag.VALID]
        with open(val_file, 'w') as f:
            for img_path in val_img_paths:
                _, tail = os.path.split(img_path)
                f.write(f'data/valid/images/{tail}')
                f.write('\n')
    
    def to_file(self, file_path: str, exclude_tags: List[DataTag] = []) -> None:
        root_path = file_path
        self._prepare_root_path(root_path)
        self._make_tag_dirs(root_path)
        
        img_path_dict = dict({DataTag.TRAIN: [], DataTag.VALID: []})
        for img_data in self._get_images():
            if img_data.tag in exclude_tags:
                continue
            bboxes_data = self._get_bboxes(img_data.id)
            img_path = self._move_image(root_path, img_data)
            img_path_dict[img_data.tag].append(img_path)
            self._create_ann(root_path, img_data, bboxes_data)
            
        self._create_index(root_path, self._get_labels(), img_path_dict)

class Yolov8Serializer(YoloSerializer):
    def __init__(self, dataset: YoloDataset) -> None:
        super(Yolov8Serializer, self).__init__(dataset)
    
    def _create_index(self, 
        root_path: str, labels: List[LabelData], 
        exclude_tags: List[DataTag] = []):
        labels = sorted(labels, key=lambda l: l.id)
        
        index_file = os.path.join(root_path, 'data.yaml')
        with open(index_file, 'w') as f:
            if not DataTag.TRAIN in exclude_tags:
                f.write('train: ../train/images')
                f.write('\n')
            if not DataTag.VALID in exclude_tags:
                f.write('val: ../valid/images')
                f.write('\n')
            if not DataTag.TEST in exclude_tags:
                f.write('test: ../test/images')
                f.write('\n')
            f.write('\n')
            f.write('names:\n')
            for i, l in enumerate(labels):
                f.write(f" {i}: '{l.name}'\n")
            f.write('\n')
    
    def to_file(self, file_path: str, exclude_tags: List[DataTag] = []) -> None:
        root_path = file_path
        self._prepare_root_path(root_path)
        self._make_tag_dirs(root_path)
        
        for img_data in self._get_images():
            if img_data.tag in exclude_tags:
                continue
            bboxes_data = self._get_bboxes(img_data.id)
            self._move_image(root_path, img_data)
            self._create_ann(root_path, img_data, bboxes_data)
            
        self._create_index(root_path, self._get_labels(), exclude_tags)

class YoloDatasetBuilder(DatasetBuilder):
    def __init__(self) -> None:
        super(YoloDatasetBuilder, self).__init__()
        self.categories = dict()
        self.images = dict()
        self.bboxes = dict()
    
    def image_exists(self, img_path: str, tag: DataTag = DataTag.TRAIN, img_name: str = None) -> bool:
        if img_name is None:
            img_name = img_name = Path(img_path).name
        return img_name in self.images.keys()

    def add_image(self, img_path: str, tag: DataTag = DataTag.TRAIN, img_name: str = None) -> int:
        if img_name is None:
            img_name = img_name = Path(img_path).name
        if img_name in self.images.keys():
            return self.images[img_name].id
        img_id = len(self.images)
        self.images[img_name] = ImageData(**{
            'id': img_id,
            'name': img_name,
            'path': img_path,
            'tag': tag
        })
        self.bboxes[img_id] = []
        return img_id
    
    def add_category(self, category_name: str) -> int:
        if category_name in self.categories.keys():
            return self.categories[category_name].id
        cat_id = len(self.categories)
        self.categories[category_name] = LabelData(**{
            'id': cat_id,
            'name': category_name
        })
        return cat_id
    
    def add_categories(self, category_names: List[str]) -> Dict[str, int]:
        result = dict()
        for name in category_names:
            id = self.add_category(name)
            result[name] = id
        return result
    
    def add_bbox(self, img_id: int, category_id: int, bbox: np.ndarray = None):
        if img_id not in range(0, len(self.images)) \
            or category_id not in range(0, len(self.categories)):
            raise ValueError('check if image and category is added!')
        
        def _get_label(category_id: str) -> str:
            for name, cat_data in self.categories.items():
                if cat_data.id == category_id:
                    return name
            return None

        def _get_image_name(img_id: str) -> str:
            for name, img_data in self.images.items():
                if img_data.id == img_id:
                    return name
            return None
        
        if len(bbox) > 4:
            if not len(bbox) % 2 == 0:
                raise ValueError(f'segmentation coords len({len(bbox)}) not even, img_id={_get_image_name(img_id)}, label={_get_label(category_id)}')
            coords = bbox.reshape(len(bbox)//2, 2)
            box_xy = np.array([np.min(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,0]), np.max(coords[:,1])])
            box_cxy = corner_to_centers(box_xy)
            # bounding box
            bbox_data = BoundingBox(**{
                'label': _get_label(category_id),
                '_label_id': category_id,
                '_img_id': img_id,
                'center_x': box_cxy[0],
                'center_y': box_cxy[1],
                'width': box_cxy[2],
                'height': box_cxy[3],
                'area': cv2.contourArea(coords),
                'segmentation': [coords]
            })
        else:
            # bounding box
            bbox_data = BoundingBox(**{
                'label': _get_label(category_id),
                '_label_id': category_id,
                '_img_id': img_id,
                'center_x': bbox[0],
                'center_y': bbox[1],
                'width': bbox[2],
                'height': bbox[3],
                'area': bbox[2] * bbox[3]
            })

        self.bboxes[img_id].append(bbox_data)
        
    def add_bboxes(self, img_id: int, category_ids: List[int], bboxes: np.ndarray = None):
        for i, bbox in enumerate(bboxes):
            self.add_bbox(img_id, category_ids[i], bbox)
    
    def build(self) -> YoloDataset:
        return YoloDataset(self.images, self.categories, self.bboxes)

class YoloDatasetCopier:
    def __init__(
            self, 
            src_ds: Dataset, 
            categories: List[str] = None, 
            img_tag_pairs: Dict[str, DataTag] = None) -> None:
        self.src_ds = src_ds
        self.dst_categories = self._to_dst_categories(src_ds) if categories is None else categories
        self.img_tag_pairs = img_tag_pairs
    
    def _to_dst_categories(self, src_ds: Dataset) -> List[str]:
        return [lbl_data.name for lbl_data in src_ds.get_all_labels()]
    
    def _copy_image_bboxes(
        self, 
        dst_ds: DatasetBuilder, 
        src_img_data: ImageData,
        dst_category_dict: Dict[str, int]) -> int:
        
        dst_tag = src_img_data.tag if self.img_tag_pairs is None else DataTag(self.img_tag_pairs[src_img_data.name])
        dst_img_id = dst_ds.add_image(
            src_img_data.path, dst_tag, 
            src_img_data.name)
        bboxes = self.src_ds.get_bboxes(src_img_data.name)
        src_cat_names = [bbox.label for bbox in bboxes]
        src_bbox_arrays = [ bbox.to_array() if bbox.segmentation is None else bbox.segmentation[0].flatten() for bbox in bboxes]
        dst_cat_ids = [dst_category_dict[name] for name in src_cat_names]
        dst_ds.add_bboxes(dst_img_id, dst_cat_ids, src_bbox_arrays)
        return dst_img_id
    
    def copy(self) -> Dataset:
        dst_dsbuild = YoloDatasetBuilder()
        dst_category_dict = dst_dsbuild.add_categories(set(self.dst_categories))
        for img_data in self.src_ds.get_all_images():
            if self.img_tag_pairs is None or img_data.name in self.img_tag_pairs:
                self._copy_image_bboxes(
                    dst_ds=dst_dsbuild,
                    src_img_data=img_data,
                    dst_category_dict=dst_category_dict
                )
            else:
                logger.warn(f"filtering out {img_data.name} while copying to dst_ds")
        
        dst_ds = dst_dsbuild.build()
        def get_ds_count(split_tag: DataTag):
            return len(dst_ds.get_images(split_tag))
        train_cnt = get_ds_count(DataTag.TRAIN)
        val_cnt = get_ds_count(DataTag.VALID)
        test_cnt = get_ds_count(DataTag.TEST)
        total_cnt =  train_cnt +  val_cnt + test_cnt
        logger.info(f'copied dataset train_ratio={round(train_cnt/total_cnt, 2)} valid_ratio={round(val_cnt/total_cnt, 2)} test_ratio={round(test_cnt/total_cnt, 2)}')

        return dst_ds

# -- refactor below code to meet the interfaces ----------------

class COCODataset(Dataset):
    def __init__(
        self, data: dict, tag: DataTag = DataTag.TRAIN,
        image_name_extractor: Callable[[str], str] = None,
        label_name_extractor: Callable[[str], str] = None
    ) -> None:
        super(COCODataset, self).__init__()
        self.datasets = {}
        self.datasets[tag] = data
        self.data = data
        self.image_name_extractor = image_name_extractor
        self.label_name_extractor = label_name_extractor
    
    def _check_if_exists(self, value: str, lst: list, attr_name: str = 'name') -> Tuple[bool, int]:
        for i, r in enumerate(lst):
            if r[attr_name] == value:
                return (True, i)
            
        return (False, -1)
    
    def _get_label(self, annotation_id: int) -> str:
        if len(self.data["annotations"]) > annotation_id:
            cat_id = self.data["annotations"][annotation_id]["category_id"]
            if cat_id:
                return self.data["categories"][cat_id]["name"]
        
        return None
    
    def _extract(self, value, extractor):
        return extractor(value) if extractor else value
    
    def get_images(self, tag: DataTag = DataTag.TRAIN) -> List[ImageData]:
        if tag not in self.datasets:
            return []
        
        images_data = []
        data = self.datasets[tag]
        for img in data['images']:
            img['image_name'] = self._extract(img['file_name'], self.image_name_extractor)
            d = ImageData(
                id = img['id'],
                name=img['image_name'],
                path=f"{tag.value}/images/{img['file_name']}",
                width=img['width'],
                height=img['height'],
                normalized=False,
                tag=tag
            )
            images_data.append(d)
        return images_data
    
    def get_bboxes(self, img_name: str) -> List[BoundingBox]:
        exists, img_id = self._check_if_exists(img_name, self.data['images'], 'image_name')
        if not exists:
            return []
        
        anns = []
        img = self.data['images'][img_id]
        for ann in filter(lambda x: x['image_id'] == img['id'], self.data['annotations']):
            _, cat_id = self._check_if_exists(ann["category_id"], self.data['categories'], 'id')
            cat = self.data['categories'][cat_id]
            cat['label'] = self._extract(cat['name'], self.label_name_extractor)
            b = BoundingBox(
                label=cat['label'],
                _label_id=cat['id'],
                _img_id=img['id'],
                center_x= (ann['bbox'][0] + ann['bbox'][2] * 0.5) / img['width'],
                center_y= (ann['bbox'][1] + ann['bbox'][3] * 0.5) / img['height'],
                width=ann['bbox'][2] / img['width'],
                height=ann['bbox'][3] / img['height'],
                normalized=True
            )
            b.area=ann['area'] if ann['area'] / (img['width'] * img['height'])  else (b.width * b.height)
            segments = []
            if ann['segmentation']:
                for seg in ann['segmentation']:
                    segments.append(np.array([ [seg[i] / img['width'], seg[i+1] / img['height']] for i in range(0, len(seg), 2) ]))
            b.segmentation = segments
            anns.append(b)
        return anns
    
    def get_labels(self, img_name: str) -> List[LabelData]:
        bboxes = self.get_bboxes(img_name)
        filtered_labels  = []
        for bbox in bboxes:
            _, cat_id = self._check_if_exists(bbox.label, self.data['categories'], 'label')
            cat = self.data['categories'][cat_id]
            filtered_labels.append(LabelData(id=cat['id'], name=cat['name']))
        return filtered_labels
    
    def get_bbox(self, img_name: str) -> List[Dict]:
        exists, img_id = self._check_if_exists(img_name, self.data['images'], 'file_name')
        if not exists:
            return (None, None)
        
        anns = []
        for ann in self.data['annotations']:
            if ann['image_id'] == img_id:
                label_name = self._get_label(ann["id"])
                anns.append((label_name, np.array(ann['bbox'])))
        return anns

    def get_labels(self) -> List[str]:
        super_cats = set([c['supercategory'] for c in self.data['categories']])
        cats = []
        for c in self.data['categories']:
            if c['name'] not in super_cats:
                cats.append(c['name'])
        return cats
    
    def to_file(self, file_path: str) -> None:
        f = open(file_path, 'w')
        f.write(json.dumps(self.data, indent=2))
        f.close()

class VOCDataset(Dataset):
    def __init__(self, data: dict) -> None:
        super(VOCDataset, self).__init__()
        self.data = data
        
        labels = set()
        for img_name in self.data.keys():
            for obj in self.data[img_name]['annotation']['object']:
                labels.add(obj['name'])
        self.label_names = sorted([l for l in labels])
    
    def get_bbox(self, img_name: str) -> List[Dict]:
        anns = []
        for ann in self.data[img_name]['annotation']['object']:
            bbox = [
                ann['bndbox']['xmin'], 
                ann['bndbox']['xmax'], 
                ann['bndbox']['ymin'],
                ann['bndbox']['ymax']
            ]
            anns.append((ann['name'], bbox))
        return anns
    
    def get_labels(self) -> List[str]:
        self.label_names
    
    # TODO: mode to a different class dict to xml
    def _create_element(self, name, value) -> Element:
        ele = Element(name)
        ele.text = '' if value is None else str(value)
        return ele
    
    def _build_xml(self, root_elem, obj, name=None):
        if isinstance(obj, Dict):
            for k in obj.keys():
                if not isinstance(obj[k], (Tuple, List, np.ndarray)):
                    sub_elem = SubElement(root_elem, k)
                else:
                    sub_elem = root_elem
                self._build_xml(sub_elem, obj[k], k)
        elif isinstance(obj, (Tuple, List, np.ndarray)):
            for v in obj:
                sub_elem = SubElement(root_elem, name)
                self._build_xml(sub_elem, v, name)
        elif obj is None:
            root_elem.text = ''
        elif isinstance(obj, (str)):
            root_elem.text = obj
        else:
            root_elem.text = str(obj)
        return root_elem

    def _to_xml(self, img_name) -> str:
        root_elem = Element('annotation')
        ann_data = self.data[img_name]['annotation']
        
        root_elem = self._build_xml(root_elem, ann_data)
        indent(root_elem, space="  ", level=0)
        
        # tree = ElementTree(root_elem)
        # tree.write(file_name, encoding="utf-8")
        return str(tostring(root_elem), 'UTF-8')
    
    def to_file(self, dir_path: str) -> None:
        for img_name in self.data.keys():
            ann_path = str(Path(dir_path) / img_name.replace(
                Path(img_name).suffix, '.xml'))
            f = open(ann_path, 'w')
            f.write(self._to_xml(img_name))
            f.close()
    
    
class COCODatasetBuilder(DatasetBuilder):
    def __init__(self, version=0, category_names: List[str]=[]) -> None:
        super(COCODatasetBuilder, self).__init__()
        
        self.now_date = "2023-10-28T18:23:51+00:00"
        self.info = {
            "year": "2023",
            "version": str(version),
        }
        
        self.licenses = [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0"
            }
        ]
        
        self.categories = []
        self.images = []
        self.annotations = []
        
        for name in category_names:
            self.add_category(name)
    
    def __check_if_exists(self, value: str, lst: List[str], attr_name: str = 'name') -> Tuple[bool, int]:
        for i, r in enumerate(lst):
            if r[attr_name] == value:
                return (True, i)
            
        return (False, -1)
    
    def add_category(self, category: str) -> int:
        exists, i = self.__check_if_exists(category, self.categories)
        if exists:
            return i
        
        i = len(self.categories)
        self.categories.append(dict({
            'id': i,
            'name': category,
            "supercategory": None
        }))
        return i
    
    def add_image(self, img_path: str) -> int:
        img_name = Path(img_path).name
        exists, i  = self.__check_if_exists(img_name, self.images, 'file_name')
        if exists:
            return i
        
        img = Image.open(img_path)
        i = len(self.images)
        self.images.append(dict({
            'id': i,
            'license': 1,
            'file_name': img_name,
            "height": img.height,
            'width': img.width,
            "date_captured": self.now_date
        }))
        return i
    
    def add_annotation(self, img_id: int, category_id: int, bbox: np.ndarray = None) -> int:
        if len(self.images) <= img_id or len(self.categories) <= category_id:
            raise ValueError("either img_id or category_id is not found")
        
        i = len(self.annotations)
        self.annotations.append(dict({
            'id': i,
            'image_id': img_id,
            'category_id': category_id,
            "bbox": [v for v in bbox],
            "iscrowd": 0
        }))
        return i
    
    def build(self) -> COCODataset:
        return COCODataset(dict({
            'info': dict(self.info),
            'licenses': self.licenses,
            'categories': self.categories,
            'images': self.images,
            'annotations': self.annotations
        }))
    
    def build_from_file(
        file_path: str,
        tag: DataTag = DataTag.TRAIN,
        image_name_extractor: Callable[[str], str] = None,
        label_name_extractor: Callable[[str], str] = None
    ) -> COCODataset:
        return COCODataset(
            json.loads(Path(file_path).read_text()),
            tag = tag,
            image_name_extractor = image_name_extractor,
            label_name_extractor = label_name_extractor
        )

class VOCDatasetBuilder(DatasetBuilder):
    def __init__(self) -> None:
        super(VOCDatasetBuilder, self).__init__()
        self.data = {}
    
    def add_image(self, img_path: str) -> str:
        img_name = Path(img_path).name
        if img_name in self.data.keys():
            return img_name
        
        self.data[img_name] = {'annotation': {}}
        img = Image.open(img_path)
        self.data[img_name]['annotation'] = dict({
            'folder': None,
            'filename': img_name,
            'path': img_name,
            'segmented': 0,
            'size': dict({
                "height": img.height,
                'width': img.width,
                'depth': 3 # TODO: read from cv2/PIL
            }),
        })
        return img_name
    
    def add_annotation(self, img_id: str, category_id: str, bbox: np.ndarray = None) -> str:
        if img_id not in self.data.keys():
            raise ValueError("img_id is not found")
        
        img_map = self.data[img_id]['annotation']
        
        if 'object' not in img_map.keys():
            img_map['object'] = []
        
        i = len(img_map['object'])
        img_map['object'].append(dict({
            'name': category_id,
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': 0,
            'occluded': 0,
            'bndbox': dict({
                'xmin': bbox[0],
                'xmax': bbox[1],
                'ymin': bbox[2],
                'ymax': bbox[3]
            })
        }))
        self.data[img_id]['annotation'] = dict(img_map)
        return '{}/annotation/object[{}]'.format(img_id, i)
    
    def build(self) -> Dataset:
        return VOCDataset(self.data)
    
    def _xml_to_dict(t):
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(VOCDatasetBuilder._xml_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v
                        for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update(('@' + k, v)
                            for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d
    

    def build_from_file(dir_path: str) -> Dataset:
        data = {}
        for f in FileIterator(dir_path, ['.xml']):
            xml_tree = parse(f)
            d = VOCDatasetBuilder._xml_to_dict(xml_tree.getroot())
            data[d['annotation']['filename']] = d
        return VOCDataset(data)