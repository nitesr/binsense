from .utils import FileIterator, ImageFileIterator

from pathlib import Path

from xml.etree.ElementTree import indent, parse, ElementTree, Element, SubElement, tostring
from collections import defaultdict
from PIL import Image

import numpy as np
import json

from typing import List, Dict, Tuple

class DatasetBuilder:
    """
    Marker class to declare the methods across all the DatasetBuilder(s).
    """
    def __init__(self) -> None:
        pass
    
    def add_image(self, img_path: str):
        """
        should add the image to the dataset

        Args:
            img_path (`str`):
                complete file path to the image.
        """
        pass
    
    def add_category(self, category: str):
        """
        should add the category/label to the dataset

        Args:
            category (`str`):
                category name.
        """
        pass
    
    def add_annotation(self, img_id, category_id, bbox: np.ndarray=None):
        pass
    
    def build(self) -> dict:
        return dict()

class Dataset:
    """
    Marker class to declare the methods across all Dataset(s).
    """
    def __init__(self) -> None:
        pass
    
    def get_bbox(self, img_name: str) -> List[Dict]:
        pass
    
    def get_labels(self) -> List[str]:
        pass
    
    def to_file(self, file_path: str) -> None:
        pass

class COCODataset(Dataset):
    def __init__(self, data: dict) -> None:
        super(COCODataset, self).__init__()
        self.data = data
    
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

class YoloDataset(Dataset):
    def __init__(self, data: Dict, label_names: List[str]=[]) -> None:
        super(YoloDataset, self).__init__()
        self.data = data
        self.label_names = label_names
    
    def get_bbox(self, img_name: str) -> List[Dict]:
        anns = []
        for ann in self.data[img_name]:
            lbl_idx = ann[0]
            bbox = ann[1:]
            anns.append((self.label_names[lbl_idx], bbox))
        return anns
    
    def get_labels(self) -> List[str]:
        self.label_names
    
    def _stringify_bbox(self, img_name):
        lines = []
        for ann in self.data[img_name]:
            f_ann = [str(int(ann[0]))]+[str(a) for a in ann[1:]]
            lines.append(' '.join(f_ann)+'/n')
        return lines
        
    def to_file(self, dir_path: str) -> None:
        for img_name in self.data.keys():
            ann_path = str(Path(dir_path) / img_name.replace(
                Path(img_name).suffix, '.txt'))
            f = open(ann_path, 'w')
            f.writelines(self._stringify_bbox(img_name))
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
    
    def __check_if_exists(self, value: str, lst: list, attr_name: str = 'name') -> (bool, int):
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
    
    def build(self) -> Dataset:
        return COCODataset(dict({
            'info': dict(self.info),
            'licenses': self.licenses,
            'categories': self.categories,
            'images': self.images,
            'annotations': self.annotations
        }))
    
    def build_from_file(file_path: str) -> Dataset:
        return COCODataset(json.loads(Path(file_path).read_text()))

class VOCDatasetBuilder(DatasetBuilder):
    def __init__(self) -> None:
        super(VOCDatasetBuilder, self).__init__()
        self.data = {}
    
    def add_image(self, img_path: str):
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
                'depth': 3 # TODO: read from cv2
            }),
        })
        return img_name
    
    def add_annotation(self, img_id: str, category_id: str, bbox: np.ndarray = None):
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

class YoloDatasetBuilder(DatasetBuilder):
    def __init__(self) -> None:
        super(YoloDatasetBuilder, self).__init__()
        self.data = {}
    def add_image(self, img_path: str):
        img_name = Path(img_path).name
        if img_name in self.data.keys():
            return img_name
        
        self.data[img_name] = []
        return img_name
    
    def add_annotation(self, img_id: str, category_id: int, bbox: np.ndarray = None):
        if img_id not in self.data.keys():
            raise ValueError("img_id is not found")
        
        ann = self.data[img_id]
        i = len(ann)
        ann.append([category_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        self.data[img_id] = ann
        return '{}[{}]'.format(img_id, i)
    
    def build(self) -> Dataset:
        return YoloDataset(self.data, label_names=[])
    
    def build_from_lines(lines: List) -> List:
        anns = []
        
        for line in lines:
            tokens = line.split(' ')
            if len(tokens) == 5:
                ann = [
                    int(tokens[0]),
                    float(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                    float(tokens[4])
                ]
                anns.apend(ann)
        return anns

    def build_from_file(
        ann_dir_path: str, 
        img_dir_path: str=None, 
        label_names: List=[]) -> Dataset:
        
        if img_dir_path is None:
            img_dir_path = Path(ann_dir_path).parent + '/images'
            
        data = {}
        for img_path in ImageFileIterator(img_dir_path):
            img_name = Path(img_path).name
            ann_name = img_name.replace(Path(img_name).suffix, '.txt')
            
            annf = open(ann_dir_path+'/'+ann_name, 'r')
            lines = annf.readlines()
            annf.close()
            
            data[img_name] = YoloDatasetBuilder.build_from_lines(lines)
        return YoloDataset(data, label_names)


