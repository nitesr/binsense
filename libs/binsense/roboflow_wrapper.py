import os
import yaml
import numpy as np
import json

from typing import List
from pathlib import Path

from roboflow import Roboflow
from roboflow.core.project import Project

class RoboDataset:
    def __init__(self) -> None:
        pass
    
    def get_all_images_dir(self) -> List:
        pass
    
    def get_all_labels_dir(self) -> List:
        return []
    
    def get_images_dir(self, tag: str) -> str:
        return None
    
    def get_labels_dir(self, tag: str) -> str:
        return None
    
    def get_tuples(self) -> List:
        pass
    
    def get_labels(self) -> List:
        pass

    def map_label_path(self, tag: str, img_path: str) -> str:
        return None
    
    def supports_single_annotation_file(self) -> bool:
        return False
    
    def create_tag(self, tag) -> None:
        pass
    

class RoboYoloDataset(RoboDataset):
    def __init__(self, root_dir='.', format='yolov8') -> None:
        super(RoboYoloDataset, self).__init__()
        self.root_dir = root_dir
        self.images_dir = {}
        self.labels_dir = {}
        self.label_names = []
        self.tags = []
        
        if os.path.exists(self.root_dir+'/data.yaml'):
            conf = yaml.safe_load(Path(self.root_dir+'/data.yaml').read_text())
            for tag in ['test', 'train', 'val']:
                if conf[tag]:
                    p = conf[tag]
                    p = p.replace('..', self.root_dir)
                    if os.path.exists(p):
                        self.tags.append(tag)
                        self.images_dir[tag] = p
                        self.labels_dir[tag] = str(Path(p).parent / 'labels')
            self.label_names = conf['names']
    
    def create_tag(self, tag) -> None:
        os.makedirs(self.root_dir+'/'+tag+'/images', exist_ok=True)
        os.makedirs(self.root_dir+'/'+tag+'/labels', exist_ok=True)
        self.images_dir[tag] = self.root_dir+'/'+tag+'/images'
        self.labels_dir[tag] = self.root_dir+'/'+tag+'/labels'
        if tag not in self.tags:
            self.tags.append(tag)
        
    def get_all_images_dir(self) -> List:
        return [self.images_dir[tag] for tag in self.tags]
    
    def get_all_labels_dir(self) -> List:
        return [self.labels_dir[tag] for tag in self.tags]
    
    def get_labels(self) -> List:
        return self.label_names
    
    def get_tuples(self) -> List:
        return [(tag, self.images_dir[tag], self.labels_dir[tag]) for tag in self.tags]

    def map_label_path(self, tag: str, img_path: str) -> str:
        if img_path is None:
            return None
        
        img_name = Path(img_path).name
        ann_dir = self.labels_dir[tag]
        
        return str(Path(ann_dir) / img_name.replace(
                        Path(img_name).suffix, '.txt'))
    
    def supports_single_annotation_file(self) -> bool:
        return False

class RoboVOCDataset(RoboDataset):
    def __init__(self, root_dir='.', format='voc') -> None:
        super(RoboVOCDataset, self).__init__()
        self.root_dir = root_dir
        self.images_dir = {}
        self.tags = []
        self.label_names = []
        
        os.makedirs(self.root_dir, exist_ok=True)
        for dir_name in os.listdir(root_dir):
            if os.path.isdir(root_dir + '/' + dir_name):
                self.tags.append(dir_name)
                self.images_dir[dir_name] = root_dir + '/' + dir_name
    
    def create_tag(self, tag) -> None:
        os.makedirs(self.root_dir+'/'+tag, exist_ok=True)
        self.images_dir[tag] = self.root_dir+'/'+tag
        if tag not in self.tags:
            self.tags.append(tag)
        
    def get_all_images_dir(self) -> List:
        return [self.images_dir[tag] for tag in self.tags]
    
    def get_all_labels_dir(self) -> List:
        return [self.images_dir[tag] for tag in self.tags]
    
    def get_labels(self) -> List:
        return self.label_names
    
    def get_tuples(self) -> List:
        return [(tag, self.images_dir[tag], self.images_dir[tag]) for tag in self.tags]

    def map_label_path(self, tag: str, img_path: str) -> str:
        if img_path is None:
            return None
        
        img_name = Path(img_path).name
        ann_dir = self.images_dir[tag]
        
        return str(Path(ann_dir) / img_name.replace(
                        Path(img_name).suffix, '.xml'))
    
    def supports_single_annotation_file(self) -> bool:
        return False

class RoboCOCODataset(RoboDataset):
    def __init__(self, root_dir='.', ignore_supercategory=True, format='coco') -> None:
        super(RoboCOCODataset, self).__init__()
        self.root_dir = root_dir
        self.ignore_supercategory = ignore_supercategory
        self.images_dir = {}
        self.label_names = []
        
        self.tags = []
        self.annotation_files = {}
        
        os.makedirs(self.root_dir, exist_ok=True)
        for dir_name in os.listdir(root_dir):
            if os.path.isdir(root_dir + '/' + dir_name):
                self.tags.append(dir_name)
                self.images_dir[dir_name] = root_dir + '/' + dir_name
                self.annotation_files[dir_name] = root_dir + '/' + dir_name + '/_annotations.coco.json'
                
        if len(self.tags) > 0:
            ann_path = str(Path(root_dir) /  self.tags[0] / '_annotations.coco.json')
            if os.path.isfile(ann_path):
                self.label_names = self._generate_labels(ann_path)
    
    def create_tag(self, tag) -> None:
        if tag not in self.tags:
            self.tags.append(tag)
        
        os.makedirs(self.root_dir+'/'+tag, exist_ok=True)
        self.images_dir[tag] = self.root_dir+'/'+tag
        
    def _generate_labels(self, ann_path):
        categories = json.loads(Path(ann_path).read_text())['categories']
        super_categories = set([c['supercategory'] for c in categories])
        labels = []
        for c in categories:
            if c['name'] not in super_categories:
                labels.append(c['name'])
        return labels
    
    def get_labels(self) -> List:
        return self.label_names
    
    def get_all_images_dir(self) -> List:
        return [self.images_dir[tag] for tag in self.tags]
    
    def get_images_dir(self, tag: str) -> str:
        return self.images_dir[tag]
    
    def get_labels_dir(self, tag: str) -> str:
        return self.annotation_files[tag]
    
    def get_tuples(self) -> List:
        return [(tag, self.images_dir[tag], self.annotation_files[tag]) for tag in self.tags]

    def get_all_labels_dir(self) -> List:
        return [self.annotation_files[tag] for tag in self.tags]
    
    def map_label_path(self, tag: str, img_path: str) -> str:
        return self.annotation_files[tag]
    
    def supports_single_annotation_file(self) -> bool:
        return True

class RoboProj:
    def __init__(
        self,
        rf,
        rf_proj: Project,
        proj_ver=None) -> None:
        
        self.rf = rf
        self.rf_proj = rf_proj
        
        self.id = rf_proj.id
        self.ver = proj_ver
        
        temp = self.id.rsplit("/")
        self.__workspace = temp[0]
        self.__proj_name = temp[1]
        
        self.downloaders = {
            'yolov8': self._download_yolov8,
            'coco': self._download_coco,
            'voc': self._download_voc
        }
    
    def _download_voc(self, rf_proj, dataset_path) -> RoboVOCDataset:
        rf_proj.download(
            'voc', dataset_path, overwrite=True)
        
        return RoboVOCDataset(root_dir=dataset_path)
    
    def _download_yolov8(self, rf_proj, dataset_path) -> RoboYoloDataset:
        rf_proj.download(
            'yolov8', dataset_path, overwrite=True)
        
        return RoboYoloDataset(root_dir=dataset_path)
    
    def _download_coco(self, rf_proj, dataset_path) -> RoboCOCODataset:
        rf_proj.download(
            'coco', dataset_path, overwrite=True)
        
        return RoboCOCODataset(root_dir=dataset_path)
  
    def download(self, dataset_path='.', dataset_format='yolov8') -> RoboDataset:
        rf_proj = self.rf.project(self.id)
        ver = self.ver
        if ver is None:
            ver = self.get_latest_version()
        
        if not self._check_ver_exists(ver):
            raise ValueError("Version is mandatory to download. You can generate a version on app.roboflow.com")
        
        rf_proj = rf_proj.version(ver)
        os.makedirs(dataset_path, exist_ok=True)
        return self.downloaders[dataset_format](rf_proj, dataset_path)

        
    def upload(self, dataset: RoboDataset):
        project = self.rf.project(self.id)
    
        for (tag, img_dir, _) in dataset.get_tuples():
            # if dataset.supports_single_annotation_file():
            #     project.upload(image_path=img_dir, split=tag)
            #     continue
            
            for img_name in os.listdir(img_dir):
                img_path = img_dir + '/' + img_name
                if Path(img_path).suffix in ['.jpg', '.jpeg']:
                    ann_path = dataset.map_label_path(tag, img_path)
                    project.upload(
                        image_path=img_path,
                        annotation_path=ann_path,
                        split=tag
                    )
    
    def _check_ver_exists(self, ver) -> bool:
        versions = self._get_all_versions()
        if len(versions) == 0:
            return False
        
        return int(ver) in versions
    
    def _get_all_versions(self) -> list:
        versions = self.rf.project(self.id).get_version_information()
        if len(versions) == 0:
            return []
        
        return [ int(v['id'].rsplit('/')[2]) for v in versions ]
    
    def get_latest_version(self) -> str:
        versions = self._get_all_versions()
        if len(versions) == 0:
            return None
        
        return str(np.max(versions))


class RoboProjCreator:
    def __init__(
        self,
        rf,
        proj_name,
        proj_type='object-detection',
        proj_ann='facial-expressions',
        proj_license='CC BY 4.0',
        workspace='nitesh-c-eszzc',
        delete_if_exists=False) -> None:
        
        self.rf = rf
        self.proj_name = proj_name
        self.proj_type = proj_type
        self.proj_ann = proj_ann
        self.proj_license = proj_license
        self.workspace = workspace
        self.delete_if_exists = delete_if_exists
        
        self.rf_ws = self.rf.workspace('nitesh-c-eszzc')
        
    def _get_current_projects(self) -> list:
        proj_list = self.rf_ws.project_list
        return [ x['name'] for x in proj_list if x['name'] == self.proj_name ]
    
    def _get_project(self) -> list:
        proj_list = self.rf_ws.project_list
        
        proj = None
        for p in proj_list:
            if p['name'] == self.proj_name:
                proj = p
                break
        return proj
    
    def _convert(self, rf_proj) -> RoboProj:
        return RoboProj(self.rf, rf_proj)
    
    def create(self) -> RoboProj:
        proj = self._get_project()
        if proj is None:
            rf_proj = self.rf_ws.create_project(
                project_name=self.proj_name,
                project_type=self.proj_type,
                project_license=self.proj_license,
                annotation=self.proj_ann
            )
            return self._convert(rf_proj)
        else:
            return self._convert(self.rf_ws.project(proj['id']))

# class RoboCopier:
#     def __init__(
#         self, 
#         src_proj_id, 
#         src_proj_ver, 
#         dst_proj_name,
#         workspace='nitesh-c-eszzc'
#     ) -> None:

