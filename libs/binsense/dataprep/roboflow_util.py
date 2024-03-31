from ..utils import backup_file
from ..dataset_util import Yolov8Deserializer, DataTag, Dataset

from typing import List, Dict, Any
from re import Pattern
from roboflow import Roboflow
from tqdm import tqdm

import pandas as pd
import requests, re, logging, os

logger = logging.getLogger("__name__")
SUPPORTED_DATASET_FORMATS = ['yolov8']

class RoboflowCrawler:
    def __init__(
        self, 
        project_id: str, 
        workspace_id: str, 
        cookie_str: str,
        target_path: str = './robo_meta.csv',
        annotation_group: str = "bins",
        check_tags: List[Dict[str, str]] = None) -> None:
        
        self.cookie_str = cookie_str
        self.workspace_id = workspace_id
        self.project_id = project_id
        self.roboql_url = 'https://app.roboflow.com/query/roboql/dataset'
        self.ann_group = annotation_group
        self.target_fpath = target_path
        
        if check_tags is None:
            self.check_tags = [
                ('jithu', re.compile('jithu[-0-9]{0,}')),
                ('mythili', re.compile('mythili[-0-9]{0,}')),
                ('nitesh', re.compile('nitesh[-0-9]{0,}')),
                ('raghu', re.compile('nitesh[-0-9]{0,}')),
                ('adjusted', re.compile('adjusted')),
                ('assumed', re.compile('(assume[d]{0,1}|assumption)')),
                ('blurry', re.compile('(blurry|blurred|blur)')),
                ('done', re.compile('(done|dome|donne|completed|complete|finished|finish)')),
                ('hard', re.compile('(hard|hardy|very hard|hardest)'))
            ]
        else:
            self.check_tags = []
    
    def _has_this_tag(self, pattern: Pattern, user_tags: List[str]):
        for t in user_tags:
            if pattern.fullmatch(t):
                return 1
        return 0

    def _fetch_results(self, start_index=0, page_size=1):
        resp = requests.post(
            self.roboql_url,
            headers={'Cookie': self.cookie_str},
            json={\
                  "annotationGroup": self.ann_group, 
                  "pageSize": page_size, 
                  "projectId": self.project_id, 
                  "query": " sort:filename", 
                  "startingIndex": start_index, 
                  "workspaceId": self.workspace_id
            }
        )
        return resp.json()
    
    def _flush_records(self, records: List[List[Any]], file_path: str, mode: str = 'a'):
        with open(file_path, mode) as f:
            for record in records:
                f.write(','.join([str(t) for t in record]))
                f.write('\n')
    
    def crawl(self) -> str:
        data = self._fetch_results()
        logger.debug('successful=', data['success'])
        logger.debug('user_tags=', data['aggregations']['user_tags'])
        logger.debug('timed_out=', data["info"]["timed_out"], 'took=', data["info"]["took"], 'total=', data["info"]["total"])
        total = data["info"]["total"]
        
        columns = [ "rf_image_id", "image_name", "tag" ]
        for chktag in self.check_tags:
            columns.append(f'is_{chktag[0]}')
        columns.extend(["bbox_label", "bbox_count"])
        logger.info(f"number of columns = {len(columns)}")
        
        file_path = self.target_fpath
        if os.path.exists(file_path):
            bkp_file = backup_file(file_path)
            logger.info(f"backing up {file_path} to {bkp_file}")
        self._flush_records([columns], file_path, 'w')
        
        offset = 0
        page_size = 200
        progress_step = total // 10
        progress_bar = tqdm(
            total=total,
            file=open(os.devnull, 'w'),
            desc="crawling roboflow.com")
        while offset < total:
            data = self._fetch_results(offset, page_size)
            if not data['success']:
                logger.info(f"{offset} is not successful!")
            
            results = data['similarImages']
            for result in results:
                anns = result["annotations"]
                record = [
                    result["id"],
                    result["name"],
                    result["split"]
                ]
                for chktag in self.check_tags:
                    v = self._has_this_tag(chktag[1], result['user_tags'])
                    record.append(v)
                record.extend([
                    '',
                    0
                ])
                cls_records = []
                for cls, count in anns[self.ann_group]["classCounts"].items():
                    cls_record = record.copy()
                    cls_record[-2] = cls.split('|')[0]
                    cls_record[-1] = count
                    cls_records.append(cls_record)
                self._flush_records(cls_records, file_path, 'a')
                offset += 1
            progress_bar.update(len(results))
            if offset >= progress_step:
                progress_step += offset
                logger.info(str(progress_bar))
        logger.info(str(progress_bar))
        return file_path

class RoboflowDownloader:
    def __init__(
        self, 
        workspace: str = 'nitesh-c-eszzc', 
        project: str = 'binsense_bbox_mini',
        version: int = 1,
        api_key: str = None) -> None:
        self.workspace = workspace
        self.project = project
        self.version = version
        self.api_key = os.environ.get('ROBOFLOW_MY_API_KEY') if api_key is None\
            else api_key
    
    def download(self, target_dir='.', format: str = "yolov8") -> str:
        dir_path = target_dir
        if os.path.exists(dir_path):
            bkp_dir = backup_file(dir_path)
            logger.info(f"backing up {dir_path} to {bkp_dir}")
        
        if not format in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f'{format} is not in supported formats - {SUPPORTED_DATASET_FORMATS}')
        
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace)\
            .project(self.project)
        version = project.version(self.version)
        dataset = version.download(format, location=dir_path, overwrite=True)
        return dataset.location

class RoboflowDatasetReader:
    def __init__(self, dataset_dirpath: str, format: str ='yolov8') -> None:
        self.dataset_dirpath = dataset_dirpath
        self.label_extractor_pattern = re.compile('^[^-]+')
    
    def read(self) -> Dataset:
        def label_extractor(name: str) -> str:
            return self.label_extractor_pattern.match(name).group()
        
        def imgname_extractor(name: str) -> str:
            imgname, extn = os.path.splitext(name)
            clean_img_name = imgname.split('_jpg')[0].strip()
            return f'{clean_img_name}{extn}'
            
        ds_reader = Yolov8Deserializer(
            self.dataset_dirpath, 
            image_name_extractor=imgname_extractor,
            label_name_extractor=label_extractor,
            img_extns=['.jpg']
        )
        return ds_reader.read()
    
class RoboflowDatasetValidator:
    def __init__(
        self, 
        crawler_filepath: str,
        dataset_dirpath: str,
        dataset_format: str = "yolov8") -> None:
        """
        Args:
            crawler_filepath(`str`): file path of RoboflowCrawler output
            dataset_dirpath(`str`): dir path of RoboflowDownloader output
            dataset_format(`str`): format of the downloaded dataset by RoboflowDownloader
        """
        
        if not self._validate_exists(crawler_filepath, True):
            raise ValueError(f'{crawler_filepath} doesn\'t exist or is not a file.')
        
        if not self._validate_exists(dataset_dirpath, False):
            raise ValueError(f'{dataset_dirpath} doesn\'t exist or is not a dir.')
        
        if not dataset_format in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f'{dataset_format} is not in supported formats - {SUPPORTED_DATASET_FORMATS}')
        
        self.crawler_filepath = crawler_filepath
        self.dataset_dirpath = dataset_dirpath
        self.dataset_format = dataset_format
        self.dataset_reader = RoboflowDatasetReader(dataset_dirpath, dataset_format)
    
    
    @classmethod
    def _validate_exists(self, file_path: str, is_file: bool) -> bool:
        return os.path.exists(file_path) \
            and os.path.isfile(file_path) if is_file \
                else os.path.isdir(file_path)
    
    def _validate_full_dataset(self, full_dataset: pd.DataFrame):
        if not isinstance(full_dataset, pd.DataFrame):
            raise ValueError(f'expecting full_dataset to be a pandas DataFrame')
        cols = full_dataset.columns
        if (not 'image_name' in cols) \
            or (not 'bbox_label' in cols) \
            or (not 'bbox_count' in cols):
            raise ValueError(f'mandatory columns [image_name, bbox_label, bbox_count] missing in full_dataset')
    
    def _validate_crawl_dataset(self, crawler_filepath: str):
        with open(crawler_filepath, 'r') as f:
            header = f.readline()
        
        cols = [col.strip() for col in header.strip().split(',')]
        if (not 'image_name' in cols) \
            or (not 'bbox_label' in cols) \
            or (not 'bbox_count' in cols):
            raise ValueError(f'mandatory columns [image_name, bbox_label, bbox_count] missing in crawl dataset')
    
    def _load_downloaded_dataset(self):
        ds = self.dataset_reader.read()
        df = pd.DataFrame(columns=['image_name', 'bbox_label', 'bbox_count', 'tag'])
        
        def _read_ds(ds_tag):
            for img_data in ds.get_images(ds_tag):
                image_name = img_data.name
                bbox_label_counts = {}
                for bbox_data in ds.get_bboxes(image_name):
                    if not bbox_data.label in bbox_label_counts:
                        bbox_label_counts[bbox_data.label] = 1
                    else:
                        bbox_label_counts[bbox_data.label] += 1
                for bbox_label, bbox_count in bbox_label_counts.items():
                    df.loc[len(df)] = [image_name, bbox_label, bbox_count, str(img_data.tag)]
        
        _read_ds(DataTag.TRAIN)
        _read_ds(DataTag.VALID)
        return df
    
    def validate(self, full_dataset: pd.DataFrame) -> bool:
        """
        Args:
            full_dataset(`pandas.DataFrame`):
                expects the full dataset from where uploaded dataset was prepared and uploaded to roboflow.
                this is the dataset against which the downloaded dataset is validated.
                It should have below columns
                    image_name: image name
                    bbox_label: bbox label
                    bbox_count: bbox count
                    tag: 'train' or 'valid' or None
                    uploaded: 1 if uploaded to roboflow or 0
        Return:
            valid (`bool`): valid or not
            results(`List[Tuple(str, bool)]`): list of check & true/false tuples
        """
        self._validate_full_dataset(full_dataset)
        self._validate_crawl_dataset(self.crawler_filepath)
        
        rfds_df = self._load_downloaded_dataset(self.dataset_dirpath)
        rf_df = pd.read_csv(self.crawler_filepath, 
                    dtype={'image_name': str, 'bbox_label': str})
        full_df = full_dataset
        
        results = [
            ('rfmeta: not_empty', self.check_not_empty(rf_df)),
            ('rfmeta: image_count', self.check_image_count(rf_df, full_df)),
            ('rfmeta: label_count', self.check_label_phantom(rf_df, full_df)),
            ('rfmeta: bbox_count', self.check_bbox_count(rf_df, full_df)),
            ('rfds: not_empty', self.check_not_empty(rf_df)),
            ('rfds: image_count', self.check_image_count(rfds_df, full_df)),
            ('rfds: label_count', self.check_label_phantom(rfds_df, full_df)),
            ('rfds: bbox_count', self.check_bbox_count(rfds_df, full_df))
        ]
        valid = all([t[1] for t in results])
        return valid, results
    
    def check_not_empty(self, rf_df) -> bool:
        return rf_df.shape[0] > 0
    
    def check_image_count(self, rf_df, full_df) -> bool:
        return rf_df['image_name'].nunique() <= full_df['image_name'].nunique()
    
    def check_label_count(self, rf_df, full_df) -> bool:
        rf_label_count = rf_df['bbox_label'].nunique()
        full_label_count = full_df['bbox_label'].nunique()
        valid = ( rf_label_count <= full_label_count )
        if not valid:
            logger.info(f"label count not valid rf's {rf_label_count} & full's {full_label_count}")
        return valid
    
    
    def check_label_phantom(self, rf_df, full_df) -> bool:
        rf_labels = set(rf_df['bbox_label'].unique())
        full_labels = set(full_df['bbox_label'].unique())
        phantom_labels = rf_labels - full_labels
        valid = len(phantom_labels) == 0
        if not valid:
            logger.info(f"phantom labels in rf discovered - {phantom_labels}")
        return valid
    
    def check_bbox_count(self, rf_df, full_df):
        missing_images = set(full_df['image_name'].unique()) - set(rf_df['image_name'].unique())
        df = full_df[~full_df.image_name.isin(missing_images)]
        
        df1 = df[["image_name", "bbox_label", "bbox_count"]].\
            rename(columns={ 'bbox_count': 'full_bbox_count'})
        df2 = rf_df[["image_name", "bbox_label", "bbox_count"]]\
            .rename(columns={'bbox_count': 'rf_bbox_count'})
        result = pd.merge(
            left=df1, right=df2, 
            on=["image_name", "bbox_label"], 
            how='outer')
        
        result_incorrect_labels = result[result['full_bbox_count'].isna()]
        result_higher_bbox_qts = result[result['full_bbox_count'] < result['rf_bbox_count']]
        
        logger.info("check for validity of annotated data")
        logger.info(f"#annotations with incorrect items: {result_incorrect_labels.shape[0]}")
        if result_incorrect_labels.shape[0] > 0:
            logger.info(f'\n{result_incorrect_labels}')
            
        logger.info(f"#annotations with higher item quantities: {result_higher_bbox_qts.shape[0]}")
        if result_higher_bbox_qts.shape[0] > 0:
            logger.info(f'\n{result_higher_bbox_qts}')
        
        return \
            result_incorrect_labels.shape[0] == 0 \
                and result_higher_bbox_qts.shape[0] == 0