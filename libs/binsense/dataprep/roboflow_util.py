from ..utils import backup_file, get_default_on_none
from ..dataset_util import Yolov8Deserializer, \
    ImageData, DataTag, Dataset, YoloDatasetBuilder, DatasetBuilder
from .config import DataPrepConfig

from typing import List, Dict, Any, Tuple
from re import Pattern
from roboflow import Roboflow
from tqdm import tqdm

import pandas as pd
import numpy as np
import requests, re, logging, os, random


logger = logging.getLogger("__name__")
SUPPORTED_DATASET_FORMATS = ['yolov8']

class RoboflowCrawler:
    def __init__(
        self, 
        project_id: str = None, 
        workspace_id: str = None, 
        cookie_str: str = None,
        target_path: str = None,
        annotation_group: str = None,
        check_tags: List[Dict[str, str]] = None,
        cfg: DataPrepConfig = None
    ) -> None:
        
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        
        self.cookie_str = cookie_str
        self.workspace_id = get_default_on_none(workspace_id, self.cfg.robo_workspace_id)
        self.project_id = get_default_on_none(project_id, self.cfg.robo_project_id)
        self.roboql_url = self.cfg.roboql_dataset_url
        self.ann_group = get_default_on_none(annotation_group, self.cfg.robo_ann_group)
        self.target_fpath = get_default_on_none(target_path, self.cfg.rfmeta_file_path)
        self.check_tags = get_default_on_none(check_tags, self.cfg.robo_meta_check_tags)
    
    def _has_this_tag(self, pattern: Pattern, user_tags: List[str]):
        for t in user_tags:
            if pattern.fullmatch(t):
                return 1
        return 0

    def _fetch_results(self, start_index=0, page_size=1):
        resp = requests.post(
            self.roboql_url,
            headers={'Cookie': self.cookie_str},
            json={ \
                "annotationGroup": self.ann_group, 
                "pageSize": page_size, 
                "projectId": self.project_id, 
                "query": " sort:filename", 
                "startingIndex": start_index, 
                "workspaceId": self.workspace_id
            }
        )
        if resp.status_code >= 300:
            raise ValueError(f'invalid status({resp.status_code}) from server. {resp.text}')
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
        logger.info(str(progress_bar))
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
        workspace: str = None, 
        project: str = None,
        version: int = None,
        api_key: str = None,
        cfg: DataPrepConfig = None) -> None:
        
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.workspace = get_default_on_none(workspace, self.cfg.robo_workspace)
        self.project = get_default_on_none(project, self.cfg.robo_project)
        self.version = get_default_on_none(version, self.cfg.robo_dataset_version)
        self.api_key = get_default_on_none(api_key, os.environ.get('ROBOFLOW_MY_API_KEY'))
    
    def download(self, target_dir: str = None, format: str = "yolov8") -> str:
        dir_path = get_default_on_none(target_dir, self.cfg.dataset_download_path)
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
    def __init__(
        self, 
        dataset_dirpath: str = None, 
        format: str ='yolov8',
        cfg: DataPrepConfig = None) -> None:
        
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.dataset_dirpath = get_default_on_none(dataset_dirpath, self.cfg.dataset_download_path)
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

class RoboflowDatasetCopier:
    def __init__(
        self,
        target_dirpath: str,
        crawler_filepath: str,
        dataset_dirpath: str,
        dataset_format: str = "yolov8",
        cfg: DataPrepConfig = None,
        use_crawler: bool = True) -> None:
        """
        Args:
            target_dirpath(`str`): dir path to where the filtered dataset need to be stored
            crawler_filepath(`str`): file path of RoboflowCrawler output
            dataset_dirpath(`str`): dir path of RoboflowDownloader output
            dataset_format(`str`): format of the downloaded dataset by RoboflowDownloader
        """
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.target_dirpath = get_default_on_none(target_dirpath, self.cfg.filtered_dataset_path)
        self.crawler_filepath = get_default_on_none(crawler_filepath, self.cfg.rfmeta_file_path) if use_crawler else None
        self.dataset_dirpath = get_default_on_none(dataset_dirpath, self.cfg.dataset_download_path)
        
        if use_crawler and not self._validate_exists(self.crawler_filepath, True):
            raise ValueError(f'{crawler_filepath} doesn\'t exist or is not a file.')
        
        if not self._validate_exists(self.dataset_dirpath, False):
            raise ValueError(f'{dataset_dirpath} doesn\'t exist or is not a dir.')
        
        if not dataset_format in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f'{dataset_format} is not in supported formats - {SUPPORTED_DATASET_FORMATS}')
        
        self.dataset_format = dataset_format
        self.dataset_reader = RoboflowDatasetReader(
            self.dataset_dirpath, self.dataset_format, self.cfg)
    
    def _load_rfmeta_and_log_stats(self) -> pd.DataFrame:
        def calc_prec(portion, total) -> float:
            return round(portion/total * 100, 1)

        def calc_tag_perc(user_tag, df):
            if user_tag not in df.columns:
                return 0
            cnt_y = df[df[user_tag] == 1].shape[0]
            return cnt_y, calc_prec(cnt_y, df.shape[0])
        
        rf_meta_df = pd.read_csv(self.cfg.rfmeta_file_path, dtype={'image_name': str, 'bbox_label': str})
        user_tags = ['is_adjusted', 'is_assumed', 'is_blurry', 'is_done', 'is_hard']
        for user_tag in user_tags:
            logger.info(f'user_tag={user_tag} percent={calc_tag_perc(user_tag, rf_meta_df)}%')
        
        df = rf_meta_df.query('is_done == 1 & is_hard != 1 & is_blurry != 1')
        logger.info(f'train+val data: {df.shape[0]}({calc_prec(df.shape[0], rf_meta_df.shape[0])}%)')
        
        train_cnt = df.query('tag == "train"').shape[0]
        val_cnt = df.query('tag == "valid"').shape[0]
        logger.info(f'train_ratio={round(train_cnt/df.shape[0], 1)} valid_ratio={round(val_cnt/df.shape[0], 1)}')
        return df
    
    def _validate_rfmeta_dataset(self, rfmeta_filepath: str):
        with open(rfmeta_filepath, 'r') as f:
            header = f.readline()
        
        cols = [col.strip() for col in header.strip().split(',')]
        if (not 'image_name' in cols) \
            or (not 'is_done' in cols) \
            or (not 'is_hard' in cols) \
            or (not 'is_blurry' in cols):
            raise ValueError(f'mandatory columns [image_name, is_done, is_hard, is_blurry] missing in crawl dataset')
    
    def _validate_full_dataset(self, full_dataset: pd.DataFrame):
        if not isinstance(full_dataset, pd.DataFrame):
            raise ValueError(f'expecting full_dataset to be a pandas DataFrame')
        cols = full_dataset.columns
        if (not 'image_name' in cols) \
            or (not 'bbox_label' in cols) \
            or (not 'bbox_count' in cols) \
            or (not 'tag' in cols):
            raise ValueError(f'mandatory columns [image_name, tag, bbox_label, bbox_count] missing in full_dataset')
    
    def _copy_image_bboxes(
        self, 
        src_ds: Dataset, 
        dst_ds: DatasetBuilder, 
        src_img_data: ImageData,
        dst_category_dict: Dict[str, int]) -> int:
        
        dst_img_id = dst_ds.add_image(
            src_img_data.path, src_img_data.tag, 
            src_img_data.name)
        bboxes = src_ds.get_bboxes(src_img_data.name)
        src_cat_names = [bbox.label for bbox in bboxes]
        src_bbox_arrays = [ bbox.to_array() for bbox in bboxes]
        dst_cat_ids = [dst_category_dict[name] for name in src_cat_names]
        dst_ds.add_bboxes(dst_img_id, dst_cat_ids, src_bbox_arrays)
        return dst_img_id
    
    def _filter_testset(
        self, test_dataset: pd.DataFrame,
        rfmeta_dataset: pd.DataFrame) -> pd.DataFrame:
        rf_df = rfmeta_dataset.query('tag != "test"')
        ts_df = test_dataset[~test_dataset.image_name.isin(rf_df["image_name"])]
        ts_df = ts_df[ts_df.bbox_label.isin(rf_df['bbox_label'])]
        ts_df.reset_index(drop=True, inplace=True)
        return ts_df
    
    def _sample_testset(
        self, test_dataset: pd.DataFrame,
        rf_meta_images: List[str]) -> pd.DataFrame:
        req_sample_size = int(len(rf_meta_images) / 0.8 * 0.2)
        test_images = test_dataset["image_name"].unique()
        sample_size = min(req_sample_size, len(test_images))
        if sample_size < req_sample_size:
            logger.warn(f"testset is under sampled req_sample_size={req_sample_size} sampled={sample_size}")
        random.shuffle(test_images)
        sampled_images = test_images[0:sample_size]
        return test_dataset[test_dataset.image_name.isin(sampled_images)]
        
    def _copy_test_dataset(
        self, test_dataset: pd.DataFrame,
        dst_ds: DatasetBuilder,
        dst_image_names: List[str],
        dst_category_dict: Dict[str, int]) -> None:
        
        for i in range(test_dataset.shape[0]):
            test_rec = test_dataset.iloc[i].to_dict()
            if test_rec['image_name'] in dst_image_names:
                # ignore if already added
                continue
            image_name = test_rec['image_name']
            image_path = os.path.join(self.cfg.data_split_images_dir, image_name)
            image_id = dst_ds.add_image(image_path, DataTag.TEST, image_name)
            
            # add bounding boxes with coordinates as zeros
            # for test we don't need coordinates
            category_name = test_rec['bbox_label']
            category_id = dst_category_dict[category_name]
            category_ids = [ category_id ] * test_rec['bbox_count']
            bboxes = [np.zeros((4,))] * test_rec['bbox_count']
            dst_ds.add_bboxes(image_id, category_ids, bboxes)
    
    def filter_and_copy(self, full_dataset: pd.DataFrame) -> str:
        self._validate_full_dataset(full_dataset)
        filtered_ds = YoloDatasetBuilder()
        dst_category_dict = filtered_ds.add_categories(full_dataset['bbox_label'].unique())
        
        # copy all valid images from rf downloaded dataset
        rf_meta_df = self._load_rfmeta_and_log_stats()
        rf_meta_images = set(rf_meta_df['image_name'].unique())
        ds = self.dataset_reader.read()
        for tag in [ DataTag.TRAIN, DataTag.VALID, DataTag.TEST ]:
            for img_data in ds.get_images(tag):
                if img_data.name in rf_meta_images:
                    self._copy_image_bboxes(ds, filtered_ds, img_data, dst_category_dict)
        
        
        # copy the TEST dataset, which are not annotated
        test_dataset = full_dataset.query(f'tag == "{DataTag.TEST.value}"').reset_index(drop=True)
        test_dataset = self._filter_testset(test_dataset, rf_meta_df)
        test_dataset = self._sample_testset(test_dataset, rf_meta_images)
        self._copy_test_dataset(test_dataset, dst_ds=filtered_ds, dst_image_names=rf_meta_images, dst_category_dict=dst_category_dict)
        
        dst_ds = filtered_ds.build()
        def get_ds_count(split_tag: DataTag):
            return len(dst_ds.get_images(split_tag))
        train_cnt = get_ds_count(DataTag.TRAIN)
        val_cnt = get_ds_count(DataTag.VALID)
        test_cnt = get_ds_count(DataTag.TEST)
        total_cnt =  train_cnt +  val_cnt + test_cnt
        logger.info(f'filtered dataset train_ratio={round(train_cnt/total_cnt, 1)} valid_ratio={round(val_cnt/total_cnt, 1)} test_ratio={round(test_cnt/total_cnt, 1)}')
        
        dst_ds.to_file(self.target_dirpath, format='yolov8')
        return self.target_dirpath
    
    @classmethod
    def _validate_exists(self, file_path: str, is_file: bool) -> bool:
        return os.path.exists(file_path) \
            and os.path.isfile(file_path) if is_file \
                else os.path.isdir(file_path)
    

class RoboflowDatasetValidator:
    def __init__(
        self, 
        crawler_filepath: str,
        dataset_dirpath: str,
        dataset_format: str = "yolov8",
        cfg: DataPrepConfig = None,
        validate_crawled_dataset: bool = True) -> None:
        """
        Args:
            crawler_filepath(`str`): file path of RoboflowCrawler output
            dataset_dirpath(`str`): dir path of RoboflowDownloader output
            dataset_format(`str`): format of the downloaded dataset by RoboflowDownloader
        """
        self.validate_crawled_dataset = get_default_on_none(validate_crawled_dataset, True)
        self.cfg = get_default_on_none(cfg, DataPrepConfig())
        self.crawler_filepath = get_default_on_none(crawler_filepath, self.cfg.rfmeta_file_path)
        self.dataset_dirpath = get_default_on_none(dataset_dirpath, self.cfg.dataset_download_path)
        
        if self.validate_crawled_dataset and not self._validate_exists(self.crawler_filepath, True):
            raise ValueError(f'{crawler_filepath} doesn\'t exist or is not a file.')
        
        if not self._validate_exists(self.dataset_dirpath, False):
            raise ValueError(f'{dataset_dirpath} doesn\'t exist or is not a dir.')
        
        if not dataset_format in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f'{dataset_format} is not in supported formats - {SUPPORTED_DATASET_FORMATS}')
        
        self.dataset_format = dataset_format
        self.dataset_reader = RoboflowDatasetReader(
            self.dataset_dirpath, self.dataset_format, self.cfg)
    
    
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
        source_ds = self.dataset_reader.read()
        source_ds_dict = {
            'image_name': [],
            'tag': [],
            'bbox_label': [],
            'bbox_count': []
        }
        for img_data in source_ds.get_all_images():
            bbox_label_counts = {}
            for bbox_data in source_ds.get_bboxes(img_data.name):
                if not bbox_data.label in bbox_label_counts:
                    bbox_label_counts[bbox_data.label] = 1
                else:
                    bbox_label_counts[bbox_data.label] += 1
            n = len(bbox_label_counts)
            source_ds_dict['image_name'].extend([img_data.name] * n)
            source_ds_dict['tag'].extend([img_data.tag.value] * n)
            for bbox_label, bbox_count in bbox_label_counts.items():
                source_ds_dict['bbox_label'].append(bbox_label)
                source_ds_dict['bbox_count'].append(bbox_count)
        
        downloaded_df = pd.DataFrame.from_dict(source_ds_dict)
        return downloaded_df
    
    def validate(self, full_dataset: pd.DataFrame) -> Tuple[bool, Tuple[str, bool]]:
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
        rfds_df = self._load_downloaded_dataset()
        full_df = full_dataset
        
        results = [
            ('rfds: not_empty', self.check_not_empty(rfds_df)),
            ('rfds: image_count', self.check_image_count(rfds_df, full_df)),
            ('rfds: label_count', self.check_label_phantom(rfds_df, full_df)),
            ('rfds: bbox_count', self.check_bbox_count(rfds_df, full_df))
        ]

        if self.validate_crawled_dataset:
            self._validate_crawl_dataset(self.crawler_filepath)
            rf_df = pd.read_csv(self.crawler_filepath, 
                    dtype={'image_name': str, 'bbox_label': str})
            results.extend([
                ('rfmeta: not_empty', self.check_not_empty(rf_df)),
                ('rfmeta: image_count', self.check_image_count(rf_df, full_df)),
                ('rfmeta: label_count', self.check_label_phantom(rf_df, full_df)),
                ('rfmeta: bbox_count', self.check_bbox_count(rf_df, full_df))
            ])

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
