from ...dataprep.roboflow_util import RoboflowDownloader, RoboflowDatasetReader
from ...dataprep.config import DataPrepConfig
from ...dataset_util import DataTag, ImageData, YoloDataset
from ...img_utils import annotate_image
from ...utils import load_params, get_default_on_none

from matplotlib import pyplot as plt
from PIL.Image import Image as PILImage

import numpy as np
import logging, os, argparse, sys, random, PIL, torch


def run_download_dataset(
    target_dir: str = None,
    workspace: str = None, 
    project: str = None, 
    version: int = None, 
    api_key: str = None):
    
    cfg = DataPrepConfig()
    if target_dir == cfg.root_dir:
        target_dir = None # default to the ones in cfg
    elif target_dir is not None:
        target_dir = os.path.join(
            target_dir, os.path.split(cfg.dataset_download_path)[1])
    
    downloader = RoboflowDownloader(
        api_key=api_key, workspace=workspace, 
        project=project, version=version, 
        cfg=cfg)
    dirpath = downloader.download(target_dir=target_dir)
    print('roboflow dataset downloaded at', dirpath)
    return dirpath

def sample_random_images(dirpath: str, n: int = 4) -> None:
    ds = RoboflowDatasetReader(
        dataset_dirpath=dirpath,
        cfg=DataPrepConfig()).read()
    
    def annotate(img_data, ds: YoloDataset) -> PILImage:
        image = PIL.Image.open(img_data.path).convert('RGB')
        labels = [bbox_data.label for bbox_data in ds.get_bboxes(img_data.name)]
        bboxes_cxy = np.array([ bbox_data.to_array() for bbox_data in ds.get_bboxes(img_data.name) ])
        seg_coords = [ bbox_data.segmentation[0] for bbox_data in ds.get_bboxes(img_data.name) ]
        return annotate_image(image, labels, bboxes_cxy, seg_coords)        

    train_images = ds.get_images(DataTag.TRAIN)
    test_images = ds.get_images(DataTag.TEST)
    val_images = ds.get_images(DataTag.VALID)
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)

    fig, axs = plt.subplots(n, 3, figsize=(3*3, n*3))
    for i in range(0, n):
        axs[i][0].imshow(annotate(train_images[i], ds))
        if i < len(val_images):
            axs[i][1].imshow(annotate(val_images[i], ds))
        if i < len(test_images):
            axs[i][2].imshow(annotate(test_images[i], ds))

    fig.suptitle("train | valid | test annotated images")
    plt.savefig(
        fname=os.path.join(
            cfg.root_dir, 'sample_gt_annotation_images.png'), 
        format='png')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    rfmeta_fname = os.path.split(cfg.rfmeta_file_path)
    download_dirname = os.path.split(cfg.dataset_download_path)
    params = load_params('./params.yaml')
    
    parser.add_argument(
        "--download", help="download metadata and dataset",
        action="store_true")
    parser.add_argument(
        "--workspace", help="workspace name from https://app.roboflow.com/",
        default=params.binseg_roboflow.workspace, required=False)
    parser.add_argument(
        "--project", help="project name from https://app.roboflow.com/",
        default=params.binseg_roboflow.project, required=False)
    parser.add_argument(
        "--dataset_version", help="dataset version generated from https://app.roboflow.com/.",
        default=params.binseg_roboflow.dataver, required=False, type=int)
    parser.add_argument(
        "--target_dir", help=f"target directory to download dataset ({download_dirname})",
        default=cfg.root_dir, required=False)
    parser.add_argument(
        "--api_key", help="roboflow api key", required=True)
    
    args = parser.parse_args()
    
    if args.download:
        print("downloading dataset..")
        dirpath = run_download_dataset(
            target_dir=args.target_dir,
            workspace=args.workspace,
            project=args.project,
            version=args.dataset_version,
            api_key=args.api_key)
        sample_random_images(dirpath)

    sys.exit(0)
