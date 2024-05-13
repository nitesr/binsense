
from ...dataprep.preannotate import Preannotator, RoboflowUploadBuilder
from ...lightning.owlv2_model import Owlv2BboxPredictor
from ...dataprep.config import DataPrepConfig
from ...owlv2 import Owlv2ForObjectDetection, Owlv2Config
from ...owlv2 import hugg_loader as hloader
from ...utils import load_params, get_default_on_none

from ...dataprep.downloader import download as bindata_downloader
from ...dataprep.metadata import load as binmetadata_loader


import argparse, logging, sys, os

def run_annotate(batch_size: int, device: str = 'cpu', test_run: bool = False) -> None:
    model = Owlv2ForObjectDetection(Owlv2Config(**hloader.load_owlv2model_config()))
    model.load_state_dict(hloader.load_owlv2model_statedict())
    bbox_predictor = Owlv2BboxPredictor(model=model)
    cfg = DataPrepConfig()
    annotator = Preannotator(bbox_predictor, config=cfg, device=device, test_run=test_run)
    chkpt_files = annotator.preannotate(batch_size, test_run)
    print("checkpoint at", chkpt_files)

def run_create_dataset() -> None:
    builder = RoboflowUploadBuilder(DataPrepConfig())
    upload_path = builder.build()
    print("built at", upload_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    cfg = DataPrepConfig()
    params = load_params('./params.yaml')

    parser.add_argument(
        "--num_workers", help=f"number of workers to download orig dataset",
        default=params.data_download.num_workers, required=False, type=int)
    parser.add_argument(
        "--batch_size", help="batch size for the model", default=params.preannotate.batch_size, type=int)
    parser.add_argument(
        "--test_run", help="do a test run",
        action="store_true")
    parser.add_argument(
        "--device", help="device to run the model in", type=str,
        default="cpu")
    parser.add_argument(
        "--annotate", help="preannotate using owlv2 model",
        action="store_true")
    parser.add_argument(
        "--create_dataset", help="creates the preannotated dataset in Yolov8 format",
        action="store_true")
    parser.add_argument(
        "--upload_dataset", help="uploads the preannotated dataset to roboflow",
        action="store_true")
    
    args = parser.parse_args()
    
    if args.annotate:
        run_annotate(
            args.batch_size, 
            device=args.device, 
            test_run=args.test_run
        )
    if args.create_dataset:
        run_create_dataset()
    if args.upload_dataset is not None:
        bindata_downloader(max_workers=args.num_workers)
        _, all_item_df = binmetadata_loader(max_workers=args.num_workers)
        all_item_df["image_name"] = all_item_df["bin_id"] + '.jpg'
        first_column = all_item_df.pop('image_name') 
        all_item_df.insert(0, 'image_name', first_column) 
        all_item_df.sort_values(by=["bin_id", "item_id"], inplace=True)
        all_item_df.drop(["bin_id"], axis=1, inplace=True)
        ann_fp = os.path.join(cfg.root_dir, 'data_for_annotators.csv')
        all_item_df.to_csv(ann_fp, index=False)
        print(f'data for annotation is @ {ann_fp}')
        print(f"Upload to Roboflow not supported yet! Please upload the folder({cfg.robo_upload_dir}) manually")

    sys.exit(0)