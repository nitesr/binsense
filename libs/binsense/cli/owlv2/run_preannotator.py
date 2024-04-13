
from ...dataprep.preannotate import Preannotator, RoboflowUploadBuilder
from ...lightning.owlv2_model import Owlv2BboxPredictor
from ...dataprep.config import DataPrepConfig
from ...owlv2 import Owlv2ForObjectDetection, Owlv2Config
from ...owlv2 import hugg_loader as hloader

import argparse, logging, sys

def run_annotate(batch_size: int, test_run: bool) -> None:
    model = Owlv2ForObjectDetection(Owlv2Config(**hloader.load_owlv2model_config()))
    model.load_state_dict(hloader.load_owlv2model_statedict())
    bbox_predictor = Owlv2BboxPredictor(model=model)
    cfg = DataPrepConfig()
    annotator = Preannotator(bbox_predictor, cfg)
    chkpt_files = annotator.preannotate(batch_size, test_run)
    print("checkpoint at", chkpt_files)

def run_create_dataset() -> None:
    builder = RoboflowUploadBuilder(DataPrepConfig())
    upload_path = builder.build()
    print("built at", upload_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotate", help="preannotate using owlv2 model",
        action="store_true")
    parser.add_argument(
        "--batch_size", help="batch size for the model", default=8, type=int)
    parser.add_argument(
        "--test_run", help="do a test run",
        action="store_true")
    parser.add_argument(
        "--create_dataset", help="creates the preannotated dataset in Yolov8 format",
        action="store_true")
    parser.add_argument(
        "--upload_dataset", help="uploads the preannotated dataset to roboflow")
    
    args = parser.parse_args()
    
    if args.annotate:
        run_annotate(args.batch_size, args.test_run)
    if args.create_dataset:
        run_create_dataset()
    if args.upload_dataset is not None:
        print("Not supported! can be done manually")
        sys.exit(1)
    sys.exit(0)