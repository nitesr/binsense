from ...dataprep.embedding_util import BBoxDatasetEmbedder
from ...dataprep.model_util import Owlv2BBoxEmbedder
from ...dataprep.config import DataPrepConfig
from ...embed_datastore import SafeTensorEmbeddingDatastore
from ...owlv2 import Owlv2ForObjectDetection, Owlv2Config
from ...owlv2 import hugg_loader as hloader

import argparse, logging

def _get_embed_store(
    cfg: DataPrepConfig, 
    num_bbox_labels: int, 
    owl_model_cfg: Owlv2Config) -> SafeTensorEmbeddingDatastore:
    
    embedstore_size =  num_bbox_labels * owl_model_cfg.vision_config.class_embed_size * 16
    embedstore_partition_size = 4 * 1024 * 1024 # 4MB per partition
    req_partitions = max(embedstore_size // embedstore_partition_size, 1)
    print(f"required partitions on embedding store are {req_partitions}")
    embed_store = SafeTensorEmbeddingDatastore(
        cfg.embed_store_dirpath, 
        req_partitions=max(embedstore_size // embedstore_partition_size, 1),
        read_only=False,
        clean_state=True
    )
    return embed_store

def _get_bbox_embedder(
    cfg: DataPrepConfig, 
    owl_model_cfg: Owlv2Config) -> Owlv2BBoxEmbedder:
    
    model = Owlv2ForObjectDetection(owl_model_cfg)
    model.load_state_dict(hloader.load_owlv2model_statedict())
    bbox_embedder = Owlv2BBoxEmbedder(model=model)
    return bbox_embedder
    
def run_embedder(num_bbox_labels: int, batch_size: int, test_run: bool):
    cfg = DataPrepConfig()
    owl_model_cfg = Owlv2Config(**hloader.load_owlv2model_config())
    bbox_embedder = _get_bbox_embedder(cfg, owl_model_cfg)
    embed_store =  _get_embed_store(cfg, num_bbox_labels, owl_model_cfg)

    generator = BBoxDatasetEmbedder(bbox_embedder, embed_store, cfg)
    generator.generate(batch_size, test_run)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s : %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate", help="preannotate using owlv2 model",
        action="store_true")
    parser.add_argument(
        "--batch_size", help="batch size for the model", default=8, type=int)
    parser.add_argument(
        "--num_bbox_labels", help="number of bbox labels", default=2000, type=int)
    parser.add_argument(
        "--test_run", help="do a test run",
        action="store_true")
    
    args = parser.parse_args()
    if args.generate:
        run_embedder(args.num_bbox_labels, args.batch_size, args.test_run)