from typing import Dict, Any, Mapping
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from .model import Owlv2ForObjectDetection
from .config import Owlv2Config

import json, torch

HUB_PRETRAINED_MODEL  = "google/owlv2-base-patch16-ensemble"
PROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"
MODEL_CONFIG_FILENAME = "config.json"
MODEL_SAFE_TENSORS_FILENAME = "model.safetensors"
MODEL_PYTORCH_FILENAME = "pytorch_model.bin"

def _download(repo_id, file_name) -> str:
    """
    Downloads the file from the hugging face hub
    Args:
        repo_id: public repository id
        file_name: relative file path in the repository
    Returns:
        local_file_path: file path in local cache directory
    """
    return hf_hub_download(repo_id=repo_id, filename=file_name)

def load_config(
    file_name: str, 
    repo_id: str = HUB_PRETRAINED_MODEL) -> Dict[str, Any]:
    """
    Downloads the file, loads into a Dict and returns
    """
    file_path = _download(repo_id, file_name)
    with open(file_path, 'r') as f:
        cfg = json.load(f)
    return cfg


def load_owlv2processor_config(repo_id: str = HUB_PRETRAINED_MODEL) -> Dict[str, Any]:
    """
    Downloads the processor config file, loads into a Dict and returns
    """
    return load_config(
        file_name=PROCESSOR_CONFIG_FILENAME, 
        repo_id=HUB_PRETRAINED_MODEL)

def load_owlv2model_config(repo_id: str = HUB_PRETRAINED_MODEL) -> Dict[str, Any]:
    """
    Downloads the model config file, loads into a Dict and returns
    """
    return load_config(
        file_name=MODEL_CONFIG_FILENAME, 
        repo_id=HUB_PRETRAINED_MODEL)

def _get_model_module_keys():
    #TODO: check if we can do without loading the model
    model = Owlv2ForObjectDetection(Owlv2Config())
    model_module_keys = model.__dict__['_modules'].keys()
    del model
    return model_module_keys

def _filter_statedict_modules(state_dict):
    """
    filters the passed state_dict for the applicable 
        modules in the model `Owlv2ForObjectDetection`
    """
    model_module_keys = _get_model_module_keys()
    filtered_state_dict = dict()
    for key in state_dict.keys():
        new_key = key
        first_token = key.split('.')[0]
        if first_token == 'owlv2':
            new_key = '.'.join(key.split('.')[1:])
            
        if new_key.split('.')[0] in model_module_keys:
            filtered_state_dict[new_key] = state_dict[key]
    return filtered_state_dict

def load_owlv2model_statedict(repo_id: str = HUB_PRETRAINED_MODEL, safe_tensors: bool =True) -> Mapping[str, Any]:
    """
    Downloads the `Owlv2ForObjectDetection` model state and returns as a Dict.
    Args:
        repo_id: str
            public repository id for the owlv2
        safe_tensors: bool
            either load safe tensors or the pytorch pickle
    """
    model_filename = MODEL_SAFE_TENSORS_FILENAME if safe_tensors else MODEL_PYTORCH_FILENAME
    model_file_path = _download(repo_id, model_filename)
    state_dict = {}
    
    if safe_tensors:
        with safe_open(model_file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(model_file_path)
    
    return _filter_statedict_modules(state_dict)
    
