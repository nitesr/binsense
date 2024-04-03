from .utils import backup_file

from typing import List, Tuple, Dict, Union, Any, Iterator
from uhashring import HashRing
from safetensors import torch as safetensors_torch
import safetensors, torch, os, logging

logger = logging.getLogger("__name__")

class EmbeddingDatastore:
    def __init__(self) -> None:
        pass
    
    def has(self, key: str) -> bool:
        pass
    
    def get_keys(self) -> Iterator:
        pass
    
    def get(self, key: str) -> torch.Tensor:
        pass
    
    def get_many(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        pass
    
    def put(self, key: str, value: torch.Tensor) -> None:
        pass
    
    def put_many(self, keys: List[str], value: torch.Tensor) -> None:
        pass
    
    def lookup(self, query: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        raise ValueError("Not supported!")


class SafeTensorEmbeddingDatastore(EmbeddingDatastore):
    def __init__(
        self, 
        dir_path: str, 
        req_partitions: int = 10, 
        read_only: bool = True,
        clean_state: bool = False) -> None:
        
        super(SafeTensorEmbeddingDatastore, self).__init__()
        self.dir_path = dir_path
        self.read_only = read_only
        if clean_state:
            self._clean_up(dir_path)
        
        self._initialize(req_partitions)
        if not self.read_only:
            self._check_initial_write_state(clean_state, req_partitions)
        
        self.hring = HashRing(nodes=[f'par{i}' for i in range(self.num_partitions)])
        self.file_paths = { key : os.path.join(dir_path, f'embeddings-{key}.safetensors') \
            for key in self.hring.get_nodes() }
    
    def _initialize(self, req_partitions: int):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        else:
            if not os.path.isdir(self.dir_path):
                raise ValueError(f'{self.dir_path} is not a directory!')
        
        par_count_fp = os.path.join(self.dir_path, 'partition_count.dat')
        if not os.path.exists(par_count_fp):
            with open(par_count_fp, 'w') as f:
                f.write(str(req_partitions))
            self.num_partitions = req_partitions
        else:
            with open(par_count_fp, 'r') as f:
                self.num_partitions = int(f.readline().strip())
    
    def _check_initial_write_state(self, clean_state: bool, req_partitions: int) -> None:
        if not clean_state and req_partitions != self.num_partitions:
            raise ValueError('start on a clean state to change the partitions')
    
    def _clean_up(self, dir_path : str) -> None:
        for chld_fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, chld_fname)
            _, ext = os.path.splitext(fpath)
            if ext == '.safetensors' or ext == ".dat":
                bkp_fpath = backup_file(fpath)
                logger.info(f"backed up {fpath} to {bkp_fpath}")
    
    def _to_partition(self, key: str) -> str:
        par = self.hring.get_node(key)
        return self.file_paths[par]
    
    def _validate_read_only(self) -> None:
        if self.read_only:
            raise ValueError('this is read-only datastore!')
    
    def get_keys(self) -> Iterator:
        class LocalIterator(Iterator):
            def __init__(self, fpaths: Any) -> None:
                super(LocalIterator, self).__init__()
                self.fpaths = fpaths
                self.fpaths_ptr = -1
                self.par_keys = []
                self.par_keys_ptr = -1
                self._fetch_keys()
            
            def _fetch_keys(self) -> bool:
                while self.par_keys_ptr >= len(self.par_keys)-1:
                    if self.fpaths_ptr >= len(self.fpaths)-1:
                        self.par_keys = []
                        self.par_keys_ptr = -1
                        return False
                    
                    self.fpaths_ptr += 1
                    with safetensors.safe_open(
                        self.fpaths[self.fpaths_ptr], framework="pt") as f:
                        #TODO: use key iterator instead of loading
                        self.par_keys = [k for k in f.keys()]
                        self.par_keys_ptr = -1
                
                return True
            
            def __iter__(self) -> Iterator:
                return self
            
            def __next__(self) -> str:
                if not self._fetch_keys():
                    raise StopIteration
                
                self.par_keys_ptr += 1
                val = self.par_keys[self.par_keys_ptr]
                
                return val
        
        return LocalIterator([fpath for fpath in self.file_paths.values()])
    
    def _bulk_get(self, part_fpath: str, keys: List[str], device: Union[str, Any] = "cpu") -> Dict[str, torch.Tensor]:
        tensors_dict = {}
        if os.path.exists(part_fpath):
            with safetensors.safe_open(part_fpath, framework="pt", device=device) as f:
                for key in keys:
                    if key in f.keys():
                        tensors_dict[key] = f.get_tensor(key)
        return tensors_dict
            
    def _upsert(self, part_fpath: str, tensor_dict: Dict[str, torch.Tensor]) -> None:
        self._validate_read_only()
        tensors = {}
        if os.path.exists(part_fpath):
            with safetensors.safe_open(part_fpath, framework="pt", device="cpu") as f:
                for fkey in f.keys():
                    tensors[fkey] = f.get_tensor(fkey)
        tensors.update(tensor_dict)
        safetensors_torch.save_file(tensors, part_fpath)
    
    def get(self, key: str, device: Union[str, Any] = "cpu") -> torch.Tensor:
        fpath = self._to_partition(key)
        tensors = self._bulk_get(fpath, [key], device)
        return tensors[key] if key in tensors.keys() else None
            
    def put(self, key: str, value: torch.Tensor) -> None:
        fpath = self._to_partition(key)
        self._upsert(fpath, {key: value})
    
    def put_many(self, keys: List[str], values: torch.Tensor) -> None:
        fpath_dict = {}
        for i, k in enumerate(keys):
            fpath = self._to_partition(k)
            if not fpath in fpath_dict:
                fpath_dict[fpath] = []
            fpath_dict[fpath].append((i, k))
        
        for fpath in fpath_dict.keys():
            tensor_dict = {}
            for i, k in fpath_dict[fpath]:
                tensor_dict[k] = values[i]
            self._upsert(fpath, tensor_dict)
    
    def get_many(self, keys: List[str], device: Union[str, Any] = "cpu") -> Dict[str, torch.Tensor]:
        fpath_dict = {}
        for k in keys:
            fpath = self._to_partition(k)
            if not fpath in fpath_dict:
                fpath_dict[fpath] = []
            fpath_dict[fpath].append(k)
        
        tensors = {}
        for fpath in fpath_dict.keys():
            ftensors = self._bulk_get(fpath, fpath_dict[fpath], device)
            tensors.update(ftensors)
        return tensors