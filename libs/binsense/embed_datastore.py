from typing import List, Tuple

import numpy as np
from numpy.core.multiarray import array as array

class EmbeddingDatastore:
    def __init__(self) -> None:
        pass
    
    def has(self, key: str) -> bool:
        pass
    
    def get(self, key: str) -> np.array:
        pass
    
    def put(self, key:str, value: np.array) -> None:
        pass
    
    def lookup(self, query: np.array) -> Tuple[List[str], np.ndarray]:
        pass

class InMemoryEmbeddingDatastore(EmbeddingDatastore):
    def __init__(self) -> None:
        super().__init__()
        self.datastore = dict()
    
    def has(self, key: str) -> bool:
        return key in self.datastore.keys()
    
    def get(self, key: str) -> np.array:
        return self.datastore[key]
    
    def put(self, key: str, value: np.array) ->None:
        self.datastore[key] = value
    
    def lookup(self, query: np.array) -> Tuple[List[str] | np.ndarray]:
        raise ValueError("Not supported!")
