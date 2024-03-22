from typing import Any, Optional, List
from pathlib import Path
import os

def default_on_none(dict_obj : dict, hkeys: list, default_value: Optional[Any] = None):
    if dict_obj:
        obj = dict_obj
        for key in hkeys:
            if obj and key in obj.keys():
                obj = obj[key]
            else:
                return default_value
        return obj
    else:
        return default_value

class FileIterator:
    def __init__(self, dir, extensions: Optional[List[str]]=[]) -> None:
        self.dir = dir
        self.files = []
        
        for f in os.listdir(self.dir):
            if Path(f).suffix in extensions:
                self.files.append(f)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        if i < len(self.files):
            return str(Path(self.dir) / self.files[i])
        else:
            raise IndexError('{} is out of {}'.format(i, len(self.files)))

class ImageFileIterator(FileIterator):
    def __init__(self, dir, extensions=['.jpg', '.jpeg']) -> None:
        super(ImageFileIterator, self).__init__(dir, extensions)
