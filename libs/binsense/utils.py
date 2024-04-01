from typing import Any, Optional, List, TypeVar
from pathlib import Path
import os, shutil, re

T = TypeVar("T")

def get_default_on_none(val: T, default_val: T) -> T:
    return val if val is not None else default_val

def backup_file(file_path: str, bkp_extn: str = 'bkp') -> str:
    bkp_number = 0
    dir_path, file_name = os.path.split(file_path)
    bkp_pattern = re.compile(f'.*\.[0-9]+\.bkp$')
    for fname in os.listdir(dir_path):
        if fname == file_name and bkp_pattern.match(fname):
            bkp_number = max(bkp_number, int(fname.split('.')[-2]))
    bkp_number += 1
    shutil.move(file_path, file_path+f'.{bkp_number}.bkp')
    return file_path+f'.{bkp_number}.bkp'

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
