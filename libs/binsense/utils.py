from typing import Any, Optional

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