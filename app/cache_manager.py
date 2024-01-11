import os
import pickle
from PIL import Image
import utils
import glob
import yaml
import numpy as np

from typing import Callable, Hashable, Union, Dict, List, Any


IMG_QUALITY = 50

Cachable = Union[Dict, List, str, np.ndarray, Any]
Path = str

class CacheManager:

    CACHE_BASE = '.similarity_cache/'

    def __init__(self,
                 generate_func: Callable[[Hashable], Cachable],
                 target_path: Path = '',
                 cache_root_path: Path = '',
                 cache_folder_name: str = '',
                 cache_key_type: str = 'path', # 'path' or 'hashable'
                 format_str: str = "{base}_{ext}_{file_hash}.jpg"):

        assert target_path or (cache_root_path and cache_folder_name), \
            'Either target path or (cache root path and cache folder name) should be provided'

        assert cache_key_type in ('path', 'hashable'), \
            "cache_key_type should be within 'path' and 'hashable'."


        self.generate_func = generate_func
        self.format_str = format_str

        if target_path:
            cache_root_path, cache_folder_name = os.path.split(target_path.rstrip('/').rstrip('\\'))

        self.cache_folder_name = cache_folder_name
        self.cache_folder = os.path.abspath(os.path.join(cache_root_path, self.CACHE_BASE, cache_folder_name))

        self._to_cache_path_func = {
            'path': self._get_cache_file_path_from_path,
            'hashable': self._get_cache_file_path_from_hashable,
        }[cache_key_type]
        self.cache_key_type = cache_key_type

    def to_cache_path(self, key_obj):
        return self._to_cache_path_func(key_obj)

    def _get_cache_file_path_from_hashable(self, hashable_obj):
        basename = utils.stable_hash(hashable_obj)
        basepath = self.format_str.format(base=basename, ext='hashable', file_hash=basename)
        cache_p = utils.MyPath(os.path.join(self.cache_folder, basepath))
        return cache_p.abspath

    def _get_cache_file_path_from_path(self, path):
        p = utils.MyPath(path)
        basename = self.format_str.format(base=p.basename, ext=p.extension, file_hash=p.hash)
        cache_p = utils.MyPath(os.path.join(self.cache_folder, basename))
        return cache_p.abspath

    def load(self, key_obj):

        def _post_save(path):
            if self.cache_key_type != 'path':
                return
            utils.inplace_overwrite_meta(key_obj, path)

        def _save(data, path):
            if isinstance(data, Image.Image):
                data.save(path, quality=IMG_QUALITY)
                data.close()
            elif isinstance(data, str):
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(data)
            elif isinstance(data, dict):
                with open(path, 'w', encoding='utf-8') as file:
                    yaml.dump(data, file)
            elif isinstance(data, np.ndarray):
                np.save(path, data)  # NumPy arrays are saved as .npy files
            else:
                with open(path, 'wb') as file:
                    pickle.dump(data, file)

            _post_save(path)

        def _load_individual_file(path):
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                with Image.open(path) as img:
                    img.load()
                    return img
            elif path.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif path.lower().endswith('.yaml') or path.lower().endswith('.yml'):
                with open(path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            elif path.lower().endswith('.npy'):
                return np.load(path)  # Load NumPy array from .npy file
            else:
                with open(path, 'rb') as file:
                    return pickle.load(file)


        cache_file_path = self.to_cache_path(key_obj)

        # Check if cache file exists or if it matches any files when wildcard is present
        if '*' in cache_file_path:
            matched_files = glob.glob(cache_file_path)
            if not matched_files:
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                i = 0
                for item in self.generate_func(key_obj):
                    i += 1
                    item_path = cache_file_path.replace('*', str(i))
                    _save(item, item_path)

            matched_files = glob.glob(cache_file_path)
            return [_load_individual_file(file_path) for file_path in sorted(matched_files)]
        else:
            if not os.path.exists(cache_file_path):
                data = self.generate_func(key_obj)
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                _save(data, cache_file_path)

            if os.path.exists(cache_file_path):
                return _load_individual_file(cache_file_path)

        return None


