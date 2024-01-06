import os
import pickle
from PIL import Image
import utils
import glob

IMG_QUALITY = 50

class CacheManager:

    CACHE_BASE = '.similarity_cache/'

    def __init__(self,
                 generate_func,
                 cache_tag='cache',
                 target_path='',
                 cache_root_path='',
                 cache_folder_name='',
                 format_str="{base}_{ext}_{cache_tag}_{md5}.jpg"):

        assert target_path or (cache_root_path and cache_folder_name), \
            'Either target path or (cache root path and cache folder name) should be provided'

        self.cache_tag = cache_tag
        self.generate_func = generate_func
        self.format_str = format_str

        if target_path:
            cache_root_path, cache_folder_name = os.path.split(target_path.rstrip('/').rstrip('\\'))

        self.cache_folder_name = cache_folder_name
        self.cache_folder = os.path.abspath(os.path.join(cache_root_path, self.CACHE_BASE, cache_folder_name))

    def _get_cache_file_path(self, path):
        p = utils.MyPath(path)
        basename = self.format_str.format(base=p.basename, ext=p.extension, cache_tag=self.cache_tag, md5=p.md5)
        cache_p = utils.MyPath(os.path.join(self.cache_folder, basename))
        
        return cache_p.abspath
    

    def load(self, path):
        def save(data, path):
            if isinstance(data, Image.Image):
                data.save(path, quality=IMG_QUALITY)
            elif isinstance(data, str):
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(data)
            else:
                with open(path, 'wb') as file:
                    pickle.dump(data, file)

        def load_individual_file(path):
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                return Image.open(path)
            elif path.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                with open(path, 'rb') as file:
                    return pickle.load(file)

        cache_file_path = self._get_cache_file_path(path)

        # Check if cache file exists or if it matches any files when wildcard is present
        if '*' in cache_file_path:
            matched_files = glob.glob(cache_file_path)
            if not matched_files:
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                i = 0
                for item in self.generate_func(path):
                    i += 1
                    item_path = cache_file_path.replace('*', str(i))
                    save(item, item_path)
                    
            matched_files = glob.glob(cache_file_path)
            return [load_individual_file(file_path) for file_path in sorted(matched_files)]
        else:
            if not os.path.exists(cache_file_path):
                data = self.generate_func(path)
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                save(data, cache_file_path)

            if os.path.exists(cache_file_path):
                return load_individual_file(cache_file_path)

        return None


