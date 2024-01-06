import os
import pickle
from PIL import Image


class CacheManager:

    ROOT_BASE = '~/.cache/ai_album/'
    
    def __init__(self, cache_path_prefix, root_path, cache_tag, generate_func, format_str="{base}_{cache_tag}.cache"):
        self.cache_path_prefix = cache_path_prefix
        self.root_path = os.path.abspath(root_path)
        self.cache_tag = cache_tag
        self.generate_func = generate_func
        self.format_str = format_str

    def _get_cache_file_path(self, path):

        root_p, root_folder_name = os.path.split(self.root_path.rstrip('/'))
        cache_path = os.path.abspath(path).replace(root_p, self.ROOT_BASE + self.cache_path_prefix)

        base, ext = os.path.splitext(cache_path)
        ext = ext[1:]
        p = self.format_str.format(base=base, ext=ext, cache_tag=self.cache_tag)
        
        return os.path.expanduser(p).lower()

    def load(self, path):
        def save(data, path):
            if isinstance(data, Image.Image):
                data.save(path, quality=50)
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


