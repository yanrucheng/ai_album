import os
import shutil
import shutil
import hashlib
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=4096)
def md5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class MyPath:

    def __init__(self, path):
        self.path = path

    @property
    def basename(self):
        """ Extracts the basename from a given file path. """
        return os.path.splitext(os.path.basename(self.path))[0]
    
    @property
    def extension(self):
        """ Extracts the extension from a given file path, without the dot. """
        return os.path.splitext(self.path)[1][1:]

    @property
    def abspath(self):
        return os.path.abspath(os.path.expanduser(self.path))
    
    @property
    def md5(self):
        return md5(self.path)

    @property
    def date(self):
        try:
            # Get creation time (on Windows) or the last metadata change (on Unix)
            creation_time = os.path.getctime(self.path)
    
            # Get last modification time
            modification_time = os.path.getmtime(self.path)
    
            # Format times or set to None if they're not available
            creation_date = datetime.fromtimestamp(creation_time).strftime('%y%m%d') if creation_time else None
            modification_date = datetime.fromtimestamp(modification_time).strftime('%y%m%d') if modification_time else None
    
            # Determine which date to use
            if creation_date:
                return creation_date
            elif modification_date:
                return modification_date
            else:
                return None
        except OSError as error:
            print(f"Error getting dates for {self.path}: {error}")
            return None


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        
def copy_file_as_cluster(clusters, target_path):

    def _copy(src, dst):
        # Copy the file
        shutil.copy2(src, dst)
    
        # Get timestamps from the source file
        stat_src = os.stat(src)
        atime = stat_src.st_atime  # Access time
        mtime = stat_src.st_mtime  # Modification time
    
        # Apply timestamps to the destination file
        os.utime(dst, (atime, mtime))
    
    def _copy_files_to_clusters(clusters, target_path, _current_path=""):
        for cluster_id, contents in clusters.items():
            # Create a new subdirectory for the current cluster
            cluster_dir = os.path.join(target_path, _current_path, str(cluster_id))
            os.makedirs(cluster_dir, exist_ok=True)
    
            if isinstance(contents, dict):
                # If the contents are a dictionary, recurse into it
                _copy_files_to_clusters(contents, target_path, os.path.join(_current_path, str(cluster_id)))
            elif isinstance(contents, list):
                # If the contents are a list, copy the files into the current cluster directory
                for file_path in contents:
                    _copy(file_path, cluster_dir)
                    
    _copy_files_to_clusters(clusters=clusters, target_path=target_path)