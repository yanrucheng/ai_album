import os
import shutil
import hashlib
from functools import lru_cache
from datetime import datetime
from collections import Counter
import hashlib

import sys
import contextlib


@contextlib.contextmanager
def suppress_c_stdout_stderr():
    """A context manager that redirects C-level stdout and stderr to /dev/null"""
    # Flush Python-level buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate file descriptors
    with open(os.devnull, 'wb') as fnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())

        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())

        try:
            yield
        finally:
            # Restore file descriptors
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def get_mode(data):
    '''Return a single mode. when there are multiple, return the samllest'''
    if not data:
        raise ValueError("The data list is empty")

    # Count the frequency of each item
    counts = Counter(data)

    # Find the maximum frequency
    max_frequency = max(counts.values())

    # Extract items with the maximum frequency and take the minimum
    mode = min(item for item, count in counts.items() if count == max_frequency)
    return mode

@lru_cache(maxsize=8192)
def stable_hash(obj) -> str:
    # Convert the object to a string in a consistent manner
    # Use repr for a standardized representation
    obj_str = repr(obj)

    # Encode the string into bytes
    obj_bytes = obj_str.encode('utf-8')

    # Create an MD5 hash object and update it with the byte-encoded data
    hash_object = hashlib.md5(obj_bytes)

    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex

@lru_cache(maxsize=8192)
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

Path = str
def copy_with_meta(src: Path, dst: Path):
    # Copy the file
    shutil.copy2(src, dst)
    inplace_overwrite_meta(src, dst)

def inplace_overwrite_meta(src: Path, target: Path):
    # Get timestamps from the source file
    stat_src = os.stat(src)
    atime = stat_src.st_atime  # Access time
    mtime = stat_src.st_mtime  # Modification time

    # Apply timestamps to the destination file
    os.utime(target, (atime, mtime))
    
