import os
import platform
import shutil
import hashlib
import functools
from functools import lru_cache
from datetime import datetime
import time
from collections import Counter
import hashlib
from pathlib import Path

import sys
import contextlib

import json
import re
import pytz

import logging
logger = logging.getLogger(__name__)


# Print related
@contextlib.contextmanager
def suppress_c_stdout_stderr(suppress_stdout=True, suppress_stderr=False):
    """A context manager that redirects C-level stdout and/or stderr to /dev/null"""
    # Flush Python-level buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate file descriptors
    old_fds = {}
    with open(os.devnull, 'wb') as fnull:
        if suppress_stdout:
            old_fds['stdout'] = os.dup(sys.stdout.fileno())
            os.dup2(fnull.fileno(), sys.stdout.fileno())

        if suppress_stderr:
            old_fds['stderr'] = os.dup(sys.stderr.fileno())
            os.dup2(fnull.fileno(), sys.stderr.fileno())

        try:
            yield
        finally:
            # Restore file descriptors
            if suppress_stdout:
                os.dup2(old_fds['stdout'], sys.stdout.fileno())
                os.close(old_fds['stdout'])

            if suppress_stderr:
                os.dup2(old_fds['stderr'], sys.stderr.fileno())
                os.close(old_fds['stderr'])

@contextlib.contextmanager
def suppress_stdout_stderr(suppress_stdout=True, suppress_stderr=False):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    if suppress_stdout:
        sys.stdout = StringIO()
    if suppress_stderr:
        sys.stderr = StringIO()

    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# Arithmetic calculation related

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

# Hash related

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


@lru_cache(maxsize=8192)
def partial_file_hash(path, chunk_size=4096):
    hash_md5 = hashlib.md5()
    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        if file_size <= chunk_size * 3:
            # If the file is small, read the entire file
            hash_md5.update(f.read())
        else:
            # Read the start, middle, and end of the file
            f.seek(0)
            hash_md5.update(f.read(chunk_size))
            f.seek(file_size // 2)
            hash_md5.update(f.read(chunk_size))
            f.seek(-chunk_size, os.SEEK_END)
            hash_md5.update(f.read(chunk_size))

    return hash_md5.hexdigest()


# Path related

# a decorator
def ensure_unique_path(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_path = func(*args, **kwargs)
        if not original_path:
            return None

        # Ensure the path is unique by appending a number if it already exists
        base, extension = os.path.splitext(original_path)
        counter = 1
        unique_path = original_path
        while os.path.exists(unique_path):
            unique_path = f"{base}-{counter}{extension}"
            counter += 1

        return unique_path
    return wrapper

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
    def hash(self):
        return partial_file_hash(self.path)

    @property
    def timezone(self):
        return pytz.timezone('Asia/Shanghai')

    @property
    def timestamp(self):
        timestamp = None
        try:
            stat_info = os.stat(self.path)
            if platform.system() == 'Windows':
                timestamp = stat_info.st_ctime
            else:
                try:
                    timestamp = stat_info.st_birthtime  # macOS
                except AttributeError:
                    try:
                        timestamp = stat_info.st_mtime  # Linux/Unix fallback
                    except AttributeError:
                        timestamp = time.time()  # Ultimate fallback
        except (OSError, AttributeError):
            timestamp = time.time()
        return timestamp

    @property
    def timestr(self):
        if self.timestamp is None:
            return None
        return datetime.fromtimestamp(self.timestamp, self.timezone).strftime('%Y-%m-%d %H:%M:%S')

    @property
    def date(self):
        if self.timestamp is None:
            return None
        return datetime.fromtimestamp(self.timestamp, self.timezone).strftime('%y%m%d')

    @property
    def time_of_a_day(self):
        """
        Return a human-readable time of day for a given timestamp.
        """
        if self.timestamp is None:
            return None
        timestamp = datetime.fromtimestamp(self.timestamp, self.timezone)
        hour = timestamp.hour

        if 5 <= hour < 12:
            return 'Morning'
        elif hour == 12:
            return 'Noon'
        elif 12 < hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        elif 21 <= hour < 24:
            return 'Night'
        else:  # from midnight to 5 am
            return 'Midnight'

# OS related

PathType = str

def create_relative_symlink(target_path: PathType, link_folder: PathType):
    """
    Create a relative symbolic link inside 'link_folder' pointing to 'target_path'.
    The name of the symbolic link will be the same as the name of the target.

    Args:
    target_path (str): The path to the target file or directory.
    link_folder (str): The folder where the symbolic link will be created.
    """
    assert target_path and link_folder, \
            f'Both target_path, link_folder are a must. got target_path={target_path}; link_folder={link_folder}'

    # Convert strings to Path objects and resolve to absolute paths
    target_path = Path(target_path).resolve()
    link_folder = Path(link_folder).resolve()

    link_folder.mkdir(parents=True, exist_ok=True)
    link_name = target_path.name
    rel_path = os.path.relpath(target_path, link_folder)
    link_path = link_folder / link_name

    try:
        os.symlink(rel_path, link_path)
    except Exception as e:
        print(f"Fail to build symlink at {str(rel_path)} for {str(target_path)}. Details: {e}")


def safe_delete(file_path):
    # Check if the file exists to avoid FileNotFoundError
    if not os.path.isfile(file_path):
        return

    try:
        os.remove(file_path)
    except PermissionError as e:
        # Raise PermissionError to be handled by the caller
        raise PermissionError(f"Permission denied: {e}")
    except Exception as e:
        # Handle other possible exceptions
        print(f"An error occurred: {e}")


# Example usage
# create_relative_symlink('/path/to/target', '/path/to/link_folder')

def safe_move(src: PathType, dst: PathType) -> bool:
    """
    Safely move a file from src to dst while preserving metadata and avoiding overwrites.
    If destination exists, automatically appends (1), (2), etc. to the filename.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if operation succeeded, False otherwise
    """
    assert src and dst, f'Both src, dst are a must. got src={src}; dst={dst}'

    try:
        # Validate inputs
        if not src or not dst:
            raise ValueError(f'Both src and dst are required. Got src={src}, dst={dst}')


        # Convert strings to Path objects and resolve to absolute paths
        src = Path(src).resolve()
        dst = Path(dst).resolve()

        dst.mkdir(parents=True, exist_ok=True)
        dst_name = src.name
        rel_path = os.path.relpath(src, dst)

        src_path = src
        dst_path = dst / dst_name

        # Check if source exists
        if not src.exists():
            raise FileNotFoundError(f'Source file does not exist: {src}')
            
        # Check if source is a file
        if not src.is_file():
            raise ValueError(f'Source is not a file: {src}')
            
        # Create parent directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle filename collisions
        if dst_path.exists():
            base = dst_path.stem
            ext = dst_path.suffix
            counter = 1
            while True:
                new_name = f"{base} ({counter}){ext}"
                new_path = dst_path.with_name(new_name)
                if not new_path.exists():
                    dst_path = new_path
                    break
                counter += 1
        
        # First copy with metadata
        shutil.copy2(src, dst_path)
        
        # Handle additional metadata if needed
        try:
            inplace_overwrite_meta(src, dst_path)
        except Exception as meta_error:
            logger.debug(f'Failed to copy metadata from {src} to {dst_path}: {meta_error}')
            # The copy succeeded even if metadata failed, so we continue
            return False
            
        # Only remove source if copy succeeded
        src.unlink()
        logger.info(f'{src} successfully moved to {dst_path}.')
        
        return True
        
    except Exception as e:
        logger.debug(f'Failed to move file from {src} to {dst}: {e}', exc_info=True)
        return False

def copy_with_meta(src: PathType, dst: PathType) -> bool:
    """
    Safely copy a file from src to dst while preserving metadata.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if operation succeeded, False otherwise
    """
    try:
        # Validate inputs
        if not src or not dst:
            raise ValueError(f'Both src and dst are required. Got src={src}, dst={dst}')
            
        src_path = Path(src) if not isinstance(src, Path) else src
        dst_path = Path(dst) if not isinstance(dst, Path) else dst
        
        # Check if source exists
        if not src_path.exists():
            raise FileNotFoundError(f'Source file does not exist: {src_path}')
            
        # Check if source is a file
        if not src_path.is_file():
            raise ValueError(f'Source is not a file: {src_path}')
            
        # Create parent directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dst_path)
        
        # Handle metadata
        try:
            inplace_overwrite_meta(src_path, dst_path)
        except Exception as meta_error:
            logger.debug(f'Failed to copy metadata from {src_path} to {dst_path}: {meta_error}')
            # The file copy succeeded even if metadata failed, so we don't return False here
            
        return True
        
    except Exception as e:
        logger.debug(f'Failed to copy file from {src} to {dst}: {e}', exc_info=True)
        return False

def inplace_overwrite_meta(src: PathType, target: PathType):
    # Get timestamps from the source file
    stat_src = os.stat(src)
    atime = stat_src.st_atime  # Access time
    mtime = stat_src.st_mtime  # Modification time

    # Apply timestamps to the destination file
    os.utime(target, (atime, mtime))


# Model related

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


# collections related
# key generation

def get_unique_key(key, d):
    """
    Generate a unique key for a dictionary by appending a numeric suffix.

    Args:
    key (str): The original key.
    d (dict): The dictionary in which the key needs to be unique.

    Returns:
    str: A unique key.
    """
    if key not in d:
        return key

    i = 1
    while f"{key}-{i}" in d:
        i += 1

    return f"{key}-{i}"

# String related

def replace_ing_words(text, filename='verb.json'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(dir_path, filename)
    """ Replace '-ing' words in the text with their original form using the verb dictionary. """
    with open(json_path, 'r') as file:
        verb_dict = json.load(file)

    return re.sub(r'\b(\w+ing)\b', lambda match: verb_dict.get(match.group(0), match.group(0)), text)
