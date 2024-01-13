import os
import shutil
import hashlib
import functools
from functools import lru_cache
from datetime import datetime
from collections import Counter
import hashlib
from pathlib import Path

import sys
import contextlib

import json
import re


# Print related

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
    def timestamp(self):
        try:
            stat = os.stat(self.path)
            # order of birthtime, create time, modification time
            target_ts = next(
                (ts
                 for ts in [getattr(stat, 'st_birthtime', None), stat.st_ctime, stat.st_mtime] if ts is not None),
                None
            )
            return target_ts

        except OSError as error:
            print(f"Error getting dates for {self.path}: {error}")
            return None

    @property
    def date(self):
        return datetime.fromtimestamp(self.timestamp).strftime('%y%m%d')

    @property
    def time_of_a_day(self):
        """
        Return a human-readable time of day for a given timestamp.
        """
        timestamp = datetime.fromtimestamp(self.timestamp)
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

def copy_with_meta(src: PathType, dst: PathType):
    assert src and dst, f'Both src, dst are a must. got src={src}; dst={dst}'

    # Copy the file
    shutil.copy2(src, dst)
    inplace_overwrite_meta(src, dst)

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

# String 

def replace_ing_words(text, filename='verb.json'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(dir_path, filename)
    """ Replace '-ing' words in the text with their original form using the verb dictionary. """
    with open(json_path, 'r') as file:
        verb_dict = json.load(file)

    return re.sub(r'\b(\w+ing)\b', lambda match: verb_dict.get(match.group(0), match.group(0)), text)

def remove_quantifier(text):
    return re.sub(r'\b(a|an|some|is|are)\b', '', text)

