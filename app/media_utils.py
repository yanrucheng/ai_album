import os
import subprocess
from pathlib import Path
from PIL import Image
import cv2
from pillow_heif import register_heif_opener

register_heif_opener()

import utils  # Assumes utils provides Singleton, copy_with_meta, inplace_overwrite_meta, and suppress_c_stdout_stderr
from utils import PathType
import logging

logger = logging.getLogger(__name__)


class MediaValidator(utils.Singleton):
    """
    Provides validation routines for a single media file.
    Checks for supported media types and attempts to open and verify the file.
    """
    
    image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic', '.heif')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_image(path: str) -> bool:
        """Return True if the given path corresponds to a supported image file."""
        return path.lower().endswith(MediaValidator.image_exts)

    @staticmethod
    def is_video(path: str) -> bool:
        """Return True if the given path corresponds to a supported video file."""
        return path.lower().endswith(MediaValidator.video_exts)

    @classmethod
    def validate(cls, path: str) -> bool:
        """
        Validate a single media file.
        
        Rejects files that start with "._". For images, attempts to open and verify using PIL.
        For videos, uses OpenCV to attempt to open the file.
        """
        file_name = os.path.split(path)[1]
        if file_name.startswith("._"):
            return False

        try:
            if cls.is_image(path):
                with Image.open(path) as img:
                    img.verify()
                return True
            elif cls.is_video(path):
                with utils.suppress_c_stdout_stderr():
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        raise IOError("Cannot open video")
                    cap.release()
                return True
            else:
                logger.info(f"Unsupported file format: {path}")
        except (IOError, SyntaxError) as e:
            logger.error(f"Invalid media file: {path}. Error: {e}")
        return False


class MediaOperator(utils.Singleton):
    """
    Provides operations to transform or copy media files.
    
    These include rotating images and copying files with updated rotation metadata.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def rotate_image(img: Image.Image, clockwise_degree: int = 0) -> Image.Image:
        """
        Rotate an image by the given clockwise degree.
        
        If the degree is zero, the original image is returned.
        """
        if clockwise_degree == 0:
            return img
        # A negative degree rotates the image in a clockwise direction
        return img.rotate(-clockwise_degree, expand=True)

    @classmethod
    def copy_with_meta_rotate(cls, src: PathType, dst: PathType, clockwise_degree: int = 0):
        """
        Copy a media file from src to dst while handling rotation metadata.
        
        For images, a warning is issued if rotation is requested (due to potential quality loss)
        and the file is copied as-is. For videos, rotation metadata is applied without re-encoding.
        """
        assert clockwise_degree in (0, 90, 180, 270), "clockwise_degree must be a multiple of 90"

        if clockwise_degree == 0:
            utils.copy_with_meta(src, dst)
            return

        # Use the validator functions to decide how to proceed with the copy
        if MediaValidator.is_image(src):
            print(f"Warning: Image {dst} may lose quality when rotated. Copying without actual rotation.")
            utils.copy_with_meta(src, dst)
        elif MediaValidator.is_video(src):
            print(f"Warning: Video {dst} marked as rotated {clockwise_degree}°. No re-encoding is performed.")
            cls._copy_video_with_ffmpeg(src, dst, clockwise_degree)
            utils.inplace_overwrite_meta(src, dst)
        else:
            utils.copy_with_meta(src, dst)

    @staticmethod
    def _copy_video_with_ffmpeg(input_file: str, output_file: str, degrees: int):
        """
        Set video rotation metadata using ffmpeg without re-encoding.
        """
        metadata_value = {
            90: 'rotate=90',
            180: 'rotate=180',
            270: 'rotate=270'
        }.get(degrees, 'rotate=0')
        command = [
            'ffmpeg',
            '-i', input_file,
            '-c', 'copy',
            '-metadata:s:v', metadata_value,
            output_file
        ]
        subprocess.run(command, check=True)

    @staticmethod
    def as_720_thumbnail_inplace(img: Image.Image):
        # Determine the orientation and set the target size
        if img.width > img.height:
            # Horizontal image
            target_size = (1280, 720)
        else:
            # Vertical image
            target_size = (720, 1280)

        # Create the thumbnail. This maintains the aspect ratio.
        img.thumbnail(target_size)
        return img

