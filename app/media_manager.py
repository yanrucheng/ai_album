from PIL import Image
import cv2
from pillow_heif import register_heif_opener
register_heif_opener()
import os
from utils import PathType
import utils
import subprocess



class MediaManager(utils.Singleton):
    image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.heic', '.heif')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')

    def __init__(self):
        super().__init__()

    def rotate_image(img: Image.Image, clockwise_degree: int = 0):
        if clockwise_degree == 0:
            return img
        rotated_img = img.rotate(-clockwise_degree, expand=True)
        return rotated_img

    @classmethod
    def copy_with_meta_rotate(cls, src: PathType, dst: PathType, clockwise_degree: int = 0):
        assert clockwise_degree in (0, 90, 180, 270), 'clockwise_degree should be in multiple of 90'

        if clockwise_degree == 0:
            utils.copy_with_meta(src, dst)
            return

        if cls.is_image(src):
            if clockwise_degree:
                print(f'Warning: Image: {dst} rotation introduce slight quality loss. Rotation will not be performed.')
            utils.copy_with_meta(src, dst)

        if cls.is_video(src):
            if clockwise_degree:
                print(f'Warning: Video: {dst} are marked as clockwise {clockwise_degree} degree rotated. No actual re-encoding happens. Your video player might not respect this metadata settings.')
            cls.copy_video_with_ffmpeg_rotated(src, dst, clockwise_degree)
            utils.inplace_overwrite_meta(src, dst)
            return

        utils.copy_with_meta(src, dst)


    @staticmethod
    def copy_video_with_ffmpeg_rotated(input_file, output_file, degrees):
        '''
        This method set the metadata of the target video file.
        This flag is typically interpret as clockwise.
        But essentially it is determined by the video player.
        Some video player might ignore it.
        '''
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
        subprocess.run(command)

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

    @classmethod
    def get_all_valid_media(cls, folder_path):
        img_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if cls.is_image(f))
        vid_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if cls.is_video(f))
        fps = img_fps + vid_fps
        abs_fps = [ os.path.abspath(p) for p in fps]
        valid_fps = cls.validate_media(fps)
        return valid_fps


    @classmethod
    def is_image(cls, path):
        return path.lower().endswith(cls.image_exts)

    @classmethod
    def is_video(cls, path):
        return path.lower().endswith(cls.video_exts)

    @classmethod
    def validate_media(cls, paths):
        print('Validating all media...')
        valid_paths = []

        for path in paths:
            if os.path.split(path)[1].startswith("._"):
                continue

            try:
                if cls.is_image(path):
                    # Try opening an image file
                    with Image.open(path) as img:
                        img.verify()
                elif cls.is_video(path):
                    with utils.suppress_c_stdout_stderr():
                        # Try opening a video file
                        cap = cv2.VideoCapture(path)
                        if not cap.isOpened():
                            raise IOError("Cannot open video")
                        cap.release()
                else:
                    print(f"Unsupported file format: {path}")
                    continue

                valid_paths.append(path)
            except (IOError, SyntaxError) as e:
                print(f"Invalid media file detected: {path}. An error occurred: {e})")

        return valid_paths
