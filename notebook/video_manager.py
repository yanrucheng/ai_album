import os
import cv2
from PIL import Image

from utils import SingletonModelLoader
from cache_manager import CacheManager

INTERVAL = 10

def format_filename(file_path):
    """Truncate and format the file name to a maximum of 30 characters."""
    file_name = os.path.basename(file_path)
    return (file_name[:27] + '...') if len(file_name) > 30 else file_name

class VideoManager:
    def __init__(self, folder_path, model_name='OFA-Sys/chinese-clip-vit-huge-patch14'):
        self.folder_path = folder_path
        self.model_name = model_name.replace('/', '_')
        self.similarity_model = SingletonModelLoader.get_model(model_name)
        self.frame_cache_manager = CacheManager(cache_path_prefix=".similarity_cache/video/",
                                                root_path=folder_path,
                                                cache_tag="frames",
                                                generate_func=self._extract_and_cache_frames,
                                                format_str='{base}/thumbnail_{cache_tag}_*.jpg')
        self.emb_cache_manager = CacheManager(cache_path_prefix=".similarity_cache/video/",
                                              root_path=folder_path,
                                              cache_tag="emb",
                                              generate_func=self._generate_embeddings,
                                              format_str='{base}/sim_{cache_tag}_*.emb')

    def extract_key_frame(self, path):
        file_name = format_filename(path)
        print(f"Extracting frames from video '{file_name}'...")

        embeddings = self.emb_cache_manager.load(path)

        max_avg_similarity = 0
        key_frame = None
        pil_frames = self.frame_cache_manager.load(path)

        for i, emb_a in enumerate(embeddings):
            total_similarity = 0
            for j, emb_b in enumerate(embeddings):
                if i != j:
                    similarity = self.similarity_model.score_functions['cos_sim'](emb_a, emb_b)
                    total_similarity += similarity

            avg_similarity = total_similarity / (len(embeddings) - 1)
            if avg_similarity > max_avg_similarity:
                max_avg_similarity = avg_similarity
                key_frame = pil_frames[i]

        return key_frame

    def extract_frames(self, video_path):
        return self.frame_cache_manager.load(video_path)

    def _generate_embeddings(self, video_path):
        frames = self.extract_frames(video_path)
        batch_embeddings = self.similarity_model.get_embeddings(frames, show_progress_bar=True, batch_size=8)
        return batch_embeddings

    def _extract_and_cache_frames(self, video_path):
        cache_dir = self._create_cache_directory(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * INTERVAL)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0 or frame_count == total_frames - 1:
                frame_path = os.path.join(cache_dir, f"frame_{frame_count}.jpg")
                pil_image = self._cv_frame_to_pil_image(frame)
                pil_image.thumbnail((1280, 720))
                yield pil_image

            frame_count += 1

        cap.release()

    def _create_cache_directory(self, video_path):
        cache_dir = os.path.splitext(os.path.basename(video_path))[0]
        cache_dir = f".similarity_cache/video/{cache_dir}_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_dir

    @staticmethod
    def _cv_frame_to_pil_image(frame):
        cv2_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image_rgb)
        return pil_image

    