import os
import cv2
from PIL import Image

from cache_manager import CacheManager
from myllm import ImageSimilarityCalculator

INTERVAL = 3
TOP_K_KEY_FRAME_SELECTION = 0.5

def format_filename(file_path):
    """Truncate and format the file name to a maximum of 30 characters."""
    file_name = os.path.basename(file_path)
    return (file_name[:27] + '...') if len(file_name) > 30 else file_name

class VideoManager:
    def __init__(self, folder_path, model_name=''):
        self.folder_path = folder_path
        self.model_name = model_name.replace('/', '_')
        self.similarity_model = ImageSimilarityCalculator()
        self.frame_cache_manager = CacheManager(target_path=folder_path,
                                                cache_tag="frames",
                                                generate_func=self._extract_and_cache_frames,
                                                format_str='{base}_{md5}/{base}_thumbnail_*.jpg')
        self.emb_cache_manager   = CacheManager(target_path=folder_path,
                                                cache_tag="emb",
                                                generate_func=self._generate_embeddings,
                                                format_str='{base}_{md5}/{base}_*.emb')

    def extract_key_frame(self, path, top_k=TOP_K_KEY_FRAME_SELECTION):

        assert isinstance(top_k, int) or isinstance(top_k, float), 'top_k should be int or float'
        
        file_name = format_filename(path)
        print(f"Extracting frames from video '{file_name}'...")
    
        embeddings = self.emb_cache_manager.load(path)
        pil_frames = self.frame_cache_manager.load(path)
        
        if isinstance(top_k, float):
            top_k = max(int(round(len(pil_frames) * top_k)), 2) # at least 3 samples should be considered
    
        max_avg_similarity = 0
        key_frame = pil_frames[0] # use the first one as the default result
    
        for i, emb_a in enumerate(embeddings):
            similarities = [self.similarity_model.similarity_func(emb_a, emb_b) for j, emb_b in enumerate(embeddings) if i != j]
            top_k_similarities = sorted(similarities, reverse=True)[:top_k]
            if top_k_similarities:
                avg_similarity = sum(top_k_similarities) / len(top_k_similarities)
            else:
                avg_similarity = 0
                
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

    