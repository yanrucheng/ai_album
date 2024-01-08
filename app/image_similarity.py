import os
import fnmatch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from tqdm import tqdm

from cache_manager import CacheManager
from video_manager import VideoManager
from myllm import ImageSimilarityCalculator, ImageCaptioner, NudeTagger, VQA, ImageTextMatcher
from media_questionare import MediaQuestionare
from utils import MyPath
from similarity_cluster import HierarchicalCluster, Cluster

CAPTION_MIN_LENGTH = 10
CAPTION_MAX_LENGTH = 30

class ImageSimilarity:
    def __init__(self,
                 folder_path,
                 questionare_on=False,
                 batch_size=8,
                 show_progress_bar=True,
                 **kwargs):
        print("Initializing ImageSimilarity...")

        self.similarity_model = ImageSimilarityCalculator()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments

        self.questionare_on = questionare_on
        self.folder_path = folder_path
        self.media_fps = self._load_image_paths(folder_path)
        self.video_mng = VideoManager(folder_path)
        self.cp = ImageCaptioner()

        self.mq = MediaQuestionare()
        self.mt = ImageTextMatcher()

        self.nt = NudeTagger()

        # Initialize cache managers
        self.thumbnail_cache_manager = CacheManager(target_path=folder_path,
                                                    cache_tag="thumbnail",
                                                    generate_func=self._compute_and_save_thumbnail,
                                                    format_str="{base}_thumbnail_{md5}.jpg")
        self.embedding_cache_manager = CacheManager(target_path=folder_path,
                                                    cache_tag="emb",
                                                    generate_func=self._generate_embedding,
                                                    format_str="{base}_{md5}.emb")
        self.caption_cache_manager   = CacheManager(target_path=folder_path,
                                                    cache_tag="caption",
                                                    generate_func=self._generate_caption,
                                                    format_str="{base}_{md5}_caption.txt")
        self.tag_cache_manager       = CacheManager(target_path=folder_path,
                                                    cache_tag="tag",
                                                    generate_func=self._generate_tags,
                                                    format_str="{base}_{md5}_tag.yml")


        # Initialize similarity cluster
        self.hcluster = HierarchicalCluster(data = self.media_fps,
                                            embedding_func = self.embedding_cache_manager.load,
                                            similarity_func = self.similarity_model.similarity_func,
                                            caption_func = self.caption_cache_manager.load, # optional
                                            group_prefix_func = self._generate_cluster_folder_prefix # optional
                                           )
        
        print("Loaded similarity model and image file paths.")
        self._initialize()
        print("Initialization complete.")

    def _load_image_paths(self, folder_path):
        img_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if self._is_image(f))
        vid_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if self._is_video(f))

        return img_fps + vid_fps

    @property
    def media_size():
        return len(self.media_fps)

    def _is_image(self, path):
        file_extensions = ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.heif']
        return any(fnmatch.fnmatch(path.lower(), ext) for ext in file_extensions)

    def _is_video(self, path):
        file_extensions = ['*.mp4', '*.avi', '*.webm', '*.mkv', '*.mov']
        return any(fnmatch.fnmatch(path.lower(), ext) for ext in file_extensions)

    def _compute_and_save_thumbnail(self, image_path):
        def compute_thumbnail(path):
            if self._is_image(path):
                with Image.open(image_path) as img:
                    img.load()
                    img.thumbnail((1280, 720))
                    return img

            if self._is_video(path):
                return self.video_mng.extract_key_frame(path)

            return None

        return compute_thumbnail(image_path)

    def _generate_tags(self, image_path):

        d = {}

        if self.questionare_on:
            # get questionare
            img = self.thumbnail_cache_manager.load(image_path)
            d = self.mq.process_image(img)

            # add confidence
            # d['vertical_angle_confidence'] = self.mt.text_match(img, f"In terms of filming angle, this picture is in a {d['vertival_angle']} view")
            # d['horizontal_angle_confidence'] = self.mt.text_match(img, f"In terms of filming angle, this picture is in a {d['horizontal_angle']} view")
            # d['sex_confidence'] = self.mt.text_match(img, d['sex'])

        # add caption
        d['caption'] = self.caption_cache_manager.load(image_path)

        # add nude tags
        # bad implementation but no choice because of 3rd party implementation
        thumb_path = self.thumbnail_cache_manager._get_cache_file_path(image_path)
        nude_tags = self.nt.detect(thumb_path)
        d['nude_tag'] = nude_tags
        return d

    def _initialize(self):
        print("Initializing embeddings...")
        for fp in tqdm(self.media_fps, desc="Initializing embeddings"):
            _ = self.embedding_cache_manager.load(fp)

    def compute_all_captions(self):
        print("Initializing captions...")
        for fp in tqdm(self.media_fps, desc="Initializing captions"):
            _ = self.caption_cache_manager.load(fp)

    def compute_all_tags(self):
        print("Initializing tags...")
        for fp in tqdm(self.media_fps, desc="Initializing tags"):
            _ = self.tag_cache_manager.load(fp)

    def _generate_embedding(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    def _generate_caption(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        return self.cp.caption(img, max_length=CAPTION_MAX_LENGTH, min_length=CAPTION_MIN_LENGTH)[0]

    def cluster(self, distance_levels) -> Cluster:
        return self.hcluster.cluster(distance_levels)

    def _generate_cluster_folder_prefix(self, file_paths):
        lbls = set(tag_d['msg'] for f in file_paths for tag_d in self.tag_func(f)['nude_tag'].values() if tag_d['sensitive'])
        if len(lbls) <= 0:
            return ''
        elif len(lbls) <= 3:
            return f"[{'-'.join(sorted(lbls))}]"
        else:
            return 'ITMC-'