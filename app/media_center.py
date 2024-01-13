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
from utils import MyPath, get_mode
from media_manager import MediaManager
from my_cluster import HierarchicalCluster, ClusterKeyProcessor, ClusterLeafProcessor
from my_cluster import Cluster

from functools import lru_cache

from function_tracker import global_tracker

from collections import namedtuple
CacheStates = namedtuple('CacheStates', ['raw', 'rotate', 'thumb', 'caption', 'nude'])

CAPTION_MIN_LENGTH = 10
CAPTION_MAX_LENGTH = 30

class MediaCenter:
    def __init__(self,
                 folder_path,
                 batch_size=8,
                 show_progress_bar=True,
                 check_rotation=True,
                 check_nude=True,
                 cache_flags=CacheStates(True,True,True,True,True),
                 **kwargs):
        print("Initializing ImageSimilarity...")

        self.similarity_model = ImageSimilarityCalculator()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments
        self.check_rotation = check_rotation
        self.check_nude = check_nude
        self.cache_flags = cache_flags

        self.folder_path = folder_path
        self.media_fps = MediaManager.get_all_valid_media(folder_path)
        self.video_mng = VideoManager(folder_path)
        self.cp = ImageCaptioner()

        self.mq = MediaQuestionare()
        self.mt = ImageTextMatcher()

        self.nt = NudeTagger()

        # Initialize cache managers
        ## level 1: raw material
        self.raw_thumbnail_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._compute_raw_thumbnail,
                                                    format_str="{base}_raw_thumbnail_{file_hash}.jpg")
        self.raw_embedding_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_raw_embedding,
                                                    format_str="{base}_raw_emb_{file_hash}.npy")

        ## level 2: rotation detect
        self.rotation_tag_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_rotation_tag,
                                                    format_str="{base}_rotation_{file_hash}.yml")

        ## level 3: thumbnail material
        self.thumbnail_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._compute_thumbnail,
                                                    format_str="{base}_thumbnail_{file_hash}.jpg")
        self.embedding_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_embedding,
                                                    format_str="{base}_emb_{file_hash}.npy")

        ## level 4a: nude detection
        self.nude_tag_cache_manager  = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_nude_tag,
                                                    format_str="{base}_nude_{file_hash}.yml")

        ## level 4b: caption detection
        self.caption_cache_manager   = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_caption,
                                                    format_str="{base}_caption_{file_hash}.txt")

        self._invalidate_cache()


        # Initialize similarity cluster
        self.hcluster = HierarchicalCluster(
                embedding_func = self.embedding_cache_manager.load,
                similarity_func = self.similarity_model.similarity_func,
                obj_to_name = self.path_to_folder_name)
        self._initialize()

    @property
    def media_size():
        return len(self.media_fps)

    def _invalidate_cache(self):
        for f in self.media_fps:
            if not self.cache_flags.raw:
                self.raw_embedding_cache_manager.clear(f)
                self.raw_thumbnail_cache_manager.clear(f)
            if not self.cache_flags.rotate:
                self.rotation_tag_cache_manager.clear(f)
            if not self.cache_flags.thumb:
                self.embedding_cache_manager.clear(f)
                self.thumbnail_cache_manager.clear(f)
            if not self.cache_flags.caption:
                self.caption_cache_manager.clear(f)
            if not self.cache_flags.nude:
                self.nude_tag_cache_manager.clear(f)

    @global_tracker
    def _compute_raw_thumbnail(self, image_path):
        def compute_thumbnail(path):
            if MediaManager.is_image(path):
                with Image.open(image_path) as img:
                    img.load()
                    MediaManager.as_720_thumbnail_inplace(img)
                    return img

            if MediaManager.is_video(path):
                return self.video_mng.extract_key_frame(path)

            return None

        return compute_thumbnail(image_path)

    @global_tracker
    def _compute_thumbnail(self, image_path):
        raw_img = self.raw_thumbnail_cache_manager.load(image_path)
        clockwise_degrees = self._get_media_rotation_clockwise_degree(image_path)
        rotated_img = MediaManager.rotate_image(raw_img, -clockwise_degrees)
        return rotated_img

    @global_tracker
    def _get_media_rotation_clockwise_degree(self, image_path):
        if not self.check_rotation: return 0
        return self.rotation_tag_cache_manager.load(image_path).get('rotate', 0)

    @global_tracker
    def _generate_nude_tag(self, image_path):
        # bad implementation but no choice because of 3rd party implementation
        thumb_path = self.thumbnail_cache_manager.to_cache_path(image_path)
        nude_tags = self.nt.detect(thumb_path)
        return nude_tags

    @global_tracker
    def _get_nude_tag(self, image_path):
        if not self.check_nude: return {}
        return self.nude_tag_cache_manager.load(image_path)

    @global_tracker
    def _generate_rotation_tag(self, image_path):
        img = self.raw_thumbnail_cache_manager.load(image_path)
        return self.mq.is_rotated(img)

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
            _ = self._get_nude_tag(fp)
            if self.check_rotation:
                _ = self.rotation_tag_cache_manager.load(fp)

    @global_tracker
    def _generate_raw_embedding(self, image_path):
        img = self.raw_thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    @global_tracker
    def _generate_embedding(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    @global_tracker
    def _generate_caption(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        return self.cp.caption(img, max_length=CAPTION_MAX_LENGTH, min_length=CAPTION_MIN_LENGTH)[0]

    @lru_cache(maxsize=64)
    def cluster(self, *distance_levels) -> Cluster:
        c_named = self.hcluster.cluster( {0: self.media_fps}, distance_levels,)
        c_named_formatted = ClusterKeyProcessor.name(c_named,
                self._generate_cluster_name_formatter)
        return c_named_formatted

    def path_to_folder_name(self, image_path):
        caption = self.caption_cache_manager.load(image_path)
        caption = caption.lower().replace('a', '').replace(' ' * 2, ' ')
        folder_name = '-'.join(x.title() for x in caption.split())
        return folder_name

    def cluster_to_thumbnail(self, cluster):
        return ClusterLeafProcessor.process(
            cluster,
            self.thumbnail_cache_manager.to_cache_path,
        )

    def _generate_cluster_name_formatter(self, file_paths):
        def labels(file_paths):
            lbls = set(
                tag_d['msg']
                for f in file_paths
                for tag_d in self._get_nude_tag(f).values()
                if tag_d['sensitive'])
            if len(lbls) <= 0:
                return ''
            elif len(lbls) <= 3:
                return f"[{'-'.join(sorted(lbls))}]"
            else:
                return 'NSFW-'

        def date(file_paths):
            ds = [MyPath(f).date for f in file_paths]
            return get_mode(ds)

        label = labels(file_paths)
        date = date(file_paths)
        return label + date + '-{key}'

    def copy_with_meta_rotate(self, src, dst):
        assert src in self.media_fps, f'src={src} provided are not maintained by MediaCenter (not in self.media_fps). Maybe it is a thumbnail?'
        clockwise_degrees = self._get_media_rotation_clockwise_degree(src)
        MediaManager.copy_with_meta_rotate(src, dst, clockwise_degrees)










