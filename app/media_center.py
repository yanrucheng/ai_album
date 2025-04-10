import os
import fnmatch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import pprint

from tqdm import tqdm

from cache_manager import CacheManager
from video_manager import VideoManager
import my_llm
from media_questionare import MediaQuestionare
from utils import MyPath, get_mode

from media_utils import MediaValidator, MediaOperator
from media_libs import MediaOrganizer

from my_cluster import LinearHierarchicalCluster
from my_cluster import ClusterKeyProcessor, ClusterLeafProcessor
from my_cluster import Cluster
import my_metadata

from functools import lru_cache

import utils
import re

import logging
logger = logging.getLogger(__name__)

from collections import namedtuple
CacheStates = namedtuple('CacheStates', ['raw', 'meta', 'rotate', 'thumb', 'caption', 'nude', 'title'])

class MediaCenter:

    def __init__(self,
                 folder_path,
                 batch_size=8,
                 show_progress_bar=True,
                 check_rotation=True,
                 cache_flags=CacheStates(True,True,True,True,True,True,True),
                 **kwargs):
        print("Initializing ImageSimilarity...")

        self.similarity_model = my_llm.ImageSimilarityCalculator()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments
        self.check_rotation = check_rotation
        self.cache_flags = cache_flags


        self.folder_path = folder_path
        self.mo = MediaOrganizer()
        self.media_fps = self.mo.get_all_valid_files(folder_path)
        self.video_mng = VideoManager(folder_path, show_progress_bar)

        self.llm_gen = my_llm.ImageLLMGen()

        self.mq = MediaQuestionare()
        self.mt = my_llm.ImageTextMatcher()

        self.nt = my_llm.NudeTagger()

        # Initialize cache managers
        ## level 1: raw material
        self.raw_thumbnail_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._compute_raw_thumbnail,
                                                    format_str="{base}_raw_thumbnail_{file_hash}.jpg")
        self.raw_embedding_cache_manager = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_raw_embedding,
                                                    format_str="{base}_raw_emb_{file_hash}.npy")

        ## level 2: metadata extraction
        self.meta_tag_cache_manager =  CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_meta_tag,
                                                    format_str="{base}_meta_{file_hash}.yml")

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

        ## level 5: title detection
        self.title_cache_manager     = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_title,
                                                    format_str="{base}_title_{file_hash}.txt")

        self._invalidate_cache()


        # Initialize similarity cluster
        self.date_cluster = LinearHierarchicalCluster(
                embedding_func = lambda x: MyPath(x).date,
                similarity_func = lambda x,y: 0 if x != y else np.inf,
                sort_key_func = lambda x: MyPath(x).date,
                obj_to_name = lambda x: MyPath(x).date)
        self.geo_cluster = LinearHierarchicalCluster(
                embedding_func = self._get_gps,
                similarity_func = self._get_gps_similarity,
                sort_key_func = lambda x: MyPath(x).timestamp,
                obj_to_name = self._get_gps_name,
                needs_prune = False,
                )
        self.image_cluster = LinearHierarchicalCluster(
                embedding_func = self.embedding_cache_manager.load,
                similarity_func = self.similarity_model.similarity_func,
                sort_key_func = lambda x: MyPath(x).timestamp,
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
            if not self.cache_flags.meta:
                self.meta_tag_cache_manager.clear(f)
            if not self.cache_flags.rotate:
                self.rotation_tag_cache_manager.clear(f)
            if not self.cache_flags.thumb:
                self.embedding_cache_manager.clear(f)
                self.thumbnail_cache_manager.clear(f)
            if not self.cache_flags.caption:
                self.caption_cache_manager.clear(f)
            if not self.cache_flags.nude:
                self.nude_tag_cache_manager.clear(f)
            if not self.cache_flags.title:
                self.title_cache_manager.clear(f)

    def _compute_raw_thumbnail(self, image_path):
        def compute_thumbnail(path):
            if MediaValidator.is_image(path) and MediaValidator.validate(path):
                with Image.open(image_path) as img:
                    img.load()
                    MediaOperator.as_720_thumbnail_inplace(img)
                    return img

            if MediaValidator.is_video(path) and MediaValidator.validate(path):
                return self.video_mng.extract_key_frame(path)

            return None

        return compute_thumbnail(image_path)

    def _compute_thumbnail(self, image_path):
        raw_img = self.raw_thumbnail_cache_manager.load(image_path)
        clockwise_degrees = self._get_media_rotation_clockwise_degree(image_path)
        rotated_img = MediaOperator.rotate_image(raw_img, -clockwise_degrees)
        return rotated_img

    def _get_media_rotation_clockwise_degree(self, image_path):
        if not self.check_rotation: return 0

        photo_meta = self.meta_tag_cache_manager.load(image_path).get('photo', {})
        if photo_meta:
            rotate = photo_meta.get('rotate', 0)
            return rotate

        return self.rotation_tag_cache_manager.load(image_path).get('rotate', 0)

    def _generate_nude_tag(self, image_path):
        # bad implementation but no choice because of 3rd party implementation
        thumb_path = self.thumbnail_cache_manager.to_cache_path(image_path)
        nude_tags = self.nt.detect(thumb_path)
        return nude_tags

    def _get_nude_tag(self, image_path):
        return self.nude_tag_cache_manager.load(image_path)

    def has_nude(self, image_path):
        nude_tag = self.nude_tag_cache_manager.load(image_path)
        if not nude_tag: return False
        return any(d['sensitive'] for lb, d in nude_tag.items())

    def has_mild_nude(self, image_path):
        nude_tag = self.nude_tag_cache_manager.load(image_path)
        if not nude_tag: return False
        return any(d['mild_sensitive'] for lb, d in nude_tag.items())

    def _get_gps_similarity(self, latlon_pair_a, latlon_pair_b):
        if None in (*latlon_pair_a, *latlon_pair_b):
            return np.inf
        d = utils.calculate_distance_meters(*latlon_pair_a, *latlon_pair_b)
        return 1 / d

    def _get_gps_name(self, image_path):
        meta = self._get_metadata(image_path)
        gps_resolved = meta.get('gps_resolved', [])
        tourism_locs = sorted((d.get('distance'), d.get('address',{}).get('name'))
                              for d in gps_resolved if d.get('class') == 'tourism')
        if tourism_locs:
            return tourism_locs[0][1]

        other_locs = sorted((d.get('distance'), d.get('address',{}).get('name'))
                            for d in gps_resolved)
        if other_locs:
            return other_locs[0][1]

        lat, lon = self._get_gps(image_path)
        loc = f'{lat:.6f}-{lon:.6f}'
        return loc

    def _get_gps(self, image_path):
        meta = self._get_metadata(image_path)
        gps = meta.get('gps', {})
        lat = gps.get('latitude_dec', None)
        lon = gps.get('longitude_dec', None)
        return lat, lon

    def _get_metadata(self, image_path):
        return self.meta_tag_cache_manager.load(image_path)

    def _generate_meta_tag(self, image_path):
        meta = my_metadata.PhotoMetadataExtractor.extract(image_path)
        return meta

    def _generate_rotation_tag(self, image_path):
        img = self.raw_thumbnail_cache_manager.load(image_path)
        return self.mq.is_rotated(img)

    def _initialize(self):
        print("Initializing embeddings...")
        for fp in tqdm(self.media_fps, desc="Initializing embeddings", disable=not self.show_progress_bar):
            _ = self.embedding_cache_manager.load(fp)

    def compute_all_cache(self):
        print("Computing all caches...")
        for fp in tqdm(self.media_fps, desc="Initializing tags", disable=not self.show_progress_bar):
            _ = self._get_metadata(fp)
            _ = self._get_nude_tag(fp)
            _ = self._get_caption(fp)
            _ = self._get_title(fp)
            if self.check_rotation:
                # if xmp contains rotation info then no need to compute
                _ = self._get_media_rotation_clockwise_degree(fp)

    def _generate_raw_embedding(self, image_path):
        img = self.raw_thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    def _generate_embedding(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    def _get_caption(self, image_path):
        return self.caption_cache_manager.load(image_path)

    def _generate_caption(self, image_path):
        has_mild_nude = self.has_mild_nude(image_path)
        metadata = self.meta_tag_cache_manager.load(image_path)
        _ = self.thumbnail_cache_manager.load(image_path)
        thumb_path = self.thumbnail_cache_manager._get_cache_file_path_from_path(image_path)
        caption = self.llm_gen.get_caption(thumb_path, has_nude=has_mild_nude, metadata=metadata)
        return caption

    def _get_title(self, image_path):
        return self.title_cache_manager.load(image_path)

    def _generate_title(self, image_path):
        has_mild_nude = self.has_mild_nude(image_path)
        metadata = self.meta_tag_cache_manager.load(image_path)
        _ = self.thumbnail_cache_manager.load(image_path)
        thumb_path = self.thumbnail_cache_manager._get_cache_file_path_from_path(image_path)
        title = self.llm_gen.get_title(thumb_path, has_nude=has_mild_nude, metadata=metadata)
        return title

    def cluster(self, *args) -> Cluster:
        self.full_cluster(*args)

    @lru_cache(maxsize=64)
    def bundle_cluster(self, *distance_levels) -> Cluster:
        # use time to cluster
        raw = {0: self.media_fps}

        # group by date
        c_date = self.date_cluster.cluster(raw, [1])

        # group by location
        c_geo = self.geo_cluster.cluster(raw, [1/500])

        # distance_levels = [] means if 1 photos are taken
        c_named = self.image_cluster.cluster( c_geo, distance_levels,)

        return c_named

    def full_cluster(self, *args) -> Cluster:
        c_full = ClusterLeafProcessor.process(
            self.bundle_cluster(*args),
            lambda x: self.mo.get_bundle(x).files,
        )
        return c_full

    def thumbnail_cluster(self, *args):

        def get_thumbnail_path(path) -> str:
            if not MediaValidator.validate(path):
                return None
            return self.thumbnail_cache_manager.to_cache_path(path)

        return ClusterLeafProcessor.process(
            self.bundle_cluster(*args),
            get_thumbnail_path
        )

    def path_to_folder_name(self, image_path):

        def remove_punctuation(name):
            to_remove = '，。？'
            for c in to_remove:
                name = name.replace(c, '')
            return name

        # folder_name = '-'.join(x.title() for x in caption.split())
        folder_name = self._get_title(image_path)
        indexed_folder_name = '{idx}-' + folder_name
        cleaned_folder_name = remove_punctuation(indexed_folder_name)
        return cleaned_folder_name


    def copy_with_meta_rotate(self, src, dst):
        assert src in self.media_fps, f'src={src} provided are not maintained by MediaCenter (not in self.media_fps). Maybe it is a thumbnail?'
        clockwise_degrees = self._get_media_rotation_clockwise_degree(src)
        MediaOperator.copy_with_meta_rotate(src, dst, clockwise_degrees)










