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

import functools

import utils
import re

import logging
logger = logging.getLogger(__name__)

from collections import namedtuple
CacheStates = namedtuple('CacheStates', ['raw', 'meta', 'rotate', 'thumb', 'caption', 'nude', 'title', 'location'])

class MediaCenter:

    def __init__(self,
                 folder_path,
                 batch_size=8,
                 show_progress_bar=True,
                 check_rotation=True,
                 cache_flags=CacheStates(True,True,True,True,True,True,True,True),
                 datum=my_metadata.MapDatum.WGS84,
                 max_gap_for_bundle=15,
                 skip_validate=False,
                 **kwargs):
        print("Initializing ImageSimilarity...")

        self.similarity_model = my_llm.ImageSimilarityCalculator()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments
        self.check_rotation = check_rotation
        self.cache_flags = cache_flags
        self.datum = datum
        self.max_gap_for_bundle = max_gap_for_bundle
        self.skip_validate = skip_validate

        self.folder_path = folder_path
        self.mo = MediaOrganizer(
                max_gap_for_bundle=max_gap_for_bundle,
                verbosity=utils.Verbosity.Once,
                skip_validate=self.skip_validate,
                )
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
        self.title_cache_manager     = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_title,
                                                    format_str="{base}_title_{file_hash}.txt")
        self.location_cache_manager  = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_location,
                                                    format_str="{base}_location_{file_hash}.txt")

        self._invalidate_cache()


        # Initialize similarity cluster
        self.date_cluster = LinearHierarchicalCluster(
                embedding_func = lambda x: MyPath(x).date,
                similarity_func = lambda x,y: 1 if x == y else -np.inf,
                sort_key_func = lambda x: MyPath(x).date,
                obj_to_name = lambda x: MyPath(x).date,
                merge_none_with_neighbor = True,
                )
        self.site_cluster = LinearHierarchicalCluster(
                embedding_func = self._get_gps,
                similarity_func = self._get_site_similarity,
                sort_key_func = lambda x: MyPath(x).timestamp,
                obj_to_name = self._get_location,
                needs_index = True,
                merge_adjacent_same_key = True,
                merge_none_with_neighbor = True,
                verbosity = utils.Verbosity.Full,
                )
        self.image_cluster = LinearHierarchicalCluster(
                embedding_func = lambda x:x,
                similarity_func = self._emb_sim_func,
                sort_key_func = lambda x: MyPath(x).timestamp,
                obj_to_name = self.path_to_folder_name,
                needs_index = True,
                )
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
            if not self.cache_flags.location:
                self.location_cache_manager.clear(f)

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

    def _get_site_similarity(self, lonlat_pair_a, lonlat_pair_b):
        if None in (lonlat_pair_a, lonlat_pair_b):
            return None
        d = utils.calculate_distance_meters(*lonlat_pair_a, *lonlat_pair_b)
        return -d

    @functools.lru_cache(maxsize=8192)
    def _get_gps(self, image_path):
        meta = self._get_metadata(image_path)
        gps = meta.get('gps', {})
        lat = gps.get('latitude_dec', None)
        lon = gps.get('longitude_dec', None)
        if None in (lat, lon):
            return None
        return lon, lat

    def _get_metadata(self, image_path):
        return self.meta_tag_cache_manager.load(image_path)

    def _generate_meta_tag(self, image_path):
        meta = my_metadata.PhotoMetadataExtractor.extract(image_path, datum=self.datum)
        return meta

    def _generate_rotation_tag(self, image_path):
        img = self.raw_thumbnail_cache_manager.load(image_path)
        return self.mq.is_rotated(img)

    def _initialize(self):
        print("Initializing embeddings...")
        for fp in tqdm(self.media_fps, desc="Initializing embeddings", disable=not self.show_progress_bar):
            _ = self.embedding_cache_manager.load(fp)

    @functools.lru_cache(maxsize=1024)
    def _get_embedding(self, image_path):
        return self.embedding_cache_manager.load(image_path)

    @functools.lru_cache(maxsize=8192)
    def _emb_sim_func(self, x, y):
        emb_x = self._get_embedding(x)
        emb_y = self._get_embedding(y)
        return self.similarity_model.similarity_func(emb_x, emb_y)

    def compute_all_cache(self):
        print("Computing all caches...")
        for fp in tqdm(self.media_fps, desc="Initializing tags", disable=not self.show_progress_bar):
            _ = self._get_metadata(fp)
            _ = self._get_nude_tag(fp)
            _ = self._get_caption(fp)
            _ = self._get_title(fp)
            _ = self._get_location(fp)
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

    @functools.lru_cache(maxsize=8192)
    def _get_location(self, image_path):
        return self.location_cache_manager.load(image_path)

    def _generate_location(self, image_path):
        has_mild_nude = self.has_mild_nude(image_path)
        metadata = self.meta_tag_cache_manager.load(image_path)
        _ = self.thumbnail_cache_manager.load(image_path)
        thumb_path = self.thumbnail_cache_manager._get_cache_file_path_from_path(image_path)
        location = self.llm_gen.get_location(thumb_path, has_nude=has_mild_nude, metadata=metadata)
        return location or 'Unknown'

    def cluster(self, *args) -> Cluster:
        self.full_cluster(*args)

    @functools.lru_cache(maxsize=64)
    def bundle_cluster(self, *distance_levels) -> Cluster:
        # use time to cluster
        raw = self.media_fps[:]

        # group by date
        logger.debug('Group by date started')
        c_date = self.date_cluster.cluster(raw, [1])
        # pprint.pprint(c_date)
        logger.debug('Group by date finished')

        # group by location
        logger.debug('Group by site started')
        c_geo = self.site_cluster.cluster(c_date, [500])
        # pprint.pprint(c_geo)
        logger.debug('Group by site finished')

        # distance_levels = [] means if 1 photos are taken
        logger.debug('Group by content started')
        c_named = self.image_cluster.cluster( c_geo, distance_levels,)
        # pprint.pprint(c_named)
        logger.debug('Group by content finished')

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

    @functools.lru_cache(maxsize=8192)
    def path_to_folder_name(self, image_path):

        def remove_punctuation(name):
            to_remove = '，。？'
            for c in to_remove:
                name = name.replace(c, '')
            return name

        # folder_name = '-'.join(x.title() for x in caption.split())
        folder_name = self._get_title(image_path)
        indexed_folder_name = folder_name
        cleaned_folder_name = remove_punctuation(indexed_folder_name)
        return cleaned_folder_name


    def copy_with_meta_rotate(self, src, dst):
        assert src in self.media_fps, f'src={src} provided are not maintained by MediaCenter (not in self.media_fps). Maybe it is a thumbnail?'
        clockwise_degrees = self._get_media_rotation_clockwise_degree(src)
        MediaOperator.copy_with_meta_rotate(src, dst, clockwise_degrees)










