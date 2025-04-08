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
import myllm
from media_questionare import MediaQuestionare
from utils import MyPath, get_mode

from media_utils import MediaValidator, MediaOperator
from media_libs import MediaOrganizer

from my_cluster import LinearHierarchicalCluster
from my_cluster import ClusterKeyProcessor, ClusterLeafProcessor
from my_cluster import Cluster

from functools import lru_cache

from function_tracker import global_tracker
import utils
import re

import logging
logger = logging.getLogger(__name__)

from collections import namedtuple
CacheStates = namedtuple('CacheStates', ['raw', 'rotate', 'thumb', 'caption', 'nude'])

LANGUAGE_OPTIONS = ['en', 'zh'] # en for english, ch for chinese

class MediaCenter:
    def __init__(self,
                 folder_path,
                 batch_size=8,
                 show_progress_bar=True,
                 check_rotation=True,
                 check_nude=True,
                 cache_flags=CacheStates(True,True,True,True,True),
                 language='en',
                 **kwargs):
        print("Initializing ImageSimilarity...")

        self.similarity_model = myllm.ImageSimilarityCalculator()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments
        self.check_rotation = check_rotation
        self.check_nude = check_nude
        self.cache_flags = cache_flags
        self.language = language
        assert language in LANGUAGE_OPTIONS, f'only {LANGUAGE_OPTIONS} are supported. got: {language}'


        self.folder_path = folder_path
        self.mo = MediaOrganizer()
        self.media_fps = self.mo.get_all_valid_files(folder_path)
        self.video_mng = VideoManager(folder_path, show_progress_bar)
        self.cp = myllm.ImageCaptioner()
        self.tl = myllm.MyTranslator()

        self.mq = MediaQuestionare()
        self.mt = myllm.ImageTextMatcher()

        self.nt = myllm.NudeTagger()

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
        self.caption_en_cache_manager   = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_caption_en,
                                                    format_str="{base}_caption_en_{file_hash}.txt")
        self.caption_zh_cache_manager   = CacheManager(target_path=folder_path,
                                                    generate_func=self._generate_caption_zh,
                                                    format_str="{base}_caption_zh_{file_hash}.txt")

        self._invalidate_cache()


        # Initialize similarity cluster
        # self.time_cluster = AgglomerativeHierarchicalCluster(
        #         embedding_func = lambda p: np.array([MyPath(p).timestamp]),
        #         similarity_func = lambda a,b: abs(a - b),
        #         obj_to_name = lambda p: MyPath(p).date + MyPath(p).time_of_a_day)
        # self.image_cluster = AgglomerativeHierarchicalCluster(
        #         embedding_func = self.embedding_cache_manager.load,
        #         similarity_func = self.similarity_model.similarity_func,
        #         obj_to_name = self.path_to_folder_name)
        self.date_cluster = LinearHierarchicalCluster(
                embedding_func = lambda x: MyPath(x).date,
                similarity_func = lambda x,y: 0 if x != y else np.inf,
                sort_key_func = lambda x: MyPath(x).date,
                obj_to_name = lambda x: MyPath(x).date)
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
            if not self.cache_flags.rotate:
                self.rotation_tag_cache_manager.clear(f)
            if not self.cache_flags.thumb:
                self.embedding_cache_manager.clear(f)
                self.thumbnail_cache_manager.clear(f)
            if not self.cache_flags.caption:
                self.caption_en_cache_manager.clear(f)
                self.caption_zh_cache_manager.clear(f)
            if not self.cache_flags.nude:
                self.nude_tag_cache_manager.clear(f)

    @global_tracker
    def _compute_raw_thumbnail(self, image_path):
        def compute_thumbnail(path):
            if MediaValidator.is_image(path) and MediaValidator.validate(path):
                with Image.open(image_path) as img:
                    img.load()
                    MediaValidator.as_720_thumbnail_inplace(img)
                    return img

            if MediaValidator.is_video(path) and MediaValidator.validate(path):
                return self.video_mng.extract_key_frame(path)

            return None

        return compute_thumbnail(image_path)

    @global_tracker
    def _compute_thumbnail(self, image_path):
        raw_img = self.raw_thumbnail_cache_manager.load(image_path)
        clockwise_degrees = self._get_media_rotation_clockwise_degree(image_path)
        rotated_img = MediaOperator.rotate_image(raw_img, -clockwise_degrees)
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
        for fp in tqdm(self.media_fps, desc="Initializing embeddings", disable=not self.show_progress_bar):
            _ = self.embedding_cache_manager.load(fp)

    def compute_all_captions(self):
        print("Initializing captions...")
        for fp in tqdm(self.media_fps, desc="Initializing captions", disable=not self.show_progress_bar):
            _ = self._get_caption(fp)

    def compute_all_tags(self):
        print("Initializing tags...")
        for fp in tqdm(self.media_fps, desc="Initializing tags", disable=not self.show_progress_bar):
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

    def _get_caption(self, image_path):
        if self.language == 'en':
            return self.caption_en_cache_manager.load(image_path)
        if self.language == 'zh':
            return self.caption_zh_cache_manager.load(image_path)
        return 'langauge not supported'

    @global_tracker
    def _generate_caption_zh(self, image_path):
        caption_eng = self.caption_en_cache_manager.load(image_path)
        caption_eng_shorten = self.shorten_caption(caption_eng)
        caption = self.tl.translate(caption_eng_shorten)
        return caption

    @global_tracker
    def _generate_caption_en(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        return self.cp.caption(img, max_new_tokens=20)

    def cluster(self, *args) -> Cluster:
        self.full_cluster(*args)

    @lru_cache(maxsize=64)
    def bundle_cluster(self, *distance_levels) -> Cluster:
        # use time to cluster
        raw = {0: self.media_fps}

        # group by date
        c_date = self.date_cluster.cluster(raw, [1])

        # distance_levels = [] means if 1 photos are taken
        c_named = self.image_cluster.cluster( c_date, distance_levels,)

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
        caption = self._get_caption(image_path)
        caption_short = self.shorten_caption(caption)
        # remove all chinese/english punctuation
        caption = re.sub(r'[\u3000-\u303F\uff01-\uffee]|[^\w\s]', '', caption)

        folder_name = '-'.join(x.title() for x in caption.split())
        indexed_folder_name = '{idx}-' + folder_name
        return indexed_folder_name

    def shorten_caption(self, caption):
        caption = caption.replace(' and ', ' & ')
        caption = caption.replace('group of ', '').replace('couple of ', '').replace('pair of ', '')
        # stem verb
        caption = utils.replace_ing_words(caption)
        # remove quantifiers
        caption = re.sub(r'\b(a|an|some|is|are|be|do)\b', '', caption)
        # remove multiple space and leading trailing space
        caption = re.sub(' +', ' ', caption).strip()
        return caption

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
                return f"[{'-'.join(sorted(lbls))}]-"
            else:
                return 'NSFW-'

        def date(file_paths):
            ds = [MyPath(f).date for f in file_paths]
            return get_mode(ds)

        label = labels(file_paths)
        date = date(file_paths)
        return date + '-' + label + '{key}'

    def copy_with_meta_rotate(self, src, dst):
        assert src in self.media_fps, f'src={src} provided are not maintained by MediaCenter (not in self.media_fps). Maybe it is a thumbnail?'
        clockwise_degrees = self._get_media_rotation_clockwise_degree(src)
        MediaOperator.copy_with_meta_rotate(src, dst, clockwise_degrees)










