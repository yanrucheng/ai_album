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

CAPTION_MIN_LENGTH = 10
CAPTION_MAX_LENGTH = 30

class MediaSimilarity:
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
        self.media_fps = MediaManager.get_all_valid_media(folder_path)
        self.video_mng = VideoManager(folder_path)
        self.cp = ImageCaptioner()

        self.mq = MediaQuestionare()
        self.mt = ImageTextMatcher()

        self.nt = NudeTagger()

        # Initialize cache managers
        self.thumbnail_cache_manager = CacheManager(target_path=folder_path,
                                                    cache_tag="thumbnail",
                                                    generate_func=self._compute_and_save_thumbnail,
                                                    format_str="{base}_thumbnail_{file_hash}.jpg")
        self.embedding_cache_manager = CacheManager(target_path=folder_path,
                                                    cache_tag="emb",
                                                    generate_func=self._generate_embedding,
                                                    format_str="{base}_{file_hash}_emb.npy")
        self.caption_cache_manager   = CacheManager(target_path=folder_path,
                                                    cache_tag="caption",
                                                    generate_func=self._generate_caption,
                                                    format_str="{base}_{md5}_caption.txt")
        self.tag_cache_manager       = CacheManager(target_path=folder_path,
                                                    cache_tag="tag",
                                                    generate_func=self._generate_tags,
                                                    format_str="{base}_{md5}_tag.yml")


        # Initialize similarity cluster
        self.hcluster =   HierarchicalCluster(data = self.media_fps,
                                              embedding_func = self.embedding_cache_manager.load,
                                              similarity_func = self.similarity_model.similarity_func,
                                              obj_to_name = self.caption_cache_manager.load)
        self.ckp =     ClusterKeyProcessor(objs_to_cluster_prefix = self._generate_cluster_folder_prefix)
        self.clp = ClusterLeafProcessor(obj_to_obj = self.thumbnail_cache_manager.to_cache_path)

        self._initialize()

    @property
    def media_size():
        return len(self.media_fps)

    def _compute_and_save_thumbnail(self, image_path):
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
        thumb_path = self.thumbnail_cache_manager.to_cache_path(image_path)
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

    @lru_cache(maxsize=64)
    def cluster(self, *distance_levels) -> Cluster:
        c = self.hcluster.cluster(distance_levels)
        c_named = self.ckp.name_cluster(c)
        return c_named

    def cluster_to_thumbnail(self, cluster):
        return self.clp.process_cluster(cluster)

    def _generate_cluster_folder_prefix(self, file_paths):
        def labels(file_paths):
            lbls = set(
                tag_d['msg']
                for f in file_paths
                for tag_d in self.tag_cache_manager.load(f)['nude_tag'].values()
                if tag_d['sensitive'])
            if len(lbls) <= 0:
                return ''
            elif len(lbls) <= 3:
                return f"[{'-'.join(sorted(lbls))}]"
            else:
                return 'ITMC-'

        def date(file_paths):
            ds = [MyPath(f).date for f in file_paths]
            return get_mode(ds)

        label = labels(file_paths)
        date = date(file_paths)
        return label + date + '-'

