import os
import fnmatch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

from utils import SingletonModelLoader

from cache_manager import CacheManager
from video_manager import VideoManager
from image_text_model import ImageQuestionAnswerer


class ImageSimilarity:
    def __init__(self, folder_path, model_name='OFA-Sys/chinese-clip-vit-huge-patch14', batch_size=8, show_progress_bar=True, **kwargs):
        print("Initializing ImageSimilarity...")

        self.model_name = model_name.replace('/', '_')  # Replace '/' in model name for file paths
        self.similarity_model = SingletonModelLoader.get_model(model_name)
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs  # Store any additional keyword arguments

        self.folder_path = folder_path
        self.media_fps = self._load_image_paths(folder_path)
        self.video_mng = VideoManager(folder_path)
        self.qa = ImageQuestionAnswerer("blip_caption", "large_coco")

        # Initialize CacheManagers
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

        print("Loaded similarity model and image file paths.")
        self._initialize()
        self.similarity_cache = self._cache_similarities()
        print("Initialization complete.")

    def _load_image_paths(self, folder_path):
        img_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if self._is_image(f))
        vid_fps = sorted(os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if self._is_video(f))

        return img_fps + vid_fps

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
                    img.thumbnail((1280, 720))
                    return img

            if self._is_video(path):
                return self.video_mng.extract_key_frame(path)

            return None

        return compute_thumbnail(image_path)
        
    def _initialize(self):
        print("Initializing embeddings...")
        for fp in tqdm(self.media_fps, desc="Initializing embeddings"):
            _ = self.embedding_cache_manager.load(fp)
    
        print("Initializing captions...")
        for fp in tqdm(self.media_fps, desc="Initializing captions"):
            _ = self.caption_cache_manager.load(fp)

    def _generate_embedding(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        emb = self.similarity_model.get_embeddings([img])[0]  # Extract the first (and only) embedding
        return emb

    def _generate_caption(self, image_path):
        img = self.thumbnail_cache_manager.load(image_path)
        return self.qa.caption(img, max_length=90, min_length=30)[0]
    
    def _cache_similarities(self):
        cache = {}
        batch_size = 100

        print("Caching similarities...")
        pbar = tqdm(total=len(self.media_fps), desc="Caching similarities")
        for i in range(0, len(self.media_fps), batch_size):
            end_idx = min(i + batch_size, len(self.media_fps))
            batch_fps = self.media_fps[i:end_idx]

            # Load embeddings for the batch
            embeddings = [self.embedding_cache_manager.load(fp) for fp in batch_fps]

            # Compute similarity matrix for the batch
            for j in range(len(batch_fps)):
                for k in range(j + 1, len(batch_fps)):  # Avoid duplicate computations
                    sim_score = self.similarity_model.score_functions['cos_sim'](embeddings[j], embeddings[k])
                    cache[(i + j, i + k)] = sim_score
                    cache[(i + k, i + j)] = sim_score

            pbar.update(min(batch_size, len(self.media_fps) - i))

        pbar.close()
        return cache
        
    def get_similarity_with_file_path(self, file_path_a, file_path_b):
        if file_path_a in self.media_fps and file_path_b in self.media_fps:
            idx_a = self.media_fps.index(file_path_a)
            idx_b = self.media_fps.index(file_path_b)
            return self.similarity_cache.get((idx_a, idx_b), None)
        else:
            return None  # File path not found

    def cluster_images_with_multilevel_hierarchical(self, distance_levels=None):
        """
        Cluster embeddings in a multi-level hierarchy using a list of distance thresholds.

        :param embeddings: The embeddings to cluster.
        :param distance_levels: A list of distance thresholds for each level of clustering.
        :return: Nested dictionary representing multi-level hierarchical clusters.
        """
        embeddings = [self.embedding_cache_manager.load(fp) for fp in self.media_fps]
        
        if distance_levels is None:
            distance_levels = [0.05]  # Default value

        # Pair each embedding with its corresponding file path
        paired_data = list(zip(self.media_fps, embeddings))

        # Starting with all paired data as the initial cluster
        initial_cluster = {0: paired_data}

        # Function to recursively apply clustering
        def recursive_clustering(current_clusters, level):
            if level >= len(distance_levels):
                # At the final level, return the file paths instead of (file path, embedding) pairs
                return {cluster_id: [fp for fp, _ in cluster_data] for cluster_id, cluster_data in current_clusters.items()}

            new_clusters = {}
            for cluster_id, cluster_data in current_clusters.items():
                if len(cluster_data) > 1:
                    # Extract embeddings for clustering
                    cluster_embeddings = [emb for _, emb in cluster_data]
                    clustering = AgglomerativeClustering(distance_threshold=distance_levels[level], n_clusters=None)
                    clustering.fit(cluster_embeddings)

                    sub_clusters = {}
                    for idx, label in enumerate(clustering.labels_):
                        sub_clusters.setdefault(label, []).append(cluster_data[idx])

                    new_clusters[cluster_id] = recursive_clustering(sub_clusters, level + 1)
                else:
                    # If only one item in cluster, no need for further clustering
                    new_clusters[cluster_id] = cluster_data

            return new_clusters

        # Apply recursive clustering starting from level 0
        final_clusters = recursive_clustering(initial_cluster, 0)
        return final_clusters
            

    def cluster_images_with_hierarchical(self, embeddings, distance_threshold=0.05):
        '''The best. distance_threshold = 0.5 for detailed cluster. distance_threshold = 2 for coarse cluster'''
        # Convert list of embeddings to a numpy array
        embeddings_array = np.array(embeddings)

        # Apply Hierarchical Clustering
        hierarchical_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward')
        hierarchical_cluster.fit(embeddings_array)

        # Extract cluster assignments
        labels = hierarchical_cluster.labels_

        # Group file paths by cluster labels
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(self.media_fps[idx])

        return clusters