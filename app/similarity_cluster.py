from typing import List, Dict, Callable, Any
from sklearn.cluster import AgglomerativeClustering

Cluster = Dict[str, 'Cluster'] # recursive typing

class HierarchicalCluster:

    def __init__(self,
                 data: List[Any],
                 embedding_func: List[Any],
                 similarity_func: Callable[[Any, Any], float],
                 caption_func: Callable[[Any], str] = None,
                 group_prefix_func: Callable[[List[Any]], str] = None
                ):
        self.data = data
        self.emb_func = embedding_func
        self.sim_func = similarity_func
        
        if caption_func is None:
            caption_func = lambda x: x if isinstance(x, str) else 'default_caption'
        self.caption_func = caption_func
        
        if group_prefix_func is None:
            group_prefix_func = lambda _: ''
        self.prefix_func = group_prefix_func

    @property
    def media_size(self) -> int:
        return len(self.data)

    def cluster(self, distance_levels=None) -> Cluster:
        """
        Cluster embeddings in a multi-level hierarchy using a list of distance thresholds.

        :param embeddings: The embeddings to cluster.
        :param distance_levels: A list of distance thresholds for each level of clustering.
        :return: Nested dictionary representing multi-level hierarchical clusters.
        """
        if self.media_size <= 0:
            return {}

        embeddings = [*map(self.emb_func, self.data)]

        if distance_levels is None:
            distance_levels = [2, 0.5]  # Default value

        # Pair each embedding with its corresponding file path
        paired_data = list(zip(self.data, embeddings))

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
        if self.media_size >= 2:
            result_clusters = recursive_clustering(initial_cluster, 0)[0]
        else:
            result_clusters = {0: self.data}

        named_clusters = self._cluster_naming(result_clusters)
        marked_clusters = self._cluster_marking(named_clusters)

        return marked_clusters

    def _cluster_marking(self, nested_dict):

        def recurse(d):
            if isinstance(d, dict):
                new_dict = {}
                res = set()
                for key, value in d.items():
                    d_, res_ = recurse(value)
                    prefix = self.prefix_func(res_)
                    new_key = prefix + key
                    res = res.union(res_)
                    new_dict[new_key] = d_
                return new_dict, res
            else:
                # For leaf nodes (list of file paths), just return them
                return d, set(d)

        d, _ = recurse(nested_dict)
        return d

    def _cluster_naming(self, clusters):
        def generate_folder_name(image_path):
            caption = self.caption_func(image_path)
            folder_name = '-'.join(x.title() for x in caption.split())
            p = MyPath(image_path)
            return f'{p.date}-{folder_name}'

        def calculate_similarity(item1, item2):
            emb1 = self.emb_func(item1)
            emb2 = self.emb_func(item2)
            return self.sim_func(emb1, emb2)

        def average_similarity(item, items):
            total_similarity = sum(calculate_similarity(item, other) for other in items if other != item)
            return total_similarity / (len(items) - 1) if len(items) > 1 else 0

        def select_best_representation(items):
            return max(items, key=lambda item: average_similarity(item, items))

        def process_dict_for_similarity(d):
            if isinstance(d, dict):
                new_dict = {}
                representations = list(d.keys()) + [img for sublist in d.values() if isinstance(sublist, list) for img in sublist]
                for _, value in d.items():
                    processed_value, best = process_dict_for_similarity(value)
                    new_dict[best] = processed_value

                return new_dict, select_best_representation([*new_dict.keys()])
            elif isinstance(d, list):
                return d[:], select_best_representation(d)
            else:
                return None, None

        def rename_to_captions(d):
            if isinstance(d, dict):
                return {generate_folder_name(key): rename_to_captions(value) for key, value in d.items()}
            return d

        processed_for_similarity, _ = process_dict_for_similarity(clusters)
        return rename_to_captions(processed_for_similarity)
