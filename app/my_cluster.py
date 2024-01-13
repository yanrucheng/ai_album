from typing import List, Dict, Callable, Any
from sklearn.cluster import AgglomerativeClustering
from utils import MyPath
import os
import utils

Cluster = Dict[str, 'Cluster'] # recursive typing

Path = str
def copy_file_as_cluster(cluster: Cluster,
                         target_path: Path,
                         operator: Callable[[str], str] = utils.copy_with_meta,
                         ):

    def _copy_files_to_clusters(cluster, target_path, _current_path=""):

        for cluster_id, contents in cluster.items():
            # Create a new subdirectory for the current cluster
            cluster_dir = os.path.join(target_path, _current_path, str(cluster_id))
            os.makedirs(cluster_dir, exist_ok=True)

            if isinstance(contents, dict):
                # If the contents are a dictionary, recurse into it
                _copy_files_to_clusters(contents, target_path, os.path.join(_current_path, str(cluster_id)))
            elif isinstance(contents, list):
                # If the contents are a list, copy the files into the current cluster directory
                for file_path in contents:
                    operator(file_path, cluster_dir)

    _copy_files_to_clusters(cluster=cluster, target_path=target_path)


class ClusterLeafProcessor:

    @classmethod
    def process(cls, cluster,
                obj_to_obj: Callable[[Any], Any] = None,
                ):
        if obj_to_obj is None:
            obj_to_obj = lambda x: x

        if isinstance(cluster, dict):
            return {k: cls.process(v, obj_to_obj) for k,v in cluster.items()}
        elif isinstance(cluster, list):
            return [obj_to_obj(x) for x in cluster]
        else:
            return None


class ClusterKeyProcessor:
    @staticmethod
    def name(cluster: Cluster,
             objs_to_cluster_key_formatter: Callable[[List[Any]], str] = None,
             ) -> Cluster:

        if objs_to_cluster_key_formatter is None:
            objs_to_cluster_key_formatter = lambda _: '{key}'

        def recurse(d):
            if isinstance(d, dict):
                new_dict = {}
                res = set()
                for key, value in d.items():
                    d_, res_ = recurse(value)
                    fstr = objs_to_cluster_key_formatter(res_)
                    new_key = fstr.format(key=key)
                    res = res.union(res_)
                    new_dict[new_key] = d_
                return new_dict, res
            else:
                # For leaf nodes (list of file paths), just return them
                return d, set(d)

        d, _ = recurse(cluster)
        return d


class HierarchicalCluster:

    def __init__(self,
                 embedding_func: List[Any],
                 similarity_func: Callable[[Any, Any], float],
                 obj_to_name: Callable[[Any], str] = None,
                ):
        self.emb_func = embedding_func
        self.sim_func = similarity_func

        if obj_to_name is None:
            obj_to_name = lambda x: x if isinstance(x, str) else 'default'
        self.obj_to_name = obj_to_name

    def cluster(self, cluster: Cluster, distance_levels=None) -> Cluster:
        """
        Cluster embeddings in a multi-level hierarchy using a list of distance thresholds.

        :param embeddings: The embeddings to cluster.
        :param distance_levels: A list of distance thresholds for each level of clustering.
        :return: Nested dictionary representing multi-level hierarchical cluster.
        """
        if distance_levels is None:
            distance_levels = [2, 0.5]  # Default value

        initial_cluster = ClusterLeafProcessor.process(
                cluster, obj_to_obj = lambda f: (f, self.emb_func(f)))

        # Function to recursively apply clustering
        def recursive_clustering(current_clusters, level = 0):
            if level >= len(distance_levels):
                # At the final level, return the file paths instead of (file path, embedding) pairs
                return {cluster_id: [fp for fp, _ in cluster_data]
                        for cluster_id, cluster_data in current_clusters.items()}

            new_c = {}
            for cluster_id, cluster_data in current_clusters.items():
                if len(cluster_data) > 1:
                    cluster_embeddings = [emb for _, emb in cluster_data]
                    clustering = AgglomerativeClustering(
                            distance_threshold=distance_levels[level],
                            n_clusters=None)
                    clustering.fit(cluster_embeddings)

                    sub_c = {}
                    for idx, label in enumerate(clustering.labels_):
                        sub_c.setdefault(label, []).append(cluster_data[idx])

                    new_c[cluster_id] = recursive_clustering(sub_c, level + 1)
                elif len(cluster_data) == 1:
                    # If only one item in cluster, no need for further clustering
                    fp, _ = cluster_data[0]
                    new_c[cluster_id] = [fp]
                else:
                    pass

            return new_c

        # Apply recursive clustering starting from level 0
        res_cluster = recursive_clustering(initial_cluster)
        pruned_cluster = self.prune(res_cluster)
        named_cluster = self.name(pruned_cluster)
        return named_cluster

    @classmethod
    def prune(cls, cluster: Cluster) -> Cluster:
        if isinstance(cluster, list):
            return cluster[:]
        if len(cluster) == 1 and isinstance(res := next(iter(cluster.values())), dict):
            return cls.prune(res)
        return {k: cls.prune(v) for k,v in cluster.items()}

    def name(self, cluster: Cluster) -> Cluster:
        def calculate_similarity(item1, item2):
            emb1 = self.emb_func(item1)
            emb2 = self.emb_func(item2)
            return self.sim_func(emb1, emb2)

        def average_similarity(item, items):
            total_similarity = sum(calculate_similarity(item, other) for other in items if other != item)
            return total_similarity / (len(items) - 1) if len(items) > 1 else 0

        def select_best_representation(items):
            if len(items) == 0: return None
            if len(items) == 1: return items[0]

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
                res = {}
                for key, value in d.items():
                    caption = self.obj_to_name(key)
                    caption = utils.get_unique_key(caption, res)
                    res[caption] = rename_to_captions(value)
                return res
            return d

        processed_for_similarity, _ = process_dict_for_similarity(cluster)
        return rename_to_captions(processed_for_similarity)
