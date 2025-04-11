from typing import List, Dict, Callable, Any, Union, Optional
from sklearn.cluster import AgglomerativeClustering
from utils import MyPath
import os
import numpy as np
import utils
import collections
import functools

import logging
logger = logging.getLogger(__name__)

Cluster = Union[Dict[str, 'Cluster'], Dict[str, List[Any]]] # recursive typing

Path = str
def operate_file_as_cluster(cluster: Cluster,
                         target_path: Path,
                         operator: Callable[[str], str],
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
            new_leaf = []
            for node in cluster:
                res = obj_to_obj(node)
                if res is None:
                    continue
                elif isinstance(res, list):
                    new_leaf += res
                else:
                    new_leaf += res,

            return new_leaf
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

class BaseHierarchicalCluster:
    """
    Base class for hierarchical clustering. It handles the common
    steps: processing the initial cluster, postprocessing (pruning & naming),
    and defines an abstract recursive clustering method.
    """

    def __init__(
        self,
        embedding_func: Callable[[Any], np.ndarray],
        similarity_func: Callable[[Any, Any], float],
        obj_to_name: Callable[[Any], str] = None,
        needs_prune: bool = False, # avoid single child node
    ):
        self.emb_func = embedding_func
        self.sim_func = similarity_func
        if obj_to_name is None:
            obj_to_name = lambda x: x if isinstance(x, str) else "default"
        self.obj_to_name = obj_to_name
        self.needs_prune = needs_prune

    def cluster(self, cluster: Cluster, distance_levels: List[float] = None) -> Cluster:
        """
        Perform clustering by first processing the cluster leaf nodes,
        then recursively clustering them with provided distance thresholds,
        and finally pruning and renaming the results.
        """
        if distance_levels is None:
            distance_levels = [2, 0.5]  # Default thresholds

        # Perform subclass-specific recursive clustering.
        res_cluster = self.recursive_clustering(cluster, distance_levels, level=0)

        # Prune the extra nesting levels.
        if self.needs_prune:
            res_cluster = self.prune(res_cluster)
        return res_cluster

    def recursive_clustering(
        self, current_clusters: Cluster, distance_levels: List[float], level: int
    ) -> Cluster:
        """
        Abstract method for recursive clustering.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses should implement recursive_clustering()")

    @classmethod
    def prune(cls, cluster: Cluster) -> Cluster:
        """
        Prune extraneous nesting in the cluster structure.
        If a dict has only one key that maps to another dict,
        replace it with that inner dict.
        """
        if isinstance(cluster, list):
            return cluster[:]
        if len(cluster) == 1 and isinstance((res := next(iter(cluster.values()))), dict):
            return cls.prune(res)
        return {k: cls.prune(v) for k, v in cluster.items()}

    def get_name_for_cluster_node(self, items: List[str]) -> str:
        """
        Name clusters by selecting a representative element based on the
        intra-cluster similarity. This method processes the hierarchical
        cluster and replaces cluster keys with more informative names.
        """

        def calculate_similarity(item1, item2):
            emb1 = self.emb_func(item1)
            emb2 = self.emb_func(item2)
            return self.sim_func(emb1, emb2)

        def average_similarity(item, items):
            total_similarity = sum(
                calculate_similarity(item, other) for other in items if other != item
            )
            return total_similarity / (len(items) - 1) if len(items) > 1 else 0

        def select_best_representation(items):
            if len(items) == 0:
                return None
            if len(items) == 1:
                return items[0]
            return max(items, key=lambda item: average_similarity(item, items))

        assert not any(isinstance(item, dict) or isinstance(item, list) for item in items), \
                "you can only name a collection cluster leaf node."

        obj = select_best_representation(items)
        cluster_name = self.obj_to_name(obj)
        return cluster_name


class LinearHierarchicalCluster(BaseHierarchicalCluster):
    """
    Linear hierarchical clustering assumes a sorted, linear ordering where
    each element is only compared with its immediate neighbors.
    """

    def __init__(
        self,
        embedding_func: Callable[[Any], np.ndarray],
        similarity_func: Callable[[Any, Any], float],
        sort_key_func: Callable[[Any], Any],
        obj_to_name: Callable[[Any], str] = None,
        debug_distance: bool = False,
        needs_index: bool = False,
        merge_adjacent_same_key: bool = False,
        allow_empty: bool = False,
        **kw,
    ):
        super().__init__(embedding_func, similarity_func, obj_to_name, **kw)
        self.sort_key_func = sort_key_func
        self.debug_distance = debug_distance
        self.needs_index = needs_index
        self.merge_adjacent_same_key = merge_adjacent_same_key
        self.allow_empty = allow_empty

    def recursive_clustering(
        self, cluster: Union[Dict[Any, Any], List[Any]], distance_levels: List[float], level: int
    ) -> Union[Dict[Any, Any], List[Any]]:
        # Base case: no further distance levels to apply.
        if level >= len(distance_levels):
            return self._copy_cluster(cluster)

        # If the current cluster is a dict, process each key recursively.
        if isinstance(cluster, dict):
            return {key: self.recursive_clustering(value, distance_levels, level)
                    for key, value in cluster.items()}

        # If the cluster is a list, sort and split it based on the current distance threshold.
        elif isinstance(cluster, list):
            sorted_items = sorted(cluster, key=self.sort_key_func)
            if len(sorted_items) > 1:
                clusters = self._split_clusters(sorted_items, distance_levels[level])
                sub_clusters = self._assemble_subclusters(clusters)
                # Recurse on the new subclusters at the next level.
                return self.recursive_clustering(sub_clusters, distance_levels, level + 1)
            else:
                # Single element clusters are returned as a copy.
                return sorted_items[:]

    def _copy_cluster(self, cluster: Union[Dict[Any, Any], List[Any]]) -> Union[Dict[Any, Any], List[Any]]:
        """Performs a shallow copy of the cluster depending on its type."""
        if isinstance(cluster, dict):
            return {k: self._copy_cluster(v) for k, v in cluster.items()}
        elif isinstance(cluster, list):
            return cluster[:]
        return cluster

    def _split_clusters(self, sorted_items: List[Any], distance_threshold: float) -> List[List[Any]]:
        """
        Splits a sorted list of items into clusters using the given distance threshold.
        """
        clusters = []
        current_cluster = [sorted_items[0]]
        last_embedding = self.emb_func(sorted_items[0])

        for i, current_item in enumerate(sorted_items[1:], 1):
            current_embedding = self.emb_func(current_item)

            # Allow empty embeddings if enabled.
            if current_embedding is None and self.allow_empty:
                current_cluster.append(current_item)
            else:
                similarity = self.sim_func(last_embedding, current_embedding)
                distance = 1 - similarity

                if distance >= distance_threshold:
                    if self.debug_distance:
                        # Example debug logging.
                        shorter_path = lambda x: f'{x[:15]}...{x[-15:]}' if len(x) > 33 else x
                        logger.debug(f'{shorter_path(current_item)} is {distance:.2f} to {shorter_path(sorted_items[i-1])}.')
                    clusters.append(current_cluster)
                    current_cluster = [current_item]
                else:
                    current_cluster.append(current_item)

            last_embedding = current_embedding

        clusters.append(current_cluster)
        return clusters

    def _assemble_subclusters(self, clusters: List[List[Any]]) -> Dict[str, List[Any]]:
        """
        Assigns unique keys to clusters and optionally merges adjacent clusters with the same name.
        """
        sub_cluster_keys = []
        sub_cluster_values = []
        last_name = None

        for cluster in clusters:
            cluster_name = self.get_name_for_cluster_node(cluster)
            if self.merge_adjacent_same_key and (cluster_name == last_name) and sub_cluster_values:
                sub_cluster_values[-1].extend(cluster)
            else:
                sub_cluster_keys.append(cluster_name)
                sub_cluster_values.append(cluster[:])
            last_name = cluster_name

        sub_clusters = {}
        for index, (name, cluster) in enumerate(zip(sub_cluster_keys, sub_cluster_values), start=1):
            if self.needs_index:
                name = f'{index}-{name}'
            unique_name = utils.get_unique_key(name, sub_clusters)
            sub_clusters[unique_name] = cluster

        return sub_clusters

    def get_name_for_cluster_node(self, items: List[Any]) -> str:
        """
        Generates a cluster name based on the most common name found among its items.
        """
        names = [self.obj_to_name(e) for e in items if not self.is_empty(self.obj_to_name(e))]
        if not names:
            return 'Unknown'
        most_common = collections.Counter(names).most_common(1)[0][0]
        return most_common

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def is_empty(s: Optional[str]) -> bool:
        s = (s or '').lower()
        if s == '' or s is None:
            return True
        if any(term in s for term in ('unknown', '未知')):
            return True
        return False
