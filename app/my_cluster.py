from typing import List, Dict, Callable, Any, Union
from sklearn.cluster import AgglomerativeClustering
from utils import MyPath
import os
import numpy as np
import utils
import collections
import pprint

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

        # Process the initial cluster with the common leaf processor.
        initial_cluster = ClusterLeafProcessor.process(
            cluster, obj_to_obj=lambda f: (f, self.emb_func(f))
        )

        # Perform subclass-specific recursive clustering.
        res_cluster = self.recursive_clustering(initial_cluster, distance_levels, level=0)

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


class AgglomerativeHierarchicalCluster(BaseHierarchicalCluster):
    """
    Hierarchical clustering using agglomerative clustering.
    """

    def recursive_clustering(
        self, current_clusters: Cluster, distance_levels: List[float], level: int
    ) -> Cluster:
        if level >= len(distance_levels):
            # At final level, return file paths instead of embedding pairs.
            return {
                cluster_id: [fp for fp, _ in cluster_data]
                for cluster_id, cluster_data in current_clusters.items()
            }

        new_c = {}
        for cluster_id, children in current_clusters.items():
            if isinstance(children, dict):
                new_c[cluster_id] = self.recursive_clustering(children, distance_levels, level)
                continue

            if not isinstance(children, list):
                continue

            if len(children) > 1:
                cluster_embeddings = [emb for _, emb in children]
                clustering = AgglomerativeClustering(
                    distance_threshold=distance_levels[level], n_clusters=None
                )
                clustering.fit(cluster_embeddings)

                sub_clusters = {}
                for idx, label in enumerate(clustering.labels_):
                    sub_clusters.setdefault(label, []).append(children[idx])

                named_clusters = {}
                for cluster in sub_clusters.values():
                    cluster_name = self.get_name_for_cluster_node([x for x, _ in cluster])
                    unique_cluster_name = utils.get_unique_key(cluster_name, named_clusters)
                    named_clusters[unique_cluster_name] = cluster

                new_c[cluster_id] = self.recursive_clustering(
                    named_clusters, distance_levels, level + 1
                )
            elif len(children) == 1:
                fp, _ = children[0]
                new_c[cluster_id] = [fp]

        return new_c


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
        self, current_clusters: Cluster, distance_levels: List[float], level: int
    ) -> Cluster:

        if self.debug_distance:
            print('start', level)
            pprint.pprint(current_clusters)

        if level >= len(distance_levels):
            return {
                cluster_id: [fp for fp, _ in cluster_data]
                for cluster_id, cluster_data in current_clusters.items()
            }

        new_c = {}
        for cluster_id, children in current_clusters.items():
            if isinstance(children, dict):
                new_c[cluster_id] = self.recursive_clustering(children, distance_levels, level)
                continue

            if not isinstance(children, list):
                continue

            # Sort children based on the provided sort key.
            children_sorted = sorted(children, key=lambda x: self.sort_key_func(x[0]))

            if len(children_sorted) > 1:
                clusters = []
                current_cluster = [children_sorted[0]]
                _, last_emb = children_sorted[0]
                for i in range(1, len(children_sorted)):
                    prev_fp, _ = children_sorted[i - 1]
                    curr_fp, curr_emb = children_sorted[i]
                    sim = self.sim_func(last_emb, curr_emb)

                    d = 1 - sim

                    if self.debug_distance:
                        logger.debug(f'{curr_fp[-30:]} is {d:.2f} to {prev_fp[-30:]}')

                    if curr_emb is None and self.allow_empty:
                        current_cluster.append(children_sorted[i])
                        continue
                    elif d >= distance_levels[level]:
                        # too much distance, break into new cluster
                        if self.debug_distance:
                            logger.debug('Too far away. Break into new clusters')
                        clusters.append(current_cluster)
                    else:
                        current_cluster.append(children_sorted[i])

                    last_emb = curr_emb

                clusters.append(current_cluster)

                sub_cluster_keys = []
                sub_cluster_values = []
                last_key = None
                for cluster in clusters:
                    cluster_name = self.get_name_for_cluster_node([x for x, _ in cluster])

                    adjacent_is_same = (cluster_name == last_key)
                    last_key = cluster_name

                    if self.merge_adjacent_same_key and adjacent_is_same:
                        sub_cluster_values[-1] += cluster
                        continue

                    sub_cluster_keys += cluster_name,
                    sub_cluster_values += cluster,

                sub_clusters = {}
                for idx, (cluster_name, cluster) in enumerate(zip(sub_cluster_keys, sub_cluster_values), 1):
                    if self.needs_index:
                        cluster_name = f'{idx}-{cluster_name}'
                    unique_cluster_name = utils.get_unique_key(cluster_name, sub_clusters)
                    sub_clusters[unique_cluster_name] = cluster

                if self.debug_distance:
                    print('End', level)
                    pprint.pprint(sub_clusters)

                new_c[cluster_id] = self.recursive_clustering(
                    sub_clusters, distance_levels, level + 1
                )
            elif len(children) == 1:
                fp, _ = children[0]
                new_c[cluster_id] = [fp]

        return new_c

    def get_name_for_cluster_node(self, items: List[str]) -> str:

        def is_empty(s):
            s = s.lower()
            if s in ('', None):
                return True
            if any(x in s for x in ('unknown', '未知')):
                return True
            return False

        names = [self.obj_to_name(e) for e in items]
        names = [n for n in names if not is_empty(n)]
        if not names:
            return 'Unknown'
        most_common_name = collections.Counter(names).most_common(1)[0][0]
        return most_common_name



