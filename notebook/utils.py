from similarities import ClipSimilarity, SiftSimilarity
import os
import shutil


class SingletonModelLoader:
    _instances = {}

    def __new__(cls, model_name_or_path):
        if model_name_or_path not in cls._instances:
            instance = super(SingletonModelLoader, cls).__new__(cls)
            instance.model = ClipSimilarity(model_name_or_path=model_name_or_path)
            cls._instances[model_name_or_path] = instance
        return cls._instances[model_name_or_path]

    @classmethod
    def get_model(cls, model_name_or_path):
        return cls(model_name_or_path).model
        
        
def copy_file_as_cluster(clusters, target_path):
    def _copy_files_to_clusters(clusters, target_path, _current_path=""):
        for cluster_id, contents in clusters.items():
            # Create a new subdirectory for the current cluster
            cluster_dir = os.path.join(target_path, _current_path, str(cluster_id))
            os.makedirs(cluster_dir, exist_ok=True)
    
            if isinstance(contents, dict):
                # If the contents are a dictionary, recurse into it
                copy_files_to_clusters(contents, target_path, os.path.join(_current_path, str(cluster_id)))
            elif isinstance(contents, list):
                # If the contents are a list, copy the files into the current cluster directory
                for file_path in contents:
                    shutil.copy(file_path, cluster_dir)
                    
    _copy_files_to_clusters(clusters=clusters, target_path=target_path)