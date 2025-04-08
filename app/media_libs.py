import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

from utils import MyPath
from media_utils import MediaValidator

import logging
logger = logging.getLogger(__name__)


@dataclass
class MediaBundle:
    """
    Represents a grouped media files set.
    
    The chosen representative file is stored as a single identifier.
    """
    bundle_id: str
    files: List[str]

    @property
    def representative_path(self):
        return self.bundle_id


class MediaOrganizer:
    """
    Higher-level media organization and grouping.

    Uses low-level media operations (via MediaValidator) to validate files.
    Implements grouping of file paths using filename similarity and timestamp proximity,
    and provides a method to retrieve all valid media files from a folder.
    """

    def __init__(self):
        # Union–find structure to group files
        self.parent: Dict[str, str] = {}
        self.files: List[str] = []
        self.bundles: Dict[str, MediaBundle] = {}

    def organize_files(self, filepaths: List[str]) -> List[MediaBundle]:
        """
        Register file paths, group them by similar filenames and temporal proximity,
        and update media bundles.
        """
        for fp in filepaths:
            self._add_file(fp)
        self._apply_filename_strategy()
        self._apply_temporal_strategy(threshold_sec=30)
        self._update_bundles()
        return self.get_bundles()

    def _add_file(self, filepath: str):
        self.files.append(filepath)
        self.parent[filepath] = filepath

    def _find(self, filepath: str) -> str:
        if self.parent[filepath] != filepath:
            self.parent[filepath] = self._find(self.parent[filepath])
        return self.parent[filepath]

    def _union(self, file1: str, file2: str):
        root1 = self._find(file1)
        root2 = self._find(file2)
        if root1 != root2:
            self.parent[root2] = root1

    def _apply_filename_strategy(self):
        """
        Group files that share the same base name.
        """
        groups = defaultdict(list)
        for file in self.files:
            groups[Path(file).stem].append(file)
        for group in groups.values():
            if len(group) > 1:
                # Union all files in the group
                for file in group[1:]:
                    self._union(group[0], file)

    def _apply_temporal_strategy(self, threshold_sec: float):
        """
        Group files whose timestamps are within threshold_sec seconds of each other.
        """
        file_times = [(file, MyPath(file).timestamp) for file in self.files if MediaValidator.validate(file)]
        sorted_files = sorted(file_times, key=lambda x: x[1])
        for i in range(1, len(sorted_files)):
            curr_file, curr_time = sorted_files[i]
            prev_file, prev_time = sorted_files[i - 1]
            # logger.debug(f"{curr_file} at {curr_time} (delta: {abs(curr_time - prev_time)} sec)")
            if abs(curr_time - prev_time) <= threshold_sec:
                self._union(prev_file, curr_file)

    def _update_bundles(self):
        """
        Rebuild the media bundles from the union–find structure.
        The representative file is chosen as the best candidate from the group.
        """
        groups = defaultdict(list)
        for file in self.files:
            root = self._find(file)
            groups[root].append(file)
        self.bundles.clear()
        for files in groups.values():
            representative = self._select_representative(files)
            self.bundles[representative] = MediaBundle(
                bundle_id=representative,
                files=sorted(files)
            )

    def _select_representative(self, files: List[str]) -> str:
        """
        Select a representative file from the group based on a priority extension order.
        Files starting with dot/underscore are excluded.
        """
        files = sorted(files)
        priority_extensions = ['.jpg', '.jpeg', '.heic', '.png', '.mov', '.mp4']
        for ext in priority_extensions:
            for f in files:
                if Path(f).name.startswith("._"):
                    continue
                if Path(f).suffix.lower() == ext:
                    return f
        return files[0] if files else None

    def get_bundles(self) -> List[MediaBundle]:
        """
        Return a sorted list of media bundles.
        """
        return sorted(self.bundles.values(), key=lambda bundle: bundle.bundle_id)

    def get_bundle(self, bundle_path):
        return self.bundles[bundle_path]

    def get_all_valid_files(self, folder_path: str) -> List[str]:
        """
        Walk the folder, group files using filename and timestamp strategies,
        and then validate the representative file of each group.
        
        Returns:
            A list of file paths that pass validation.
        """
        all_files = sorted(
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
        )

        bundles = self.organize_files(all_files)


        for bundle in bundles:
            logger.debug(f"bundle: {bundle.representative_path}")
            for f in bundle.files:
                path = MyPath(f)
                logger.debug(f"    - {f}, {path.timestr}")

        valid_files = []
        for bundle in bundles:
            rep = bundle.representative_path
            if MediaValidator.validate(rep):
                valid_files.append(rep)
        return valid_files


# Example usage
if __name__ == "__main__":
    folder = "/path/to/your/media"  # Replace with your actual media folder path
    organizer = MediaOrganizer()
    valid_files = organizer.get_all_valid_files(folder)
    print("Valid media files:")
    for file in valid_files:
        print(file)
