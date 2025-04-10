import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

import utils
from media_utils import MediaValidator
from function_tracker import global_tracker
import time

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

    def __init__(self,
                 max_gap_for_bundle: int = 15,
                 verbosity = utils.Verbosity.Once,
                 skip_validate = False,
                 ):
        # Union–find structure to group files
        self.parent: Dict[str, str] = {}
        self.files: List[str] = []
        self.bundles: Dict[str, MediaBundle] = {}
        self.max_gap_for_bundle = max_gap_for_bundle
        self.verbosity = verbosity
        self.skip_validate = skip_validate

    def organize_files(self, filepaths: List[str]) -> List[MediaBundle]:
        """
        Register file paths, group them by similar filenames and temporal proximity,
        and update media bundles.
        """
        logger.debug(1)
        for fp in filepaths:
            self._add_file(fp)
        logger.debug(2)
        self._apply_filename_strategy()
        logger.debug(3)
        self._apply_temporal_strategy(threshold_sec=self.max_gap_for_bundle)
        logger.debug(4)
        self._update_bundles()
        logger.debug(5)
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
            key = Path(file).with_suffix('')

            if Path(key).stem.startswith('._'):
                key = Path(key).parent / Path(key).stem[2:]

            groups[key].append(file)
        for group in groups.values():
            if len(group) > 1:
                # Union all files in the group
                for file in group[1:]:
                    self._union(group[0], file)

    def _apply_temporal_strategy(self, threshold_sec: float):
        """
        Group files whose timestamps are within threshold_sec seconds of each other.
        """
        file_times = []
        for file in self.files:
            validated = MediaValidator.validate(file, only_check_file_name = self.skip_validate)
            if not validated:
                continue
            time = utils.MyPath(file).timestamp
            file_times += (file, time),
        sorted_files = sorted(file_times, key=lambda x: (utils.MyPath(x[0]).extension, x[1]))
        for i in range(1, len(sorted_files)):
            curr_file, curr_time = sorted_files[i]
            prev_file, prev_time = sorted_files[i - 1]
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
        return sorted(self.bundles.values(),
                      key=lambda bundle: utils.MyPath(bundle.representative_path).timestamp)

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
        self._show_bundle_stat(bundles)

        valid_files = []
        for bundle in bundles:
            rep = bundle.representative_path
            if MediaValidator.validate(rep):
                valid_files.append(rep)

        return valid_files

    def _show_bundle_stat(self, bundles):
        # Initialize counters for stats
        total_bundles = len(bundles)
        total_files = sum(len(b.files) for b in bundles)

        for bundle in bundles:

            if not MediaValidator.validate(bundle.representative_path):
                continue

            if self.verbosity >= utils.Verbosity.Detail:
                logger.debug(f"bundle: {bundle.representative_path}")
            
            file_count = len(bundle.files)
            only_show = 2
            if file_count > only_show:
                first_half = only_show // 2
                for f in bundle.files[:first_half]:
                    path = utils.MyPath(f)
                    if self.verbosity >= utils.Verbosity.Detail:
                        logger.debug(f"    - {f}, {path.timestr}")
                
                # Omitted count
                omitted = file_count - only_show
                if self.verbosity >= utils.Verbosity.Detail:
                    logger.debug(f"    - ... {omitted} files omitted ...")
                
                last_half = only_show - first_half
                for f in bundle.files[-last_half:]:
                    path = utils.MyPath(f)
                    if self.verbosity >= utils.Verbosity.Detail:
                        logger.debug(f"    - {f}, {path.timestr}")
            else:
                for f in bundle.files:
                    path = utils.MyPath(f)
                    if self.verbosity >= utils.Verbosity.Detail:
                        logger.debug(f"    - {f}, {path.timestr}")

        # Calculate statistics
        average_selection_rate = total_bundles / total_files if total_files > 0 else 0

        # Log statistics
        if self.verbosity >= utils.Verbosity.Once:
            logger.info("\n=== Bundle Processing Statistics ===")
            logger.info(f"Total bundles processed: {total_bundles}")
            logger.info(f"Total files across all bundles: {total_files}")
            logger.info(f"Average selection rate: {average_selection_rate:.2%}")

def get_file_timestamps(path):
    """
    Return start and end timestamps of a file
    For images: start = end
    For videos: end = start + duration
    For other files: start = end
    """
    # Get the base timestamp (creation time or modification time)
    base_timestamp = utils.MyPath(path).timestamp
    
    # Initialize start and end timestamps
    start_timestamp = base_timestamp
    end_timestamp = base_timestamp
    
    try:
        if MediaValidator.is_image(path):
            # For images, start = end
            return (start_timestamp, end_timestamp)
        
        if MediaValidator.is_video(path):
            # Get video duration and calculate end timestamp
            duration = utils.get_video_duration(path)
            if duration > 0:
                end_timestamp = start_timestamp + duration
            return (start_timestamp, end_timestamp)
            
    except Exception:
        # If anything fails, return the base timestamps
        pass
    
    # For all other files, return start = end
    return (start_timestamp, end_timestamp)




# Example usage
if __name__ == "__main__":
    folder = "/path/to/your/media"  # Replace with your actual media folder path
    organizer = MediaOrganizer()
    valid_files = organizer.get_all_valid_files(folder)
    print("Valid media files:")
    for file in valid_files:
        print(file)
