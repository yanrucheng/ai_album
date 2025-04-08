import os
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict
from pathlib import Path
from utils import MyPath
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class MediaUnit:
    unit_id: str
    files: List[str]
    representative_path: str

class MediaGrouper:
    def __init__(self):
        self.parent = {}
        self.files = []
        self.units = []

    def deduplicate_paths(self, filepaths: List[str]):
        for fp in filepaths:
            self.add_file(fp)
        self.apply_filename_strategy()
        self.apply_temporal_strategy(threshold_sec=30)  # 5 second grouping
        units = self.get_units()
        return units
        
    def add_file(self, filepath: str):
        """Add a file with automatic metadata extraction (Python 3.8 compatible)"""
        path = Path(filepath)
            
        self.files += filepath,
        self.parent[str(filepath)] = str(filepath)
        
    def _find(self, filepath: str) -> str:
        """Find root with path compression"""
        while self.parent[filepath] != filepath:
            self.parent[filepath] = self.parent[self.parent[filepath]]
            filepath = self.parent[filepath]
        return filepath
    
    def _union(self, x: str, y: str):
        """Merge two groups"""
        x_root = self._find(x)
        y_root = self._find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root

    def apply_filename_strategy(self):
        """Group files sharing the same base name"""
        base_groups = defaultdict(list)
        for file in self.files:
            path = Path(file)
            base_groups[path.stem].append(file)
            
        for group in base_groups.values():
            if len(group) > 1:
                for i in range(1, len(group)):
                    self._union(group[0], group[i])

        self._update_units()
    
    def apply_temporal_strategy(self, threshold_sec: float):
        """Group files within time threshold"""
        file_times = [(x, MyPath(x).timestamp) for x in self.files]
        time_sorted = sorted(file_times,
                           key=lambda x: x[1])  # Sort by timestamp
        
        for i in range(1, len(time_sorted)):
            curr_file, curr_time = time_sorted[i]
            prev_file, prev_time = time_sorted[i-1]
            logger.debug(f'{curr_file} at {curr_time}, {curr_time - prev_time} sec to the prev.')
            
            if abs(curr_time - prev_time) <= threshold_sec:
                self._union(prev_file, curr_file)

        self._update_units()

    def _update_units(self) -> None:
        groups = defaultdict(list)
        for file in self.files:
            root = self._find(file)
            groups[root].append(file)
        
        self.units = {}
        for _, files in groups.items():
            f = self._pick_representative(files)
            self.units[f] = MediaUnit(
                unit_id=f,
                representative_path=f,
                files=sorted(files),
            )

    def get_unit(self, unit_id):
        return self.units[unit_id]

    def get_units(self) -> Dict[str, MediaUnit]:
        """Create final merged units after all strategies"""
        return sorted(self.units.values(), key=lambda x:x.unit_id)
    
    def _pick_representative(self, files: List[str]) -> str:
        """Select most suitable representative file"""
        files = sorted(files)
        if not files:
            return None
        priority_extensions = ['.jpg', '.jpeg', '.heic', '.png', '.mov', '.mp4']
        for ext in priority_extensions:
            for f in files:
                if Path(f).name[0] in '._': continue
                if Path(f).suffix.lower() == ext:
                    return f
        return files[0]  # Fallback to first file if no preferred extension found

# Example Usage
if __name__ == "__main__":
    grouper = MediaGrouper()
    
    # Add files (paths would normally come from your file system)
    test_files = [
        "/Users/me/Pictures/IMG_123.jpg",
        "/Users/me/Pictures/IMG_123.mov",
        "/Users/me/Pictures/DSC_456.cr3",
        "/Users/me/Pictures/DSC_456.jpg",
        "/Users/me/Pictures/PANO_789.heic"
    ]
    
    units = grouper.deduplicate_paths(test_files)

    for unit in units:
        print(f"Unit {unit.unit_id}:")
        print(f"  Rep: {unit.representative_path}")
        print(f"  Files: {unit.files}")
