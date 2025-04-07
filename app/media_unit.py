import os
import glob

class MediaUnitManager:
    # Precedence order for representative path selection
    REPRESENTATIVE_EXTENSIONS = ['.jpg', '.jpeg', '.heic', '.png', '.mov', '.mp4', '.cr3']

    
    @staticmethod
    def get_unit_id(filepath):
        return os.path.splitext(filepath)[0]

    @classmethod
    def get_unique_paths(cls, filepaths):
        seen = set()
        media_unit_paths = []

        for fp in filepaths:
            unit_id = cls.get_unit_id(fp)
            if unit_id in seen: continue
            seen.add(unit_id)
            media_unit_paths.append(cls.get_path(unit_id))

        return media_unit_paths

    @classmethod
    def get_path(cls, unit_id):
        """Update the representative path based on the precedence order."""


        file_list = cls.get_file_list(unit_id)
        if not file_list:
            return

        # Check for files in precedence order
        for ext in cls.REPRESENTATIVE_EXTENSIONS:
            for file_path in file_list:
                if file_path.lower().endswith(ext):
                    return file_path

        return file_list[0]


    @staticmethod
    def get_file_list(unit_id):
        """Return a sorted list of all files matching the unit_id with any extension."""
        return sorted(glob.glob(f"{unit_id}.*"))
