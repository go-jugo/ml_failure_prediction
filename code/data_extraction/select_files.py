
import glob
from pathlib import Path


def select_files(raw_data_path, buffer_data_path):
    raw_files = glob.glob(raw_data_path)
    processed_files = glob.glob(buffer_data_path)
    buffer_set = set([Path(Path(filename).stem).stem for filename in processed_files])
    print(buffer_set)
    for filename in set(raw_files):
        print(Path(Path(filename).stem).stem)
    not_yet_processed = [filename for filename in set(raw_files) if Path(Path(filename).stem).stem not in buffer_set]
    print(not_yet_processed)
    return not_yet_processed