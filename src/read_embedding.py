import glob
import os
import logging
from pathlib import Path

import pandas as pd

embed_log = logging.getLogger(__name__)
embed_log.setLevel(logging.DEBUG)
embed_log.addHandler(logging.StreamHandler())


def get_latest_vector_path(vector_dir: str) -> Path:
    store_path = Path(vector_dir)
    list_of_files = store_path.glob(f"*")
    latest_file = max(list_of_files, key=lambda p: p.stat().st_ctime)

    embed_log.debug(f"Reading {store_path.name} for latest an got {latest_file.name}")
    return latest_file


def main():
    vector_dir = f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
    embedded_note_chunks_frame = pd.read_pickle(get_latest_vector_path(vector_dir))
    print(f"{embedded_note_chunks_frame.columns}=")
    print(f"{embedded_note_chunks_frame.describe()}=")
    print(embedded_note_chunks_frame)


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())
