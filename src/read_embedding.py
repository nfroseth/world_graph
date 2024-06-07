import logging
from pathlib import Path

import faiss
import pandas as pd
import numpy as np

from get_embedding import timing, get_embedding_model, wrap_embedding_query

embed_log = logging.getLogger(__name__)
embed_log.setLevel(logging.DEBUG)
embed_log.addHandler(logging.StreamHandler())


def get_latest_vector_path(vector_dir: str) -> Path:
    store_path = Path(vector_dir)
    list_of_files = store_path.glob(f"*")
    latest_file = max(list_of_files, key=lambda p: p.stat().st_ctime)

    embed_log.debug(f"Reading {store_path.name} for latest an got {latest_file.name}")
    return latest_file

@timing
def fill_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

@timing
def wrap_search(index, query_vector, k):
    return index.search(query_vector, k=k)  # k ios the number of nearest neighbors to retrieve

@timing
def main():
    embedding = get_embedding_model()

    vector_dir = f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
    embedded_note_chunks_frame = pd.read_pickle(get_latest_vector_path(vector_dir))
    embed_log.debug(f"{embedded_note_chunks_frame.columns}=")
    embed_log.debug(f"{embedded_note_chunks_frame.describe()}=")
    embed_log.debug(embedded_note_chunks_frame)

    vectors = np.array(embedded_note_chunks_frame.embedding.iloc[0])
    embed_log.debug(f"{vectors.shape=}")

    index = fill_index(vectors)

    query = "Here is a list of the most important things that I would like to learn about Both in depth and breadth of knowledge "
    query_vector = np.array([wrap_embedding_query(embedding, query)])
    embed_log.debug(f"{query_vector.shape}")

    D, indexes = wrap_search(index, query_vector, k=100)  # k ios the number of nearest neighbors to retrieve

    embed_log.info(f"Question: {query} Similar Chunks:")
    for idx in indexes[0]:
        retrieved_content = embedded_note_chunks_frame.content_chunk.iloc[0][idx]
        retrieved_chunk_name = embedded_note_chunks_frame.note_name.iloc[0][idx]
        embed_log.info(f"From {retrieved_chunk_name}, {retrieved_content}")


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())
