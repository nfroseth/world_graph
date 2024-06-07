import logging
from pathlib import Path

import faiss
import pandas as pd
import numpy as np

from get_embedding import timing, get_embedding_model, wrap_embedding_query

from flashrank import Ranker, RerankRequest

embed_log = logging.getLogger(__name__)
embed_log.setLevel(logging.INFO)
embed_log.addHandler(logging.StreamHandler())


@timing
def get_ranker_model():
    # Medium (~110MB), slower model with best zeroshot performance (ranking precision) on out of domain data.
    ranker = Ranker(
        model_name="rank-T5-flan",
        cache_dir="/home/xoph/repos/github/nfroseth/world_graph/ranker_cache",
    )
    return ranker


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
    return index.search(
        query_vector, k=k
    )  # k ios the number of nearest neighbors to retrieve


@timing
def wrap_re_rank(query, ranker, passages):
    rerank_request = RerankRequest(query=query, passages=passages)
    return ranker.rerank(rerank_request)


@timing
def wrap_sort(passages):
    return sorted(passages, key=lambda x: x["score"], reverse=True)


@timing
def main():
    embedding = get_embedding_model()
    ranker = get_ranker_model()

    vector_dir = f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
    embedded_note_chunks_frame = pd.read_pickle(get_latest_vector_path(vector_dir))
    embed_log.debug(f"{embedded_note_chunks_frame.columns}=")
    embed_log.debug(f"{embedded_note_chunks_frame.describe()}=")
    embed_log.debug(embedded_note_chunks_frame)

    chunk_name = embedded_note_chunks_frame.note_name.iloc[0]
    content = embedded_note_chunks_frame.content_chunk.iloc[0]
    passages = []
    for idx, chunk_content in enumerate(content):
        name, chunk_idx = chunk_name[0].split(
            "_chunk_"
        )  # TODO: Fails on _trunc_ when chunking step failed.
        entry = {
            "id": idx,
            "text": chunk_content,
            "meta": {
                "note_name": name,
                "chunk_idx": chunk_idx,
            },
        }
        passages.append(entry)

    vectors = np.array(embedded_note_chunks_frame.embedding.iloc[0])
    embed_log.debug(f"{vectors.shape=}")

    index = fill_index(vectors)

    query = "Here is a list of the most important things that I would like to learn about Both in depth and breadth of knowledge "
    query_vector = np.array([wrap_embedding_query(embedding, query)])
    embed_log.debug(f"{query_vector.shape}")

    D, indexes = wrap_search(
        index, query_vector, k=100
    )  # k ios the number of nearest neighbors to retrieve
    passages_subset = [passages[i] for i in indexes[0]]

    re_ranked_notes = wrap_re_rank(query, ranker, passages_subset)

    embed_log.info(f"Question: {query} Top 3 Similar Chunks:")
    print("---")
    for passages in wrap_sort(re_ranked_notes)[:3]:
        print(
            f'Score: {passages["score"]} From {passages["meta"]["note_name"]}, {passages["text"]}'
        )


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())
