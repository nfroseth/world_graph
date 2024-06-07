import pickle
import time
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from functools import wraps
from pandas import DataFrame

import pathlib

from joblib import Memory
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from to_neo4j import parse_vault
from note import Note

# embed_log = timing_log.getLogger(__name__)
# embed_log.setLevel(timing_log.DEBUG)
# embed_log.addHandler(timing_log.StreamHandler())

timing_log = logging.getLogger(__name__)
timing_log.setLevel(logging.DEBUG)
timing_log.addHandler(logging.StreamHandler())

memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")

model_nick = "models--Alibaba-NLP--gte-large-en-v1.5"
model_name = f"/home/xoph/.cache/huggingface/hub/{model_nick}/snapshots/a0d6174973604c8ef416d9f6ed0f4c17ab32d78d"


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.monotonic()
        result = f(*args, **kw)
        te = time.monotonic()
        timing_log.debug(f"func:{f.__name__} Time: {te - ts}")
        return result

    return wrap


# @memory.cache
@timing
def get_embedding_model() -> HuggingFaceEmbeddings:
    # model_name = "Alibaba-NLP/gte-large-en-v1.5"
    # model = AutoModel.
    # model = AutoModel.from_pretrained(
    #     model_name, trust_remote_code=True,
    #     # force_download=True
    # )
    model_kwargs = {
        "device": "cuda",
        "trust_remote_code": True,
    }
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        show_progress=True,
    )


@timing
def wrap_embedding_query(model: HuggingFaceEmbeddings, text):
    return model.embed_query(text)


@timing
def wrap_embedding_document(model: HuggingFaceEmbeddings, text):
    return model.embed_documents(text)


@memory.cache
def wrap_notes(vault_path: str):
    return parse_vault(vault_path)


# @memory.cache
@timing
def get_vault_embeddings(
    embedding: HuggingFaceEmbeddings, notes: Dict[str, Note]
) -> Dict:
    vectors = {}
    for name, note in tqdm(notes.items()):
        vector_content = wrap_embedding_query(embedding, note.content)
        vectors[name] = vector_content
        timing_log.debug(
            f"File: {name} {len(vector_content)=} Vector: {vector_content[:1]}..."
        )

    return vectors


@timing
def get_document_embeddings(
    embedding: HuggingFaceEmbeddings, documents: List
) -> List[List[float]]:
    return embedding.embed_documents(documents)


@memory.cache
def chunking(
    notes: Dict[str, Note], html_splitter: TextSplitter, text_splitter: TextSplitter
) -> Tuple[List[str], List[str]]:
    chunks = []
    chunk_name = []
    for name, note in tqdm(notes.items()):
        try:
            md_header_splits = html_splitter.split_text(note.content)
            splits = text_splitter.split_documents(md_header_splits)
            for idx, split in enumerate(splits):
                chunks.append(str(split))
                chunk_name.append(f"{name}_chunk_{idx}")
        except Exception as e:
            timing_log.critical(
                f"Skipping Chunking on Note: {name} due to failure with {e}"
            )
            timing_log.critical(f"Truncating as an Alternative...")

            chunks.append(note.content[:24000])
            chunk_name.append(f"{name}_trunc_0")

    return chunks, chunk_name


@timing
def write_vault_vectors_to_disk():
    embedding = get_embedding_model()
    timing_log.info("Model loaded, parsing vault...")

    # vault_file = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    vault_file  = "/home/xoph/SlipBoxCopy/Slip Box"
    path = pathlib.PurePath(vault_file)
    notes = wrap_notes(path)
    timing_log.info("Vault loaded, chunking...")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    chunk_size = 250
    chunk_overlap = 30

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks, chunk_names = chunking(notes, markdown_splitter, text_splitter)
    timing_log.info("Chunking loaded, embedding...")

    vectors = get_document_embeddings(embedding, chunks)
    timing_log.debug(f"{path.name} {len(vectors)=}")

    pickle_path = (
        f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
        f'{path.name}_{model_nick[8:]}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pkl'
    )


    embedded_note_chunks = (chunks, chunk_names, vectors)
    embedded_note_chunks_columns = ["content_chunk", "note_name", "embedding"]
    embedded_note_chunks_frame = DataFrame([embedded_note_chunks], columns=embedded_note_chunks_columns) 

    timing_log.info("Vectors loaded, saving to disk...")
    # with open(pickle_path, "wb") as handle:
    #     pickle.dump(embedded_note_chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    embedded_note_chunks_frame.to_pickle(pickle_path)

    timing_log.info("Done")


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(write_vault_vectors_to_disk())