import pickle
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain.embeddings import CacheBackedEmbeddings
from tqdm import tqdm
from to_neo4j import parse_vault
from langchain.storage import LocalFileStore
from transformers import AutoModel
from datetime import datetime
import pathlib

from functools import wraps
import time

from joblib import Memory

import logging

# from sentence_transformers.util import semantic_search

# embed_log = logging.getLogger(__name__)
# embed_log.setLevel(logging.DEBUG)
# embed_log.addHandler(logging.StreamHandler())

timing_log = logging.getLogger(__name__)
timing_log.setLevel(logging.DEBUG)
timing_log.addHandler(logging.StreamHandler())

memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")

# store = LocalFileStore("/home/xoph/repos/github/nfroseth/world_graph/langchain_cache")
store = LocalFileStore("~/.cache")

# model_nick = "models--Alibaba-NLP--gte-large-en-v1.5"
# model_name = f"/home/xoph/.cache/huggingface/hub/{model_nick}/snapshots/a0d6174973604c8ef416d9f6ed0f4c17ab32d78d"

model_name = "Alibaba-NLP/gte-large-en-v1.5"
model_nick = model_name.split("/")[1]


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
    )


@timing
def wrap_embedding(model: CacheBackedEmbeddings, text: List[str]):
    return model.embed_documents(text)


@memory.cache
def wrap_notes(vault_path: str):
    return parse_vault(vault_path)


@timing
def get_vault_embeddings(embedding: CacheBackedEmbeddings, chunks) -> FAISS:
    return FAISS.from_documents(chunks, embedding)
    # vectors = {}
    # for name, chunk in tqdm(chunks.items()):
    #     vector_content = wrap_embedding(embedding, chunk)
    #     vectors[name] = [chunk, vector_content]
    #     timing_log.debug(
    #         f"File: {name} {len(vector_content)=} Vector: {vector_content[:1]} Chunk: {chunk[:5]}..."
    #     )

    # return vectors


@memory.cache
def chunking(notes, html_splitter, text_splitter) -> List[Document]:
    chunks = []
    for name, note in tqdm(notes.items()):
        md_header_splits = html_splitter.split_text(note.content)
        splits = text_splitter.split_documents(md_header_splits)
        for split in splits:
            split.metadata = {"note_name": name}
            chunks.append(split)

    return chunks


@timing
def write_vault_vectors_to_disk():
    embedding = get_embedding_model()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding, store, namespace=embedding.model_name
    )

    timing_log.info("Model loaded, parsing vault...")

    # vault_file = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    vault_file = "/home/xoph/SlipBoxCopy/Slip Box"
    path = pathlib.PurePath(vault_file)
    notes = wrap_notes(path)

    timing_log.info("Vault loaded, chunking...")
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]

    chunk_size = 500
    chunk_overlap = 30

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on, strip_headers=False
    # )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = chunking(notes, html_splitter, text_splitter)
    timing_log.info(f"Total Chunks {len(chunks)}")

    timing_log.info("Chunking loaded, embedding...")
    vectors = get_vault_embeddings(cached_embedder, chunks)
    timing_log.debug(f"{path.name} {len(vectors.index_to_docstore_id)=}")

    db_path = f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
    # file_name = f'{path.name}_{model_nick[8:]}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.vdb'
    file_name = f'{path.name}_{model_nick}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

    timing_log.info("Vectors loaded, saving to disk...")
    vectors.save_local(db_path, file_name)


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(write_vault_vectors_to_disk())
