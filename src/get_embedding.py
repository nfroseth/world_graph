import pickle
from typing import Dict
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from to_neo4j import parse_vault
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
timing_log.setLevel(logging.INFO)
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
        timing_log.debug(
            f"func:{f.__name__} Time: {te - ts}"
        )
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
    model_kwargs = {"device": "cuda", 
                    "trust_remote_code": True,
                    }
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

@timing
def wrap_embedding(model: HuggingFaceEmbeddings, text):
    return model.embed_query(text)

@memory.cache
def wrap_notes(vault_path:str):
    return parse_vault(vault_path)

# @memory.cache
@timing
def get_vault_embeddings(embedding:HuggingFaceEmbeddings, notes) -> Dict:
    vectors = {}
    for name, note in tqdm(notes.items()):
        vector_content = wrap_embedding(embedding, note.properties["content"])
        vectors[name] = vector_content
        timing_log.debug(f"File: {name} {len(vector_content)=} Vector: {vector_content[:1]}...")

    return vectors

@timing
def write_vault_vectors_to_disk():
    embedding = get_embedding_model()
    logging.info("Model loaded, parsing vault...")

    vault_file = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    # vault_file  = "/home/xoph/SlipBoxCopy/Slip Box"
    path = pathlib.PurePath(vault_file)
    notes = wrap_notes(path)

    logging.info("Vault loaded, embedding...")
    vectors = get_vault_embeddings(embedding, notes)
    timing_log.debug(f"{path.name} {len(vectors)=}")
    
    pickle_path = (f"/home/xoph/repos/github/nfroseth/world_graph/vectors/"
                   f'{path.name}_{model_nick[8:]}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pkl')

    logging.info("Vectors loaded, saving to disk...")
    with open(pickle_path, 'wb') as handle:
        pickle.dump(vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(write_vault_vectors_to_disk())

