import logging
import os
from pathlib import Path

import faiss
import pandas as pd
import numpy as np

from get_embedding import timing, get_embedding_model, wrap_embedding_query

from flashrank import Ranker, RerankRequest
from neo4j import GraphDatabase, RoutingControl

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv


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


# @timing
def get_best_passages(passages):
    return sorted(passages, key=lambda x: x["score"], reverse=True)


@timing
def get_related_linked_note(driver, name):
    cypher_query = """
        MATCH (p)-[r:RELATED_TO]->(n) WHERE p.name = $name
        RETURN n.name
        UNION
        MATCH (n)-[r:RELATED_TO]->(p) WHERE p.name = $name
        RETURN n.name
    """
    records, summary, keys = driver.execute_query(
        cypher_query, name=name, database_="neo4j", routing_=RoutingControl.READ
    )

    # pretty_cypher = cypher_query.replace('\n'," ")
    # embed_log.debug(f"Cypher query {pretty_cypher}")
    embed_log.debug(f"For {name} number of referenced notes: {len(records)=}")

    return records


@timing
def graph_expansion(URI, AUTH, selected_passages, lookup_passages):
    graph_expanded_subset = []
    notes_to_expand = {passage["meta"]["note_name"] for passage in selected_passages}
    embed_log.info(
        f"Number of notes expanding on: {len(notes_to_expand)} {notes_to_expand}"
    )

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        for note_name in notes_to_expand:
            records = get_related_linked_note(driver, note_name)
            for record in records:
                # TODO: Getting every chunk of all the notes feels that we're sending too much forward
                graph_expanded_subset += [
                    chunk
                    for chunk in lookup_passages
                    if chunk["meta"]["note_name"] == record["n.name"]
                ]
    return graph_expanded_subset


@timing
def main():
    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "neo4jneo4j")

    llm = ChatOpenAI(
        openai_api_key=getenv("WORKOUT_ENGINE_OPEN_ROUTER_AI_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1/",
        model="meta-llama/llama-3-8b-instruct:free",
    )

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
        name, chunk_idx = chunk_name[idx].split(
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

    # query = "Here is a list of the most important things that I would like to learn about Both in depth and breadth of knowledge "
    query = "What are the different ways that I can improve my keyboard typing ability to transfer my thoughts into the computer?"
    query_vector = np.array([wrap_embedding_query(embedding, query)])
    embed_log.debug(f"{query_vector.shape}")

    top_k = 250
    D, indexes = wrap_search(
        index, query_vector, k=top_k
    )  # k ios the number of nearest neighbors to retrieve
    passages_subset = [passages[i] for i in indexes[0]]

    # What notes should I seek graph expansion on?
    filter_k = 5  # Try a Cutoff
    re_ranked_notes = wrap_re_rank(query, ranker, passages_subset)
    passage_re_ranked_subset = [
        passages for passages in get_best_passages(re_ranked_notes)[:filter_k]
    ]

    graph_expanded_subset = graph_expansion(
        URI, AUTH, passage_re_ranked_subset, passages
    )
    graph_expanded_subset += passage_re_ranked_subset
    embed_log.info(f"From {filter_k} passages to {len(graph_expanded_subset)=}")

    re_ranked_graph_notes = wrap_re_rank(query, ranker, graph_expanded_subset)

    embed_log.info(f"Question: {query} Sorted Similar Chunks:")
    print("---")
    for gr_rank_chk in get_best_passages(re_ranked_graph_notes):
        print(
            f'Score: {gr_rank_chk["score"]} From {gr_rank_chk["meta"]["note_name"]} Chunk: {gr_rank_chk["meta"]["chunk_idx"]}, {gr_rank_chk["text"][12:24]}...'
        )
    print("---")

    template = """System: You are a knowledgeable assistant that uses structured data from knowledge graphs to answer questions accurately. Provide detailed and contextually relevant answers.

    ---

    Notes: {context}

    ---

    User: {query}"""
    context = "\n".join(
        [chunk["text"] for chunk in get_best_passages(re_ranked_graph_notes)[:16]]
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    llm_result = llm_chain.run(context=context, query=query)
    print(llm_result)


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())
