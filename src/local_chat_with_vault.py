import os
from typing import List
from neomodel import config, db
import chainlit as cl

import numpy as np

from neo4j import GraphDatabase, RoutingControl
from langchain_community.vectorstores import Neo4jVector
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.embeddings import InfinityEmbeddings
import requests

from perf import timing

# from parse_obsidian_vault import ObsidianVault


def setup():
    url = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

    config.DATABASE_URL = f"bolt://{username}:{password}@127.0.0.1:7687"

    embeddings = InfinityEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
        infinity_api_url="http://127.0.0.1:7997",
    )

    index_name = "vector"  # default index name

    store = Neo4jVector.from_existing_graph(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label="Chunk",
        text_node_properties=["content"],
        embedding_node_property="embedding",
    )
    return store, url, (username, password)


@timing
def get_rerank(text: str, documents: List[str]):
    results = requests.post(
        "http://127.0.0.1:7997/rerank",
        json={
            "model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "query": text,
            "documents": documents,
        },
    )
    return results.json()["results"]


@timing
def wrap_search(store: Neo4jVector, user_message, k=100):
    return store.similarity_search_with_score(user_message, k=k)


store, URI, AUTH = setup()

# CONVERSATION HISTORY:
# {conversation_history}
# POSTER:
# {poster}
template = """
You are a customer support agent, helping posters by following directives and answering questions.
Generate your response by following the steps below:
1. Recursively break-down the post into smaller questions/directives
2. For each atomic question/directive:
2a. Select the most relevant information from the context in light of the conversation history
3. Generate a draft response using the selected information, whose brevity/detail are tailored to the poster’s expertise
4. Remove duplicate content from the draft response
5. Generate your final response after adjusting it to increase accuracy and relevance
6. Now only show your final response! Do not provide any explanations or details

CONTEXT:
{context}

POST:
{post}

POSTER’S EXPERTISE: {expertise_level}

Beginners want detailed answers with explanations. Experts want concise answers without explanations.
If you are unable to help the reviewer, let them know that help is on the way.
"""

# template = """
# Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words.

# Question: {post}

# Context: {context}

# """

# llm = ChatOpenAI(
#     openai_api_key="lm-studio",
#     openai_api_base="http://172.30.128.1:1234/v1",
#     model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
#     temperature=0,
#     max_tokens=512,
#     # model_kwargs={"repeat_penalty":1.2},
#     # streaming=True
# )

llm = ChatOllama(model="qwen2:7b-instruct-q6_K")
# llm = ChatOllama(model="meta-llama-3-8b-instruct.Q5_1.gguf:latest")

@timing
def related_notes(driver, name, depth: int = 5):
    cypher_query = f"""
    MATCH (n {{name:"{name}"}}) - [r0*..{depth}] -> (p:Chunk) 
    return p
    UNION
    MATCH (p:Chunk)  - [r0:RELATED_TO*..{depth}] -> (n {{name:"{name}"}}) 
    RETURN p"""
    # MATCH (n {{name:"{name}"}}) - [r0:RELATED_TO*..{depth}] - (p:Chunk) RETURN p

    records, summary, keys = driver.execute_query(
        cypher_query, database_="neo4j", routing_=RoutingControl.READ
    )

    print(f"For {name} number of referenced notes: {len(records)=}")

    return records


@timing
def graph_expansion(URI, AUTH, notes_to_expand, depth: int = 5) -> List[str]:
    graph_expanded_subset = {}
    print(f"Number of notes expanding on: {len(notes_to_expand)} {notes_to_expand}")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        for note_name in notes_to_expand:
            records = related_notes(driver, note_name, depth=depth)
            for record in records:
                name = record["p"]["name"]
                chunk_idx = record["p"]["chunk_index"]
                content = record["p"]["content"]

                if name not in graph_expanded_subset:
                    graph_expanded_subset[name] = {chunk_idx: content}

                graph_expanded_subset[name][chunk_idx] = content

    flat_content = [
        content
        for name, chunk_dict in graph_expanded_subset.items()
        for chunk_idx, content in chunk_dict.items()
    ]
    # flat_content += [passage["content"] for passage in selected_passages]
    return flat_content


@cl.on_chat_start
async def main():
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "post",
            "context",
            # "conversation_history",
            # "poster",
            "expertise_level",
        ],
    )
    parser = StrOutputParser()
    llm_chain = prompt | llm | parser
    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("neo_store", store)

    return llm_chain


# What is the best way to improve typing speed?
@cl.on_message
async def run(message: cl.Message):
    user_message = message.content
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )
    llm_chain = cl.user_session.get("llm_chain")
    store = cl.user_session.get("neo_store")

    docs_with_score = wrap_search(store, user_message, 1000)

    docs = [doc.page_content for doc, score in docs_with_score]

    re_ranked_scores = get_rerank(user_message, docs)
    re_ranked_index = sorted(
        re_ranked_scores, key=lambda x: x["relevance_score"], reverse=True
    )
    top_k = 75
    re_ranked_docs = [docs[row["index"]] for row in re_ranked_index[:top_k]]
    re_ranked_doc_names = [
        docs_with_score[row["index"]][0].metadata["name"]
        for row in re_ranked_index[:top_k]
    ]

    depth = 5
    after_graph = (
        graph_expansion(URI, AUTH, re_ranked_doc_names, depth) + re_ranked_docs
    )
    print(f"From {len(re_ranked_doc_names)} passages to {len(after_graph)=}")

    re_ranked_scores = get_rerank(user_message, after_graph)
    re_ranked_index = sorted(
        re_ranked_scores, key=lambda x: x["relevance_score"], reverse=True
    )
    top_k = 15
    re_ranked_docs = [after_graph[row["index"]] for row in re_ranked_index[:top_k]]

    context = "\n ".join(re_ranked_docs)

    llm_input = {"context": context, "post": user_message, "expertise_level": "expert"}
    print(f'Context Length: {len(llm_input["context"])}')
    #    Start {llm_input["context"][:25]}')

    msg = cl.Message(content="From the following Notes: [1]... \n\n")
    await msg.send()

    async for chunk in llm_chain.astream(
        llm_input,
        config=RunnableConfig(callbacks=[cb]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
