import json
import re
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from joblib import Memory
import requests
from tqdm import tqdm


from neomodel import config, db
from note import Link, Note, Relationship, Node, Chunk
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.embeddings import InfinityEmbeddings
from langchain_community.vectorstores import Neo4jVector

from parse_obsidian_vault import ObsidianVault, MarkdownThenRecursiveSplit
# from get_embedding import timing


parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.INFO)
parse_log.addHandler(logging.StreamHandler())


memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")

COMMUNITY_TYPE = "tags"  # tags, folders, none
CATEGORY_NO_TAGS = "SMD_no_tags"
CATEGORY_DANGLING = "SMD_dangling"

PROP_COMMUNITY = "SMD_community"

PROP_VAULT = "SMD_vault"  # TODO: What do these mean or do?
PROP_PATH = (
    "SMD_path"  # TODO: What do these mean or do? Looking at the parsing and graph code
)
VAULT_NAME = "TEST_VAULT"

CLEAR_ON_CONNECT = True


def get_community(note: Note) -> str:
    if COMMUNITY_TYPE == "tags":
        # TODO: Why only take the First of the Tags for the community of a Note?
        community = escape_cypher(note.tags[0]) if note.has_tags else CATEGORY_NO_TAGS
    elif COMMUNITY_TYPE == "folders":
        community = str(Path(note.properties[PROP_PATH]).parent)
    return community


def cypher_replace(input):
    r = input.replace("-", "_")
    r = r.replace("/", "_")
    return r


def escape_cypher(string):
    # r = escape_quotes(string)
    # Note: CYPHER doesn't allow putting semicolons in text, for some reason. This is lossy!
    # r = r.replace(";", ",")
    r = string.replace("\\u", "\\\\u")
    if r and r[-1] == "\\":
        r += " "
    return r

# @timing
def wrap_embedding(embeddings:Embeddings, text):
    return embeddings.embed_query(text)

@db.write_transaction
def node_from_note_and_fill_communities(
    note: Note, communities: List[str], embeddings: Embeddings
):
    tags = [CATEGORY_NO_TAGS]
    escaped_tags = [escape_cypher(tag) for tag in note.tags] if note.has_tags else tags
    properties = {}
    chunk_nodes = []
    for prop, val in note.properties.items():
        if isinstance(val, str) or isinstance(val, Path):
            properties[prop] = escape_cypher(str(val))
        elif isinstance(val, list) and all(
            [isinstance(prop_it, str) for prop_it in val]
        ):
            properties[prop] = [escape_cypher(prop_it) for prop_it in val]
        elif isinstance(val, list) and all(
            [isinstance(prop_it, Document) for prop_it in val]
        ):
            # TODO: Replace the Enumeration with the Metadata Extraction of the Source header
            for idx, doc in enumerate(val):
                try:
                    chunk_properties = {}
                    chunk_properties["chunk_index"] = idx
                    chunk_properties["content"] = doc.page_content
                    chunk_properties["metadata"] = json.dumps(doc.metadata)
                    try:
                        chunk_properties["embedding"] = wrap_embedding(embeddings, doc.page_content)
                    except Exception as e:
                        parse_log.critical(
                            f"Still creating Chunk, though failed Embedding of Chunk {idx} on {note.name} with {e}"
                        )

                    # TODO: Read or load chunk embeddings

                    chunk = Chunk(**chunk_properties)
                    chunk.save()
                    chunk_nodes.append(chunk)
                except Exception as e:
                    parse_log.critical(
                        f"Failed During the creation of Chunk {idx} on {note.name} with {e}"
                    )
        else:
            parse_log.critical(
                f"During node creation Property Value was neither a str or list of str, nor list of Docs on {note.name} skipping {prop}"
            )

    community = get_community(note)
    communities.append(community)
    properties[PROP_COMMUNITY] = community.index(community)

    node = Node(**properties)
    node.save()

    prev_chunk = None
    for chunk in chunk_nodes:
        if chunk.chunk_index == 0:
            first_rel = node.head.connect(chunk)
            # first_rel.save()

        contains_rel = node.contains.connect(chunk)
        # contains_rel.save()

        if prev_chunk is not None:
            next_rel = prev_chunk.next.connect(chunk)
            # next_rel.save()
        prev_chunk = chunk

    return node, escaped_tags, community


# From: https://stackoverflow.com/a/21625285
@db.write_transaction
def apply_label_to_node(node: Node, labels: List[str]) -> None:
    node_id = node.element_id
    labels_str = ":".join(labels)
    # TODO: /world_graph/test_vault/test_file.md Fails I believe due to #/ tags
    labels_str = cypher_replace(labels_str)
    # parse_log.debug(f"Applying Labels{labels_str}, to {node_id}")
    query = f"""MATCH (n)
    WHERE elementId(n) = $node_id
    SET n:{labels_str}"""
    try:
        results, meta = db.cypher_query(query, {"node_id": node_id})
    except Exception as e:
        parse_log.critical(
            f"Failed to apply labels: {labels} on node_id: {node_id} name: {node.name} due to error: {e}"
        )
    # parse_log.debug(f"Cypher query {query} {results=}, {meta=}")


@db.write_transaction
def create_dangling(name: str, vault_name: str, all_communities: List[str]) -> Node:
    properties = {
        "name": escape_cypher(name),
        "community": all_communities.index(CATEGORY_DANGLING),
        "obsidian_url": escape_cypher(ObsidianVault.obsidian_url(name, vault_name)),
        "content": "Orphan",
        PROP_VAULT: vault_name,
    }
    node = Node(**properties)
    node.save()
    return node


@db.write_transaction
def create_links(links: List[Link], source_node: Node, target_node: Node):
    for link in links:
        properties = {}
        for property, value in link.properties.items():
            properties[property] = escape_cypher(str(value))
        rel = source_node.out_relationships.connect(target_node, properties)
        rel.save()


def load_embedding_model(model_name: Optional[str] = None) -> Tuple[Embeddings, int]:
    infinity_api_url = "http://127.0.0.1:7997"
    embeddings = InfinityEmbeddings(
        model="BAAI/bge-small-en-v1.5", infinity_api_url=infinity_api_url
    )
    dimension = 384 #TODO: Embed Example and get length

    return embeddings, dimension


@db.write_transaction
def create_vector_index(dimension: int):
    db.cypher_query(
        """ CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
            OPTIONS {indexConfig: {
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: 'cosine'
            }}""",
        {"dimensions": dimension},
    )

if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    embeddings, dim = load_embedding_model()
    parse_log.info(f"Embedding model:{embeddings} Dimensions:{dim}")

    #  don't use localhost (intermediate) anthony explains #534
    # https://www.youtube.com/watch?v=98SYTvNw1kw
    url = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
    # config.DATABASE_URL = 'bolt://neo4j:neo4jneo4j@localhost:7687'
    config.DATABASE_URL = f"bolt://{username}:{password}@127.0.0.1:7687"

    if CLEAR_ON_CONNECT:
        parse_log.info(f"Clearing Neo4j Database {VAULT_NAME=}")
        # TODO: Set Prop Vault on Chunks
        # cypher = f"MATCH (n) WHERE n.{PROP_VAULT}='{VAULT_NAME}' DETACH DELETE n"
        cypher = f"MATCH (n) DETACH DELETE n"
        results, meta = db.cypher_query(cypher)
        parse_log.debug(f"{results=}, {meta=}")

    vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    # vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"

    splitter = MarkdownThenRecursiveSplit()

    vault = ObsidianVault(vault_path=vault_path, vault_name="TEST_VAULT")
    vault.parse_obsidian_vault(splitter=splitter)
    notes = vault.parsed_notes

    nodes = {}
    parse_log.info(f"Starting the conversion to nodes.")
    all_tags = [CATEGORY_DANGLING, CATEGORY_NO_TAGS]
    all_communities = all_tags if COMMUNITY_TYPE == "tags" else [CATEGORY_DANGLING]
    for name, note in tqdm(notes.items()):
        # Communities are required to be changed and modified within the node constructor function
        node, node_tags, _ = node_from_note_and_fill_communities(
            note, all_communities, embeddings
        )
        apply_label_to_node(node, node_tags)
        all_tags += [tag for tag in node_tags if not tag in all_tags]
        nodes[name] = node
    
    parse_log.info(f"All Nodes and Tag Labels Created.")

    parse_log.info("Creating the vector index.")
    create_vector_index(dim)


    rels_to_create = []
    nodes_to_create = []
    parse_log.info("Creating relationships")
    for name, note in tqdm(notes.items()):

        no_outgoing_links = not note.out_relationships.keys()
        if no_outgoing_links:
            continue

        source_node = nodes[name]
        for target, links in note.out_relationships.items():
            if target not in nodes:
                nodes[target] = create_dangling(target, VAULT_NAME, all_communities)
                apply_label_to_node(nodes[target], [CATEGORY_DANGLING])
                nodes_to_create.append(nodes[target])
            target_node = nodes[target]
            # TODO: Move the Links to Batches
            create_links(links, source_node, target_node)
