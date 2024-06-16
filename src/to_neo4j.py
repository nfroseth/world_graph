import re
import os
import logging
from pathlib import Path
from typing import List, Tuple

from joblib import Memory
from tqdm import tqdm


from neomodel import config, db
from note import Link, Note, Relationship, Node
from parse_obsidian_vault import parse_obsidian_vault, obsidian_url, ObsidianVault


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

def escape_cypher(string):
    # r = escape_quotes(string)
    # Note: CYPHER doesn't allow putting semicolons in text, for some reason. This is lossy!
    # r = r.replace(";", ",")
    r = string.replace("\\u", "\\\\u")
    if r and r[-1] == "\\":
        r += " "
    return r


def get_community(note: Note) -> str:
    if COMMUNITY_TYPE == "tags":
        # TODO: Why only take the First of the Tags for the community of a Note?
        community = escape_cypher(note.tags[0]) if note.has_tags else CATEGORY_NO_TAGS
    elif COMMUNITY_TYPE == "folders":
        community = str(Path(note.properties[PROP_PATH]).parent)
    return community


def cypher_replace(input):
    r = input.replace("-", "_")
    r = input.replace("/", "_")
    return r


@db.write_transaction
def node_from_note_and_fill_communities(note: Note, communities: List[str]):
    tags = [CATEGORY_NO_TAGS]
    escaped_tags = [escape_cypher(tag) for tag in note.tags] if note.has_tags else tags
    properties = {}
    for prop, val in note.properties.items():
        if isinstance(val, str) or isinstance(val, Path):
            properties[prop] = escape_cypher(str(val))
        elif isinstance(val, list) and all(
            [isinstance(prop_it, str) for prop_it in val]
        ):
            properties[prop] = [escape_cypher(prop_it) for prop_it in val]
        else:
            parse_log.critical(
                f"During node creation Property Value was neither a str or list of str, on {note.name} skipping {prop}"
            )

    community = get_community(note)
    communities.append(community)
    properties[PROP_COMMUNITY] = community.index(community)

    # TODO: Removed labels from the Node object, I'm not sure Neomodel can support multi label Nodes such as with tags
    # So I was going to try to use "raw Cypher to run the commands. "
    # Example: https://stackoverflow.com/a/21625285
    # match (n {id:desired-id})
    # set n :newLabel
    # return n
    node = Node(**properties)
    node.save()
    return node, escaped_tags, community


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

if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")

    url = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
    # config.DATABASE_URL = 'bolt://neo4j:neo4jneo4j@localhost:7687'
    config.DATABASE_URL = f"bolt://{username}:{password}@localhost:7687"

    if CLEAR_ON_CONNECT:
        parse_log.info(f"Clearing Neo4j Database {VAULT_NAME=}")
        cypher = f"MATCH (n) WHERE n.{PROP_VAULT}='{VAULT_NAME}' DETACH DELETE n"
        results, meta = db.cypher_query(cypher)
        parse_log.debug(f"{results=}, {meta=}")

    vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    # vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    notes = parse_obsidian_vault(vault_path)

    nodes = {}
    parse_log.info(f"Starting the conversion to nodes")
    all_tags = [CATEGORY_DANGLING, CATEGORY_NO_TAGS]
    all_communities = all_tags if COMMUNITY_TYPE == "tags" else [CATEGORY_DANGLING]
    for name, note in tqdm(notes.items()):
        # Communities are required to be changed and modified within the node constructor function
        node, node_tags, _ = node_from_note_and_fill_communities(note, all_communities)
        apply_label_to_node(node, node_tags)
        all_tags += [tag for tag in node_tags if not tag in all_tags]
        nodes[name] = node

    parse_log.info(f"All Nodes and Tag Labels Created...")

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
