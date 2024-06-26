import re
import os
import logging
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from neomodel import config, db
from note import Link, Note, Relationship, Node, Chunk
from parse_obsidian_vault import ObsidianVault, MarkdownThenRecursiveSplit

from langchain_experimental.text_splitter import SemanticChunker

parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.INFO)
parse_log.addHandler(logging.StreamHandler())

VAULT_NAME = "TEST_VAULT"

COMMUNITY_TYPE = "tags"  # tags, folders, none
CATEGORY_NO_TAGS = "OBS_NO_TAGS"
CATEGORY_DANGLING = "OBS_DANGLING"
INDEX_PROPS = ["name", "aliases"]


def get_community(note: Note) -> str:
    if COMMUNITY_TYPE == "tags":
        # TODO: Why only take the First of the Tags for the community of a Note?
        community = escape_cypher(note.tags[0]) if note.has_tags else CATEGORY_NO_TAGS
    elif COMMUNITY_TYPE == "folders":
        community = str(
            Path(note.properties[ObsidianVault._FILE_PATH_PROP_NAME]).parent
        )
    return community


def cypher_replace(input):
    r = input
    r = r.replace(">", "_")
    r = r.replace("<", "_")
    r = r.replace("-", "_")
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
        elif val is None:
            parse_log.warning(f"Property Value of {prop} was None on {note.name}")
        else:
            parse_log.critical(
                "During node creation Property Value was neither a str, "
                f"list of str on {note.name} skipping {prop}"
            )

    community = get_community(note)
    communities.append(community)
    properties[ObsidianVault._PROP_COMMUNITY] = community.index(community)

    node = Node(**properties)
    node.save()

    prev_chunk = None
    neo_chunks = []
    for chunk_note in note.chunks:
        try:
            chunk_kwargs = chunk_note.properties
            chunk_kwargs["chunk_index"] = chunk_note.chunk_index

            chunk = Chunk(**chunk_kwargs)
            chunk.save()
            neo_chunks.append(chunk)
        except Exception as e:
            parse_log.critical(
                f"Failed to Convert and save note_chunk to neomodel Chunk Obj"
            )

        if chunk.chunk_index == 0:
            first_rel = node.head.connect(chunk)
            # first_rel.save()

        contains_rel = node.contains.connect(chunk)
        # contains_rel.save()

        if prev_chunk is not None:
            next_rel = prev_chunk.next.connect(chunk)
            # next_rel.save()
        prev_chunk = chunk

    return node, escaped_tags, community, neo_chunks


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
        ObsidianVault._OBSIDIAN_VAULT_PROP_NAME: vault_name,
    }
    node = Node(**properties)
    node.save()
    return node


@db.write_transaction
def create_links(
    links: List[Link], source_node: Node, target_node: Node, rel_name: str
):
    for link in links:
        properties = {}
        for property, value in link.properties.items():
            properties[property] = escape_cypher(str(value))
        rel = source_node.__dict__[rel_name].connect(target_node, properties)
        rel.save()


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


@db.write_transaction
def create_index(index_cypher: str):
    db.cypher_query(index_cypher)


def setup_neo4j_connection(clear_on_connect: bool = False) -> Tuple[str, str, str]:
    url = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
    config.DATABASE_URL = f"bolt://{username}:{password}@127.0.0.1:7687"

    if clear_on_connect:
        parse_log.info(f"Clearing Neo4j Database {VAULT_NAME=}")
        cypher = f"MATCH (n) DETACH DELETE n"  # Delete Everything
        # cypher = f"MATCH (n) WHERE n.{PROP_VAULT}='{VAULT_NAME}' DETACH DELETE n"
        results, meta = db.cypher_query(cypher)
        parse_log.debug(f"{results=}, {meta=}")

    return url, username, password


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    url, username, password = setup_neo4j_connection(clear_on_connect=True)

    vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    # vault_path = "/home/xoph/SlipBoxCopy/LorenzDuremdes/Second-Brain"
    # vault_path = "/home/xoph/SlipBoxCopy/Master_Daily-20240625T165113Z-001/Master_Daily"
    # vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"

    splitter = MarkdownThenRecursiveSplit()
    # embeddings, dim = ObsidianVault.load_embedding_model()
    # splitter = SemanticChunker(
    #     embeddings,
    #     breakpoint_threshold_type="percentile",
    #     breakpoint_threshold_amount=30,
    # )

    vault = ObsidianVault(
        vault_path=vault_path, vault_name="TEST_VAULT", embedding_enabled=True
    )
    vault.parse_obsidian_vault(splitter=splitter)

    notes = vault.parsed_notes

    nodes = {}
    chunks = {}

    parse_log.info(f"Starting the conversion to nodes.")
    all_tags = [CATEGORY_DANGLING, CATEGORY_NO_TAGS]
    all_communities = all_tags if COMMUNITY_TYPE == "tags" else [CATEGORY_DANGLING]
    for name, note in tqdm(notes.items()):
        # Communities are required to be changed and modified within the node constructor function
        node, node_tags, _, neo_chunks = node_from_note_and_fill_communities(
            note,
            all_communities,
        )
        apply_label_to_node(node, node_tags)
        all_tags += [tag for tag in node_tags if not tag in all_tags]
        nodes[name] = node
        chunks[name] = neo_chunks

    parse_log.info(f"All Nodes and Tag Labels Created.")

    rels_to_create = []
    nodes_to_create = []
    parse_log.info("Creating relationships")
    for name, note in tqdm(notes.items()):
        # Send batches to server. Greatly speeds up conversion.
        # Create a List of all the Nodes and relationships and save all of them
        # In batch vs one at a time.
        no_outgoing_links = (
            len(note.out_relationships) == 0 and len(note.out_chunk_relationships) == 0
        )
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
            create_links(links, source_node, target_node, "out_relationships")

        for target_chunk, links in note.out_chunk_relationships.items():
            # TODO: Do we care about Dangling Chunks... Not for now...
            title, header = target_chunk.split("#")

            if title not in nodes:
                nodes[title] = create_dangling(title, VAULT_NAME, all_communities)
                apply_label_to_node(nodes[title], [CATEGORY_DANGLING])
                nodes_to_create.append(nodes[title])

            if title not in notes:
                parse_log.warning(f"Tried to Link a Chunk to an Orphan Note")
                parse_log.debug(f"From {name=} to {title=}, {header=}")
            else:
                try:
                    for idx, chunk_note in enumerate(notes[title].chunks):
                        content = chunk_note.properties["content"]
                        pattern = rf"#{{1,6}} +({header})(?=\n)"
                        match = re.match(pattern, content)
                        if match:
                            target_chunk = chunks[name][idx]
                            create_links(links, source_node, target_chunk, "out_chunks")
                except Exception as e:
                    parse_log.critical(f"Failed on Chunk Linking with {e}")

        for idx, note_chunk in enumerate(notes[name].chunks):

            source_chunk = chunks[name][idx]

            for target, links in note_chunk.out_note_relationships.items():

                if target not in nodes:
                    nodes[target] = create_dangling(target, VAULT_NAME, all_communities)
                    apply_label_to_node(nodes[target], [CATEGORY_DANGLING])
                    nodes_to_create.append(nodes[target])

                target_node = nodes[target]
                create_links(links, source_chunk, target_node, "out_relationships")

            for target_chunk, links in note_chunk.out_chunk_relationships.items():
                # TODO: Do we care about Dangling Chunks... Not for now...
                title, header = target_chunk.split("#")

                if title not in nodes:
                    nodes[title] = create_dangling(title, VAULT_NAME, all_communities)
                    apply_label_to_node(nodes[title], [CATEGORY_DANGLING])
                    nodes_to_create.append(nodes[title])

                if title not in notes:
                    parse_log.warning(f"Tried to Link a Chunk to an Orphan Note")
                    parse_log.debug(f"From {name=} to {title=}, {header=}")
                else:
                    try:
                        title, header = target_chunk.split("#")
                        for idx, chunk_note in enumerate(notes[title].chunks):
                            content = chunk_note.properties["content"]
                            pattern = f"#{{1,6}} +({header})(?=\n)"
                            match = re.match(pattern, content)
                            if match:
                                target_chunk = chunks[name][idx]
                                create_links(
                                    links, source_chunk, target_chunk, "out_chunks"
                                )
                    except Exception as e:
                        parse_log.critical(f"Failed on Chunk Linking with {e}")

    parse_log.info("Creating the vector index.")
    _, dim = ObsidianVault.load_embedding_model()
    create_vector_index(dim)

    parse_log.info("Creating the full texts indexes.")
    # for tag in tqdm(all_tags):
    #     try:
    #         for prop in INDEX_PROPS:
    #             # Seems the / in tags breaks the index cypher
    #             cypher_index = f"CREATE INDEX index_{prop}_{tag} IF NOT EXISTS FOR (n:{tag}) ON (n.{prop})"
    #             create_index(cypher_index)
    #         cypher_index = f"CREATE INDEX index_name_vault IF NOT EXISTS for (n:{tag}) ON (n.{prop})"
    #         create_index(cypher_index)
    #     except Exception as e:
    #         parse_log.warning(f"Warning: Could not create index for {tag} due to {e}")

    indexes = ["obsidian_name_alias", "obsidian_content"]
    for index in indexes:
        try:
            cypher_index = f"DROP INDEX {index}"
            create_index(cypher_index)
        except Exception as e:
            parse_log.warning(f"Warning: Could not drop {index} due to {e}")

    # if all_tags:
    #     all_tags = set(all_tags)

    try:
        # cypher_index = "CALL db.index.fulltext.createNodeIndex(\"obsidian_name_alias\", [\"" + "\", \"".join(all_tags) + "\"], [\"name\", \"aliases\"])"
        # example = "CREATE FULLTEXT INDEX namesAndTeams FOR (n:Employee|Manager) ON EACH [n.name, n.team]"
        cypher_index = "CREATE FULLTEXT INDEX obsidian_name_alias FOR (n:Node) ON EACH [n.name, n.aliases]"
        create_index(cypher_index)
    except Exception as e:
        parse_log.warning(f"Warning: Could not create {cypher_index} due to {e}")

    try:
        # If index Content
        # cypher_index = "CALL db.index.fulltext.createNodeIndex(\"obsidian_content\", [\"" + "\", \"".join(all_tags) + "\"], [\"content\"])"
        cypher_index = "CREATE FULLTEXT INDEX obsidian_content FOR (n:Node|Chunk) ON EACH [n.content]"
        create_index(cypher_index)
    except Exception as e:
        parse_log.warning(f"Warning: Could not create {cypher_index} due to {e}")
