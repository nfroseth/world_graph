import itertools
import re
import os
import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from joblib import Memory
from tqdm import tqdm

import yaml
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from urllib.parse import quote

from neomodel import config, db
from note import Link, Note, Relationship, Node


parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.INFO)
parse_log.addHandler(logging.StreamHandler())


memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")

ENABLE_MARKDOWN_TO_HTML = False

PROP_OBSIDIAN_URL = "obsidian_url"
PROP_PATH = (
    "SMD_path"  # TODO: What do these mean or do? Looking at the parsing and graph code
)
PROP_COMMUNITY = "SMD_community"
# INDEX_PROPS = ['name', 'aliases']

PROP_VAULT = "SMD_vault"  # TODO: What do these mean or do?
VAULT_NAME = "TEST_VAULT"

CATEGORY_DANGLING = "SMD_dangling"
CATEGORY_NO_TAGS = "SMD_no_tags"

CLEAR_ON_CONNECT = True
COMMUNITY_TYPE = "tags"  # tags, folders, none

PUNCTUATION = {
    "#",
    "$",
    "!",
    ".",
    ",",
    "?",
    ":",
    ";",
    "`",
    " ",
    "+",
    "=",
    "|",
    "\\",
    os.linesep,
}


def typed_list_parse(file: io.TextIOWrapper, name: str) -> Note:
    line = file.readline()
    note_properties = {}
    while line.strip() != "":
        if line == "---" + os.linesep:
            try:
                note_properties = parse_obsidian_yaml_header(file)
            except Exception as e:
                parse_log.critical(
                    f"Detected but Failed to read the properties yaml head from note: {name} with {e}"
                )
            else:
                break
        line = file.readline()

    # print(f"{line=}")
    parse_log.info(f"{note_properties=}")

    content = []
    relations = defaultdict(list)

    tags = note_properties["tags"] if "tags" in note_properties else []
    while line:
        # TODO: Hand the code for Typed_links with their prefix, though I'm not sure what that is.
        content.append(line)
        for tag in get_tags_from_line(line):
            if tag not in tags:
                tags.append(tag)

        # TODO: Save aliases as Relation property. N: Not sure what this means, need
        # to learn more about Graphs and what specific a relational property is
        line_wikilinks = get_wiki_from_line(line, name)
        for wikilink, link_display_text in line_wikilinks:
            parse_log.debug(
                f"From {name} found link: {wikilink} named: {link_display_text}"
            )
            properties = {
                "context": line,
                "parsed_context": "",
                "link_display_text": link_display_text,
            }
            if ENABLE_MARKDOWN_TO_HTML:
                properties["parsed_context"] = markdownToHtml(line)

            rel = Link("inline", properties)
            relations[wikilink].append(rel)
        line = file.readline()

    raw_content = "".join(content)
    if ENABLE_MARKDOWN_TO_HTML:
        raw_content = markdownToHtml(raw_content)

    return Note(
        name, tags, raw_content, out_relationships=relations, properties=note_properties
    )


# TODO: Review the following hashtag Parsing Code to understand it
class HashtagExtension(Extension):
    # Code based on https://github.com/Kongaloosh/python-markdown-hashtag-extension/blob/master/markdown_hashtags/markdown_hashtag_extension.py
    # Used to extract tags from markdown
    def extendMarkdown(self, md):
        """Add FencedBlockPreprocessor to the Markdown instance."""
        md.registerExtension(self)
        md.preprocessors.register(
            HashtagPreprocessor(md), "hashtag", 10
        )  # After HTML Pre Processor


class HashtagPreprocessor(Preprocessor):
    ALBUM_GROUP_RE = re.compile(r"""(?:(?<=\s)|^)#(\w*[A-Za-z_]+\w*)""")

    def __init__(self, md):
        super(HashtagPreprocessor, self).__init__(md)

    def run(self, lines):
        """Match and store Fenced Code Blocks in the HtmlStash."""
        HASHTAG_WRAP = """<a href="#{0}" class="tag"> #{0}</a>"""
        text = "\n".join(lines)
        while True:
            hashtag = ""
            m = self.ALBUM_GROUP_RE.search(text)
            if m:
                hashtag += HASHTAG_WRAP.format(m.group()[1:])
                placeholder = self.md.htmlStash.store(hashtag)
                text = "%s %s %s" % (text[: m.start()], placeholder, text[m.end() :])
            else:
                break
        return text.split("\n")


def markdownToHtml(md_text):
    return markdown.markdown(
        md_text,
        extensions=[
            "mdx_wikilink_plus",
            "fenced_code",
            "footnotes",
            "tables",
            HashtagExtension(),
        ],
        extension_configs={
            "mdx_wikilink_plus": {"html_class": "internal-link", "url_whitespace": " "}
        },
    )


# TODO: Next steps for parsing, is find all extracting all the links from the Obsidian format
# Seems the smdc package used a regex to find everything, I think I'll follow that.
# I'll see how we do though, due to misplaced notions of speed and maintainability my
# gut reaction is to remove all regex from the application. That'll most likely need profiling
# and throughput testing be I full send the re-write.
def get_wiki_from_line(line: str, note_title: str) -> List[Tuple[str, str]]:
    found_links = re.findall("\[\[(.*?)\]\]", line)
    parsed_links = []
    for wikilink in found_links:
        title = parse_wikilink(wikilink, note_title)
        if title != "":
            parsed_links.append(title)
    return parsed_links


def parse_wikilink(between_brackets: str, note_title: str) -> Tuple[str, str]:
    split_name = iter(between_brackets.split("|"))
    link_title_maybe_header = next(split_name, "")

    link_display_text = next(split_name, link_title_maybe_header)
    # TODO: How should the display text of the link be used in [[test|name]] Cases?
    # Are they properties on the link itself?

    if len(link_title_maybe_header) != 0:
        title = link_title_maybe_header.split("#")[0]
        # TODO: How should the title(#header) piece be used when creating link
        # Very large Notes are split up with headers, so they maybe their own nodes
        # Lastly the ^ operator allows for links to specific blocks of text to
        # more specific.
        if len(title) == 0:  # Wikilinks like [[#header]] refer to itself
            return note_title, link_display_text
        else:  # Wikilinks like [[title#header]]
            return title, link_display_text
    return "", ""


def note_name(path, extension=".md"):
    return os.path.basename(path)[: -len(extension)]


def obsidian_url(name: str, vault: str) -> str:
    return "obsidian://open?vault=" + quote(vault) + "&file=" + quote(name) + ".md"


@memory.cache
def parse_vault(
    notes_path: str, note_ext_type: str = ".md", args=None
) -> Dict[str, Note]:
    note_tree = Path(notes_path).rglob("*" + note_ext_type)

    parsed_notes = {}
    parse_log.info(f"Parsing Notes")

    for path in tqdm(note_tree):
        name = note_name(path)
        parse_log.debug(f"Reading note {name=}")
        with open(path, mode="r", encoding="utf-8") as file:
            note = typed_list_parse(file, name)
            note.properties[PROP_OBSIDIAN_URL] = obsidian_url(name, VAULT_NAME)
            note.properties[PROP_PATH] = path
            note.properties[PROP_VAULT] = VAULT_NAME
            parsed_notes[name] = note

    parse_log.info("Finished parsing notes")
    return parsed_notes


def parse_obsidian_yaml_header(file):
    OVERLOADED_PREFIX = "obsidian_note_property_"

    lines = []
    line = file.readline()
    while line != "---" + os.linesep and line:
        lines.append(line)
        line = file.readline()

    properties = yaml.safe_load("".join(lines))

    for key, value in dict(properties).items():
        if key == "id":
            properties[f"{OVERLOADED_PREFIX}{key}"] = value
            del properties[key]
        elif key == "tags":
            properties[key] = from_yaml_string_get_hierarchical_tag(value)

    return properties


def from_yaml_string_get_hierarchical_tag(yaml_in: Any) -> List[str]:
    yaml_tags = yaml_in
    tags = []
    if isinstance(yaml_tags, str):
        yaml_tags = [single_tag for single_tag in re.split(", | ", yaml_in)]
    if isinstance(yaml_tags, List) and all([isinstance(tag, str) for tag in yaml_tags]):
        for single_tag in yaml_tags:
            hierarchical_split = single_tag.split("/")
            for i, j in itertools.combinations(range(len(hierarchical_split) + 1), 2):
                if i == 0:
                    tag = "/".join(hierarchical_split[i:j])
                    if tag not in tags:
                        tags.append(tag)
    return tags


# https://help.obsidian.md/Editing+and+formatting/Tags#Tag+format
# "#/ is a valid tag, but I don't want it"
def get_tags_from_line(line: str) -> List[str]:
    tags = []
    candidate_tag = ""
    tag_start_pos = 0
    is_valid_tag = False
    lower_line = line.lower()

    for idx, char in enumerate(lower_line):
        does_cand_start_with_tag = next(iter(candidate_tag), 0) == "#"

        if char == "#" or does_cand_start_with_tag:
            tag_start_pos = idx - len(candidate_tag)
            if char in PUNCTUATION and is_valid_tag:
                if tag_start_pos == 0 or lower_line[tag_start_pos - 1] == " ":
                    tags.append(candidate_tag[1:])
                candidate_tag = ""
                is_valid_tag = False
            elif char in PUNCTUATION:
                candidate_tag = ""
            elif char == "/" and candidate_tag != "#":
                tags.append(candidate_tag[1:])

            is_valid_tag = char.isalpha() or char == "/" or is_valid_tag
            candidate_tag += char
    else:
        if (
            next(iter(candidate_tag), 0) == "#"
            and is_valid_tag
            and (tag_start_pos == 0 or lower_line[tag_start_pos - 1] == " ")
            and candidate_tag != "#/"
        ):
            tags.append(candidate_tag[1:])

    return tags


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
        "obsidian_url": escape_cypher(obsidian_url(name, vault_name)),
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
    notes = parse_vault(vault_path)

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
