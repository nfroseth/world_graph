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
from urllib.parse import quote

from note import Link, Note

import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.INFO)
parse_log.addHandler(logging.StreamHandler())

memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")

ENABLE_MARKDOWN_TO_HTML = False

PROP_OBSIDIAN_URL = "obsidian_url"
# INDEX_PROPS = ['name', 'aliases']
PROP_VAULT = "SMD_vault"  # TODO: What do these mean or do?
PROP_PATH = (
    "SMD_path"  # TODO: What do these mean or do? Looking at the parsing and graph code
)

VAULT_NAME = "TEST_VAULT"



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


def obsidian_url(name: str, vault: str) -> str:
    return "obsidian://open?vault=" + quote(vault) + "&file=" + quote(name) + ".md"


def note_name(path, extension=".md"):
    return os.path.basename(path)[: -len(extension)]


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


# @memory.cache
def parse_obsidian_vault(
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


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    # vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    notes = parse_obsidian_vault(vault_path)
