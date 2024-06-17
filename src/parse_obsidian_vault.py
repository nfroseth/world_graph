import itertools
import re
import os
import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from joblib import Memory
from tqdm import tqdm

import yaml
from urllib.parse import quote

from note import Link, Note

import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    TextSplitter,
)
from langchain_core.documents import Document


parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.INFO)
parse_log.addHandler(logging.StreamHandler())

memory = Memory("/home/xoph/repos/github/nfroseth/world_graph/joblib_memory_cache")


# INDEX_PROPS = ['name', 'aliases']


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


# @memory.cache


class ObsidianVault:
    def __init__(
        self,
        vault_path: str,
        vault_name: str,
        note_ext_type: str = ".md",
        make_tmp_copy: bool = False,
    ):
        self._vault_path = vault_path
        self._note_ext_type = note_ext_type

        self._parsed_notes = {}

        self._VAULT_NAME = vault_name
        self._OBSIDIAN_OVERLOADED_PREFIX = "obsidian_note_property_"
        self._OBSIDIAN_URL_PROP_NAME = "obsidian_url"
        self._OBSIDIAN_VAULT_PROP_NAME = "obsidian_vault"
        self._FILE_PATH_PROP_NAME = "file_path"
        self._ENABLE_MARKDOWN_TO_HTML = False

        if make_tmp_copy:
            # TODO: Copy notes to a temp directory to allow for the suggestion of changes
            raise NotImplementedError(
                "Copy notes to a temp directory to allow for the suggestion of changes"
            )

    @property
    def parsed_notes(self) -> Dict[str, Note]:
        if len(self._parsed_notes) == 0:
            raise Exception("No notes have been parsed.")
        else:
            return self._parsed_notes

    @staticmethod
    def obsidian_url(name: str, vault: str, note_ext_type: str = ".md") -> str:
        return f"obsidian://open?vault={quote(vault)}&file={quote(name)}{note_ext_type}"

    @staticmethod
    def from_yaml_string_get_hierarchical_tag(yaml_in: Any) -> List[str]:
        yaml_tags = yaml_in
        tags = []
        if isinstance(yaml_tags, str):
            yaml_tags = [single_tag for single_tag in re.split(", | ", yaml_in)]
        if isinstance(yaml_tags, List) and all(
            [isinstance(tag, str) for tag in yaml_tags]
        ):
            for single_tag in yaml_tags:
                hierarchical_split = single_tag.split("/")
                for i, j in itertools.combinations(
                    range(len(hierarchical_split) + 1), 2
                ):
                    if i == 0:
                        tag = "/".join(hierarchical_split[i:j])
                        if tag not in tags:
                            tags.append(tag)
        return tags

    def parse_obsidian_yaml_header(self, file: io.TextIOWrapper):
        lines = []
        line = file.readline()
        while line != "---" + os.linesep and line:
            lines.append(line)
            line = file.readline()

        properties = yaml.safe_load("".join(lines))

        for key, value in dict(properties).items():
            if (
                key == "id"
            ):  # ID Overloaded in Down Stream Neo4j, TODO: Move the ownership of conversion to Neo4j specific code
                properties[f"{self._OBSIDIAN_OVERLOADED_PREFIX}{key}"] = value
                del properties[key]
            elif key == "tags":
                properties[key] = ObsidianVault.from_yaml_string_get_hierarchical_tag(
                    value
                )

        return properties

    def get_note_properties_from_header(
        self, file: io.TextIOWrapper, name
    ) -> Tuple[Dict[str, Any], str]:
        note_properties = {}
        line = file.readline()
        while line.strip() != "":
            if line == "---" + os.linesep:
                try:
                    note_properties = self.parse_obsidian_yaml_header(file)
                except Exception as e:
                    parse_log.critical(
                        f"Detected but Failed to read the properties yaml head from note: {name} with {e}"
                    )
                else:
                    break
            line = file.readline()

        parse_log.info(f"{note_properties=}")
        return note_properties, line

    def get_wiki_from_line(self, line: str, note_title: str, chunk_idx: int) -> List[Tuple[str, Link]]:
        found_links = re.findall("\[\[(.*?)\]\]", line)
        parsed_links = []
        for wikilink in found_links:
            title, display, header = ObsidianVault.parse_wikilink(wikilink, note_title)
            title_with_header = f"{title}{'#'+header if header else ''}"

            parse_log.debug(
                f"From {note_title} found link: {title_with_header} named: {display}"
            )

            properties = {
                "title": title,
                "context": line,
                "header": header,
                "chunk_index": chunk_idx,
                "parsed_context": "",
                "link_display_text": display,
            }

            if self._ENABLE_MARKDOWN_TO_HTML:
                properties["parsed_context"] = markdownToHtml(line)

            rel = Link("inline", properties)
            parsed_links.append((wikilink, rel))
        return parsed_links

    @staticmethod
    def parse_wikilink(between_brackets: str, note_title: str) -> Tuple[str, str, str]:
        title = ""
        header = ""
        split_name = iter(between_brackets.split("|"))
        link_title_maybe_header = next(split_name, "")

        link_display_text = next(split_name, link_title_maybe_header)
        # TODO: How should the display text of the link be used in [[test|name]] Cases?
        # Are they properties on the link itself?

        if len(link_title_maybe_header) != 0:
            # I'm linking to everything till the next header of that level
            link_split_header = link_title_maybe_header.split("#")
            if len(link_split_header) > 2:
                parse_log.critical(
                    f"At {note_title} Found link with more than two header # separators: {between_brackets}, Skipping..."
                )
            elif len(link_split_header) > 1:
                header = link_split_header[1]

            title = link_split_header[0]

            # TODO: How should the title(#header) piece be used when creating link
            # Very large Notes are split up with headers, so they maybe their own nodes
            # Lastly the ^ operator allows for links to specific blocks of text to
            # more specific.
            # https://help.obsidian.md/Linking+notes+and+files/Internal+links#Link+to+a+block+in+a+note

            # Currently Tackling this issue, I'll wait to handle the ^ block link operator, but I'd
            # like to be able to link to a the "Closest" Chunk within a Note if it's available.
            # It'll depend upon the chunking strategy
            # Note A, Chunk 3 -> Note B, Chunk 4, Chunk 5, Chunk 6
            # Philosophy, be as specific as possible when linking to another note, as expansion strategies can
            # consider what to do with something too specific.

            if len(title) == 0:  # Wikilinks like [[#header]] refer to itself
                return note_title, link_display_text, header

        return title, link_display_text, header

    def typed_list_parse(
        self,
        file: io.TextIOWrapper,
        path: Path,
        splitter: Optional[TextSplitter] = None,
    ) -> Note:
        note_name = path.stem
        note_properties, line = self.get_note_properties_from_header(file, note_name)

        relations = defaultdict(list)
        note_properties[self._OBSIDIAN_URL_PROP_NAME] = ObsidianVault.obsidian_url(
            note_name, self._VAULT_NAME
        )
        note_properties[self._FILE_PATH_PROP_NAME] = path
        note_properties[self._OBSIDIAN_VAULT_PROP_NAME] = self._VAULT_NAME

        tags = note_properties["tags"] if "tags" in note_properties else []

        chunks = []
        if splitter is not None:
            raw_content_after_header = file.read()
            chunks = splitter.split_text(raw_content_after_header)
        else:
            chunks = [Document(page_content=file.read())]

        note_properties["chunks"] = chunks

        lines = []
        for chunk_idx, chunk in enumerate(chunks):
            for line in chunk.page_content.split("\n"):
                lines.append(line)
                # TODO: Hand the code for Typed_links with their prefix, though I'm not sure what that is.
                for tag in get_tags_from_line(line):
                    if tag not in tags:
                        tags.append(tag)

                # TODO: Save aliases as Relation property. N: Not sure what this means, need
                # to learn more about Graphs and what specific a relational property is

                line_links = self.get_wiki_from_line(line, note_name, chunk_idx)
                for wikilink, rel in line_links:
                    relations[wikilink].append(rel)

        raw_content = "".join(lines)
        if self._ENABLE_MARKDOWN_TO_HTML:
            raw_content = markdownToHtml(raw_content)

        return Note(
            note_name,
            tags,
            raw_content,
            out_relationships=relations,
            properties=note_properties,
        )

    def parse_obsidian_vault(self, splitter: Optional[TextSplitter] = None) -> None:
        parse_log.info(f"Starting the parsing of obsidian vault")

        note_tree = Path(self._vault_path).rglob("*" + self._note_ext_type)
        parsed_notes = {}

        for path in tqdm(note_tree):
            name = path.stem
            parse_log.debug(f"Reading note {name=}")

            with open(path, mode="r", encoding="utf-8") as file:
                note = self.typed_list_parse(file, path, splitter)
                parsed_notes[name] = note

        parse_log.info("Finished parsing notes")
        self._parsed_notes = parsed_notes


class MarkdownThenRecursiveSplit:
    def __init__(
        self, headers_to_split_on=None, chunk_size=1024, chunk_overlap=64
    ) -> None:
        # Split an all levels of valid markdown headers
        headers_to_split_on = (
            [
                ("#", "Header 1"),
                ("##", "Header 2"),
                # ("###", "Header 3"),
                # ("####", "Header 4"),
                # ("#####", "Header 5"),
                # ("######", "Header 6"),
            ]
            if not headers_to_split_on
            else headers_to_split_on
        )

        # MD splits
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_text(self, markdown_document: str) -> List[Document]:
        md_header_splits = self.markdown_splitter.split_text(markdown_document)
        return self.char_splitter.split_documents(md_header_splits)


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    # vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    splitter = MarkdownThenRecursiveSplit()

    vault = ObsidianVault(vault_path=vault_path, vault_name="TEST_VAULT")
    vault.parse_obsidian_vault(splitter=splitter)
    parsed_notes = vault.parsed_notes
