import io
import logging
import os
from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm
import yaml
from note import Note

parse_log = logging.getLogger(__name__)
parse_log.setLevel(logging.DEBUG)
parse_log.addHandler(logging.StreamHandler())


def typed_list_parse(
    file: io.TextIOWrapper, name, parsed_notes: Sequence[Note], args
) -> Note:
    line = file.readline()
    note_properties = None
    while line.strip() != "":
        if line == "---" + os.linesep:
            try:
                note_properties = parse_yaml_header(file)
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
    relations = {}
    tags = []
    while line:
        line = file.readline()
        for tag in get_tags_from_line(line):
            if tag not in tags:
                tags.append(tag)

    return tags


def parse_yaml_header(file):
    lines = []
    line = file.readline()
    while line != "---" + os.linesep and line:
        lines.append(line)
        line = file.readline()

    return yaml.safe_load("".join(lines))


def note_name(path, extension=".md"):
    return os.path.basename(path)[: -len(extension)]


def parse_vault(notes_path: str, note_ext_type: str = ".md", args=None):
    note_tree = Path(notes_path).rglob("*" + note_ext_type)

    parse_log.info(f"Parsing Notes")
    parsed_notes = {}
    for path in tqdm(note_tree):
        name = note_name(path)
        parse_log.debug(f"Reading note {name=}")
        with open(path, mode="r", encoding="utf-8") as file:
            tags = typed_list_parse(file, name, parsed_notes, args)
            for tag in tags:
                print(f"{tag=}")


def main():
    pass

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


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")

    # vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"

    parse_vault(vault_path)
