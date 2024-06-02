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

PUNCTUATION = [
    "#",
    "$",
    "!",
    ".",
    ",",
    "?",
    # "/",
    ":",
    ";",
    "`",
    " ",
    "-",
    "+",
    "=",
    "|",
    os.linesep,
] + [str(i) for i in range(0, 10)]


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
        for tag in get_tags_from_line(line):
            if tag not in tags:
                tags.append(tag)
        # TODO: Save aliases as Relation property


# TODO: How are nested Tags handled?
def get_tags_from_line(line) -> List[str]:
    pos_tags = [i for i, char in enumerate(line) if char == "#"]
    tags = []
    for i in pos_tags:
        if i == 0 or line[i - 1] == " ":
            index = next(
                (index for index, c in enumerate(line[i + 1 :]) if c in PUNCTUATION), -1
            )
            if index == -1:
                tags.append(line[i + 1 :])
            else:
                tag = line[i + 1 : index + i + 1]
                if len(tag) > 0:
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
            typed_list_parse(file, name, parsed_notes, args)


def main():
    pass


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")

    vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    # vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"

    # parse_vault(vault_path)
    example = "This #purchase is an example tag #person/family"
    tags = get_tags_from_line(example)
    print(f"{tags=}")
