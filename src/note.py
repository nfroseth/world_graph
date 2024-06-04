import logging
import os
from typing import List


class Relationship:
    def __init__(self, type: str, properties={}):
        self.type = type
        self.properties = properties

    def __str__(self):
        return self.type + self.properties.__str__()


class Note:

    def __init__(
        self,
        name: str,
        tags: List[str],
        content: str,
        properties={},
        out_relationships={},
        in_relationships={},
    ):
        self.tags = tags
        self.out_relationships = out_relationships
        self.in_relationships = in_relationships
        self.properties = properties
        self.properties["name"] = name
        self.properties["content"] = content

    def add_out_relationship(self, to: str, relationship: Relationship):
        if to in self.out_relationships:
            self.out_relationships[to].append(relationship)
        else:
            self.out_relationships[to] = [relationship]

    def add_in_relationship(self, src: str, relationship: Relationship):
        if src in self.in_relationships:
            self.in_relationships[src].append(relationship)
        else:
            self.in_relationships[src] = [relationship]

    @property
    def name(self):
        return self.properties["name"]

    @property
    def content(self):
        return self.properties["content"]

    def __str__(self):
        props = [
            self.name,
            [str(tag) for tag in self.tags],
            f"{len(self.content)=} {self.content[:10]}...",
            [
                f"Name: {name},  Links: {list(map(print, rels))}"
                for name, rels in self.out_relationships.items()
            ],
        ]
        out_string = ""
        for prop in props:
            try:
                out_string += f"{str(prop)} "
            except Exception as e:
                logging.warning("Failed to print some properties.")
        return out_string
