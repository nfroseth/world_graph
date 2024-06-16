import logging
from typing import List
from neomodel import (
    StructuredRel,
    StringProperty,
    JSONProperty,
    IntegerProperty,
    ArrayProperty,
    RelationshipTo,
    RelationshipFrom,
    UniqueIdProperty,
)
from neomodel.contrib import SemiStructuredNode


class Link:
    def __init__(self, type: str, context:str, properties={}):
        self.type = type
        self.context = context
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
        self.out_relationships = out_relationships
        self.in_relationships = in_relationships
        self.properties = properties
        self.properties["name"] = name
        self.properties["content"] = content
        self.properties["tags"] = tags

    def add_out_relationship(self, to: str, relationship: Link):
        if to in self.out_relationships:
            self.out_relationships[to].append(relationship)
        else:
            self.out_relationships[to] = [relationship]

    def add_in_relationship(self, src: str, relationship: Link):
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

    @property
    def tags(self):
        return self.properties["tags"]

    @property
    def has_tags(self) -> bool:
        assert isinstance(self.tags, list) and all(
            [isinstance(tag, str) for tag in self.tags]
        ), "Tags are not a list or a tag is not a string"
        return len(self.tags) > 0

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


class Relationship(StructuredRel):
    relationship_type = StringProperty(default="inline")
    context = StringProperty(required=True)
    header = StringProperty(required=True)
    chunk_index = IntegerProperty(required=True)
    parsed_context = StringProperty(required=True)
    link_display_text = StringProperty(required=True)

class Chunk(SemiStructuredNode):
    chunk_index = IntegerProperty(required=True)
    content = StringProperty(required=True)
    next = RelationshipTo("Chunk", "NEXT_CHUNK")
    # prev = RelationshipTo("Chunk", "PREVIOUS")

    out_chunks = RelationshipTo("Chunk", "RELATED_TO", model=Relationship)
    in_chunks = RelationshipFrom("Chunk", "RELATED_TO", model=Relationship)

    out_relationships = RelationshipTo("Node", "RELATED_TO", model=Relationship)
    in_relationships = RelationshipFrom("Node", "RELATED_TO", model=Relationship)

class Node(SemiStructuredNode):
    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    tags = ArrayProperty(StringProperty())
    content = StringProperty(required=True)

    contains = RelationshipTo("Chunk", "FIRST_CHUNK")
    contains = RelationshipTo("Chunk", "PART_OF")

    out_chunks = RelationshipTo("Chunk", "RELATED_TO", model=Relationship)
    in_chunks = RelationshipFrom("Chunk", "RELATED_TO", model=Relationship)

    out_relationships = RelationshipTo("Node", "RELATED_TO", model=Relationship)
    in_relationships = RelationshipFrom("Node", "RELATED_TO", model=Relationship)
