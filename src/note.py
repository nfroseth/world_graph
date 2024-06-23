import logging
from typing import Dict, List
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
from langchain_core.documents import Document


class Link:
    def __init__(self, type: str, properties={}):
        self.type = type
        self.properties = properties

    def __str__(self):
        return self.type + self.properties.__str__()


class Note_Chunk:
    def __init__(
        self,
        chunk_index: int,
        properties={},
        out_note_relationships={},
        in_note_relationships={},
        out_chunk_relationships={},
        in_chunk_relationships={},
    ):
        self.chunk_index = chunk_index
        self.properties = properties
        self.out_note_relationships = out_note_relationships
        self.in_note_relationships = in_note_relationships
        self.out_chunk_relationships = out_chunk_relationships
        self.in_chunk_relationships = in_chunk_relationships

    def __str__(self):
        return self.chunk_index + self.properties.__str__()


class Note:
    def __init__(
        self,
        name: str,
        tags: List[str],
        content: str,
        chunks: List[Note_Chunk],
        # chunks: List[Document],
        properties={},
        out_relationships={},
        in_relationships={},
        out_chunk_relationships={},
        in_chunk_relationships={},
    ):
        self.chunks = chunks
        self.out_relationships = out_relationships
        self.in_relationships = in_relationships
        self.out_chunk_relationships = out_chunk_relationships
        self.in_chunk_relationships = in_chunk_relationships
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

    def to_properties_from_node() -> Dict:
        pass

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
    title = StringProperty()
    context = StringProperty(required=True)
    header = StringProperty()
    chunk_index = IntegerProperty(required=True)
    parsed_context = StringProperty()
    link_display_text = StringProperty()


class Chunk(SemiStructuredNode):
    nick_name = StringProperty()
    chunk_index = IntegerProperty(required=True)
    content = StringProperty(required=True)
    metadata = JSONProperty(required=True)
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

    head = RelationshipTo("Chunk", "FIRST_CHUNK")
    contains = RelationshipTo("Chunk", "PART_OF")

    out_chunks = RelationshipTo("Chunk", "RELATED_TO", model=Relationship)
    in_chunks = RelationshipFrom("Chunk", "RELATED_TO", model=Relationship)

    out_relationships = RelationshipTo("Node", "RELATED_TO", model=Relationship)
    in_relationships = RelationshipFrom("Node", "RELATED_TO", model=Relationship)
