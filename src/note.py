import os
from typing import List


class relationshipationship:
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

    def add_out_relationship(self, to: str, relationship: relationshipationship):
        if to in self.out_relationships:
            self.out_relationships[to].append(relationship)
        else:
            self.out_relationships[to] = [relationship]

    def add_in_relationship(self, src: str, relationship: relationshipationship):
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
        return (
            self.name
            + os.linesep
            + self.tags.__str__()
            + os.linesep
            + self.content
            + os.linesep
            + self.out_relationships.__str__()
        )
