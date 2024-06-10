from to_neo4j import get_tags_from_line 

examples = [
    "#/",
    "-#Test",
    "#test\sers",
    "#person#family This is an example tag ",
    "This #purchase-test is an example tag #person/family",
    "-#Test #123dy",
    "_#under",
    "/#slash",
    "?#quest",
    "alpha#a",
    "##Test #123dy",
    "This #1/234 is an example tag #1/2",
    "#123 This is an example tag",
    "This is an example tag #123",
    "#yes This is an example #yes tag",
    "#yes/no This is an example #yes tag",
    "This is an example tag #person/#family",
    "This is an example tag #person/family",
    "This #1/2 is an example tag #123",
    "This is an example tag #person/family This is an example tag #person/family",
    "This is an example tag #2022/03/28",
    "## Test",
    "#1234y-sdf|sdfsd "
]
for example in examples:
    # tags = get_tags_from_line_smd(example)
    tags = get_tags_from_line(example)
    print(f"{tags=}")
