# world_graph

## Limitations

### Wiki Link in Header
in to_neo4j apply_label_to_node failed due to [[wiki link]] in header

Example:
---
aliases: []
tags: daily_journal [[Purple Notebook Source Tag]]
---
202405270846


A journal Entry for:
# 05-22-2023

Suggest the user does not put invalid cypher in the tags see 
https://neo4j.com/docs/cypher-manual/current/syntax/naming/

### Empty alias in Header

---
aliases: 
tags:
  - daily_journal
---
202405090915

node_from_note_and_fill_communities