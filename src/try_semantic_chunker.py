from parse_obsidian_vault import ObsidianVault, MarkdownThenRecursiveSplit
from pprint import pprint
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import InfinityEmbeddings, InfinityEmbeddingsLocal

import pandas as pd

infinity_api_url = "http://127.0.0.1:7997"

embeddings = InfinityEmbeddings(
    model="BAAI/bge-small-en-v1.5", infinity_api_url=infinity_api_url
)

vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
# vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
percentile_semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
vault_percent = ObsidianVault(vault_path=vault_path, vault_name="TEST_VAULT")
vault_percent.parse_obsidian_vault(splitter = percentile_semantic_chunker)
parsed_notes_percent = vault_percent.parsed_notes

print("Done!")