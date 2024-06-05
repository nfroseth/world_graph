
from langchain_huggingface import HuggingFaceEmbeddings
from to_neo4j import parse_vault
from transformers import AutoModel
# from sentence_transformers.util import semantic_search

model_name = "Alibaba-NLP/gte-large-en-v1.5"

if __name__ == "__main__":
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True) 
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
    # vault_path = "/home/xoph/SlipBoxCopy/Slip Box"
    vault_path = "/home/xoph/repos/github/nfroseth/world_graph/test_vault"
    notes = parse_vault(vault_path)