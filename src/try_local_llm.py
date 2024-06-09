import logging
import os
from pprint import pprint
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from tqdm import tqdm
from unstructured.partition.auto import partition

from get_embedding import timing


PDF_DIR = "pdf_note_output/unstructured_version"
LLM_DIR = "pdf_note_output/llm_pass_version"

def note_name(path, extension=".pdf"):
    return os.path.basename(path)[: -len(extension)]

@timing
def parse_all_pdf_dir(path):
    note_ext_type = ".pdf"

    base_inpath = Path(path)
    note_tree = base_inpath.rglob("*" + note_ext_type)
    logging.info(f"Parsing PDF Notes")
    files = {}
    for path in tqdm(note_tree):
        # name = note_name(path)
        files[path] = read_pdf(path)

        write_out_path = str(path).replace(base_inpath.name, PDF_DIR).replace(".pdf", ".txt")
        with open(write_out_path, mode="w", encoding="utf-8") as file:
            file.write(files[path])

    return files

def get_and_write_llm(llm, template, files, path):
    base_inpath = Path(path)

    logging.info(f"Sending Notes to LLM...")
    llm_processed_files = {}
    for path, document in tqdm(files.items()):
        try:
            write_out_path = str(path).replace(base_inpath.name, LLM_DIR).replace(".pdf", ".md")
            llm_processed_files[write_out_path] = get_llm_response(llm, template, document).content

            with open(write_out_path, mode="w+", encoding="utf-8") as file:
                file.write(llm_processed_files[write_out_path])
        except Exception as e:
            logging.critical(f"Failed Local LLM Call on file: {path} with {e}")

    return llm_processed_files



@timing
def read_pdf(path):
    elements = partition(path)
    return "\n\n".join([str(el) for el in elements])

@timing
def get_llm_response(llm, template, context):
    prompt = PromptTemplate(template=template, input_variables=["context"])
    chain = prompt | llm
    return chain.invoke(context)

def main():
    llm = ChatOpenAI(
            openai_api_key="lm-studio",
            openai_api_base="http://172.30.128.1:1234/v1",
            model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
        )

    template = """
    System: You are a office assistant that uses unstructured text notes from pdf form convert the provided context into an Obsidian markdown document.
    Make sure lists are separated and correctly formatted.
    Without changing any of the text only provide well formatted markdown output. Do not output anything else.

    ---

    Note: {context}
    
    """

    path = "/home/xoph/repos/github/nfroseth/world_graph/test_files/pdf_note_input"

    files = parse_all_pdf_dir(path)

    print(f"PDF files: {len(files)}")
    # print(f"files: {files.keys()}")
    # print("---")
    # print(get_llm_response(llm, template, context).content)
    after_llm = get_and_write_llm(llm, template, files, path)
    print(f"LLM files: {len(after_llm )}")
    print(f"LLM files: {after_llm.keys()}")



if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())


# MIN_LENGTH=10000 METADATA_FILE=../pdf_meta.json NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out