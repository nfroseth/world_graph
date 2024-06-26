import logging
import os
from pprint import pprint
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tqdm import tqdm

# from unstructured.partition.auto import partition_pdf
from unstructured.partition.pdf import partition_pdf

from perf import timing


PDF_DIR = "pdf_note_output/unstructured_version"
LLM_DIR = "pdf_note_output/llm_pass_version"
# IMAGE_PATH = "pdf_note_output/unstructured_extracted_images"
IMAGE_PATH = "/home/xoph/repos/github/nfroseth/world_graph/test_files/pdf_note_output/unstructured_extracted_images"

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

        write_out_path = (
            str(path).replace(base_inpath.name, PDF_DIR).replace(".pdf", ".txt")
        )
        with open(write_out_path, mode="w", encoding="utf-8") as file:
            file.write(files[path])

    return files

@timing
def read_pdf(path):
    elements = partition_pdf(
        path,
        strategy="hi_res",
        infer_table_structure=True,
        # extract_images_in_pdf=True,
        # extract_image_block_output_dir=IMAGE_PATH,
        # extract_image_block_types=["Image", "Table"],
        # extract_image_block_to_payload=False,
    )
    return "\n\n".join([str(el) for el in elements])

def get_and_write_llm(chain, files, path):
    base_inpath = Path(path)

    logging.info(f"Sending Notes to LLM...")
    llm_processed_files = {}
    for path, document in tqdm(files.items()):
        try:
            write_out_path = (
                str(path).replace(base_inpath.name, LLM_DIR).replace(".pdf", ".md")
            )
            llm_processed_files[write_out_path] = get_llm_response(
                chain, document
            ).content

            with open(write_out_path, mode="a+", encoding="utf-8") as file:
                file.write(llm_processed_files[write_out_path])
                print(f"Wrote file of len {len(llm_processed_files[write_out_path])} to {write_out_path}")
        except Exception as e:
            logging.critical(f"Failed Local LLM Call on file: {path} with {e}")

    return llm_processed_files

@timing
def get_llm_response(chain, context):
    return chain.invoke(context)


def chunk_text(splitter, files):
    after_chunking = {}
    for path, file in files.items():
        file = files[path]
        for idx, chunk in enumerate(splitter.split_text(file)):
            chunk_path = str(path).replace(".pdf", f"_chunk_{idx}.pdf")
            after_chunking[chunk_path] = chunk

    return after_chunking


def main():
    llm = ChatOpenAI(
        openai_api_key="lm-studio",
        openai_api_base="http://172.30.128.1:1234/v1",
        model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
        temperature=0.8,
    )

    template = """
    System: You are an expert in generating markdown documents. Format the following text in markdown with consistent headers, lists, and code blocks.
    Make sure lists are separated and correctly formatted. You will be given chunked components of the document.
    Without changing any of the text only provide well formatted markdown output. Do not output anything else.

    ---

    Note: {context}
    
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=512,
        length_function=len,
        is_separator_regex=False,
    )
    prompt = PromptTemplate(template=template, input_variables=["context"])
    chain = prompt | llm

    path = "/home/xoph/repos/github/nfroseth/world_graph/test_files/pdf_note_input"

    # files = parse_all_pdf_dir(path)

    after_pdf = "/home/xoph/repos/github/nfroseth/world_graph/test_files/pdf_note_output/unstructured_version"
    llm_so_far = "/home/xoph/repos/github/nfroseth/world_graph/test_files/pdf_note_output/llm_pass_version"

    notes_to_do = list(Path(after_pdf).rglob("*" + ".txt"))
    print(f"PDF files: {len(notes_to_do)}")

    notes_to_do = [(path_it.stem, path_it) for path_it in notes_to_do]

    chunks_done = Path(llm_so_far).rglob("*" + ".md")

    files_done = []
    for full_path_it in chunks_done:
        path_it = full_path_it.stem
        index = len(path_it) - path_it[::-1].find('_', path_it[::-1].find('_') + 1) - 1
        files_done.append(path_it[:index])
    
    chunks_to_do = [path for stem, path in notes_to_do if stem not in files_done]
    print(f"Files remaining: {len(chunks_to_do)}")
    # print(chunks_to_do)

    files = {}
    for path_it in chunks_to_do:
        with open(path_it, mode="r", encoding="utf-8") as file:
            files[path_it] = file.read()

    chunks = chunk_text(text_splitter, files)
    print(f"Chunks files: {len(chunks)}")

    # print(f"files: {files.keys()}")
    # print("---")
    # print(get_llm_response(llm, template, context).content)

    after_llm = get_and_write_llm(chain, chunks, path)
    print(f"LLM files: {len(after_llm)}")
    print(f"LLM files: {after_llm.keys()}")


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())


# MIN_LENGTH=10000 METADATA_FILE=../pdf_meta.json NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
