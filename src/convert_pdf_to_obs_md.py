import logging
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

from perf import timing

pdf_log = logging.getLogger(__name__)
pdf_log.setLevel(logging.DEBUG)
pdf_log.addHandler(logging.StreamHandler())

def get_llm_chain():
    llm = ChatOpenAI(
        openai_api_key="lm-studio",
        openai_api_base="http://172.30.128.1:1234/v1",
        model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
        temperature=0.6,
        max_tokens=8192,
    )
    # template = """
    # System: You are an expert in generating markdown documents from hand written notes. Format the following text in markdown with consistent headers, lists, and tables. 
    # The input text will be incoherent, only output readable/correct english words and sentences.
    # ---
    # Note: {context}
    # """

    template = """
    System: You are an expert in generating markdown documents. Format the following text in markdown with consistent headers, lists, and tables. 
    Correct any lists out of order and any table which needs to be converted to markdown format. Do not output anything else before or after.
    Add double square brackets around any people or place names. Example [[Person]] or [[Place]]
    ---
    Note: {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["context"])
    chain = prompt | llm
    return chain

@timing
def sent_note_to_llm(chain, context: str, llm_out_path: Path, splitter = None):
    llm_text = ""
    context_list = [context]

    if splitter is not None:
        context_list = splitter.split_text(context)

    
    for idx, chunk in enumerate(context_list):
        if idx == 0:
            chunk_boundary = ""
        else:
            chunk_boundary = f"# chunk_boundary {idx}\n" 
        llm_text += chunk_boundary + get_llm_response(chain, chunk).content
        
    with open(llm_out_path, mode="a", encoding="utf-8") as file:
        file.write(llm_text)

    return llm_text


@timing
def get_llm_response(chain, context):
    return chain.invoke(context)

@timing
def read_pdf(input_path, image_output_path, destination_path):
    #
    elements = partition_pdf(
        input_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(image_output_path),
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
    )

    after_partitioning = "\n\n".join([str(el) for el in elements])
    with open(destination_path, mode="w", encoding="utf-8") as file:
        file.write(after_partitioning)
        pdf_log.debug(
            f"Wrote file of len {len(after_partitioning)} to {destination_path}"
        )

    return after_partitioning


def parse_pdf_notes_to_md(input_pdf_dir: Path):
    llm_chain = get_llm_chain()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=512,
        length_function=len,
        is_separator_regex=False,
    )

    after_pdf_dir_name = "text_conversion"
    pdf_content_dir_name = "extracted_content"
    after_llm_dir_name = "llm_output"

    note_ext_type = ".pdf"

    note_tree = [note_path for note_path in input_pdf_dir.rglob("*" + note_ext_type)]
    pdf_log.info(f"Found {len(note_tree)} {note_ext_type} notes in {input_pdf_dir}")

    pdf_log.debug(f"Creating {after_llm_dir_name} dir...")
    after_pdf_path = input_pdf_dir.parent.joinpath(after_pdf_dir_name)
    after_pdf_path.mkdir(parents=True, exist_ok=True)

    post_pdf_processing_ext_type = ".txt"
    pdf_processed_so_far = [
        pdf_path
        for pdf_path in Path(after_pdf_path).rglob("*" + post_pdf_processing_ext_type)
    ]
    # TODO: Handling duplicate file names use the path they came from as the key
    pdf_processed_so_far_names = [p_it.stem for p_it in pdf_processed_so_far]
    pdf_log.info(f"Files found in {after_pdf_dir_name} {len(pdf_processed_so_far)}")

    pdf_to_be_processed = [
        note_path
        for note_path in note_tree
        if note_path.stem not in pdf_processed_so_far_names
    ]
    pdf_log.info(f"{len(pdf_to_be_processed)} pdfs to be processed. Starting now... ")

    pdf_content_path = input_pdf_dir.parent.joinpath(pdf_content_dir_name)
    pdf_content_path.mkdir(parents=True, exist_ok=True)

    converted_pdfs = {}
    llm_output = {}
    for pdf_path in tqdm(pdf_to_be_processed):
        input_str = str(pdf_path)[: -len("".join(pdf_path.suffixes))]
        content_path = Path(input_str.replace(input_pdf_dir.name, pdf_content_dir_name))
        content_path.mkdir(parents=True, exist_ok=True)

        destination_path = Path(
            input_str.replace(input_pdf_dir.name, after_pdf_dir_name) + ".txt"
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        converted_pdfs[pdf_path] = read_pdf(pdf_path, content_path, destination_path)

        llm_response_path = Path(input_str.replace(input_pdf_dir.name, after_llm_dir_name)+".md")
        llm_response_path.parent.mkdir(parents=True, exist_ok=True)

        llm_response = sent_note_to_llm(llm_chain, converted_pdfs[pdf_path], llm_response_path, text_splitter)
        llm_output[pdf_path] = llm_response

    pdf_log.info(f"Total number of pdfs read and converted {len(llm_output)}")


def main():
    input_path = Path("/home/xoph/repos/github/nfroseth/world_graph/test_files/input_pdfs")
    parse_pdf_notes_to_md(input_path)


"""
Project Structure Summary
    input_pdfs/: Store original PDF files.
    text_conversion/: Convert PDFs to text documents.
    extracted_content/:
        tables/: Store extracted tables.
        images/: Store extracted images.
    final_output/: Store the final markdown document.

Workflow
    Place PDFs in input_pdfs/.
    Convert PDFs to text, save in text_conversion/.
    Extract tables/images, save in extracted_content/.
    Combine content into markdown, save in final_output/.
"""

if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())
