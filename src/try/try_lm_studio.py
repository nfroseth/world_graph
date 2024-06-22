from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def get_llm_response(chain, context):
    return chain.invoke(context)

def main():
    llm = ChatOpenAI(
        openai_api_key="lm-studio",
        openai_api_base="http://172.30.128.1:1234/v1",
        model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
        temperature=0,
    )

    template = """
    System: You are an expert in generating markdown documents. Format the following text in markdown with consistent headers, lists, and code blocks.
    Make sure lists are separated and correctly formatted. You will be given chunked components of the document.
    Without changing any of the text only provide well formatted markdown output. Do not output anything else.

    ---

    Note: {context}
    
    """

    prompt = PromptTemplate(template=template, input_variables=["context"])
    chain = prompt | llm

    context = "Example Note"

    llm_response = get_llm_response(chain, context)
    print(f"{llm_response}")


if __name__ == "__main__":
    print("Quacks like a duck, looks like a goose.")
    exit(main())