import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

template = """
You are a helpful AI assistant. Provide the answer for the following question:

Question: {question}
Answer:
"""

llm = ChatOpenAI(
    openai_api_key="lm-studio",
    openai_api_base="http://172.30.128.1:1234/v1",
    model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q5_1.gguf",
    temperature=0,
)

@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    cl.user_session.set("llm_chain", llm_chain)

    return llm_chain

@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    res = await llm_chain.acall(message.content, callbacks=[cb])


    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()