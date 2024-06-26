import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# template = """
# You are a helpful AI assistant. Provide the answer for the following question:

# Question: {question}
# Answer:
# """

template = """
You are a customer support agent, helping posters by following directives and answering questions.
Generate your response by following the steps below:
1. Recursively break-down the post into smaller questions/directives
2. For each atomic question/directive:
2a. Select the most relevant information from the context in light of the conversation history
3. Generate a draft response using the selected information, whose brevity/detail are tailored to the poster’s expertise
4. Remove duplicate content from the draft response
5. Generate your final response after adjusting it to increase accuracy and relevance
6. Now only show your final response! Do not provide any explanations or details

POST:
{question}
If you are unable to help the reviewer, let them know that help is on the way.

### Instruction:
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