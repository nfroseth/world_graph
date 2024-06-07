from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv

# template = """System: You are a knowledgeable assistant that uses structured data from knowledge graphs to answer questions accurately. Provide detailed and contextually relevant answers.
# ---
# History: {history}
# ---
# Context: {context}
# ---
# User: {query}"""

template = """System: You are a knowledgeable assistant that uses structured data from knowledge graphs to answer questions accurately. Provide detailed and contextually relevant answers.

---

Context: {context}

---

User: {query}"""

prompt = PromptTemplate(template=template, input_variables=["context", "query"])

llm = ChatOpenAI(
    openai_api_key=getenv("WORKOUT_ENGINE_OPEN_ROUTER_AI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1/",
    model="meta-llama/llama-3-8b-instruct:free",
)

llm_chain = prompt | llm 

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
context = """Justin Drew Bieber (/ˈbiːbər/ BEE-bər; born March 1, 1994)[1][2] is a Canadian singer. Regarded as a pop icon, he is recognized for his multi-genre musical performances.[3][4][5] He was discovered by American record executive Scooter Braun in 2008 and subsequently brought to American singer Usher, both of whom formed the record label RBMG Records to sign Bieber in October of that year. He gained recognition following the release of his debut extended play (EP) My World (2009), which was quickly met with international commercial success and led to his establishment as a prominent teen idol. """

llm_result = llm_chain.invoke([question, context])
print(llm_result.content)