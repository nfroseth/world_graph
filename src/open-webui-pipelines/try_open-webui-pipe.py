from typing import List

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from local_chat_with_vault import setup, get_rerank, related_notes, graph_expansion, wrap_search

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

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

CONTEXT:
{context}

POST:
{post}

POSTER’S EXPERTISE: {expertise_level}

Beginners want detailed answers with explanations. Experts want concise answers without explanations.
If you are unable to help the reviewer, let them know that help is on the way.
"""

class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "pipeline_example"

        # The name of the pipeline.
        self.name = "Obsidian Graph-RAG"

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        self.store, self.URI, self.AUTH = setup()

        llm = ChatOllama(model="qwen2:7b-instruct-q6_K")

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "post",
                "context",
                # "conversation_history",
                # "poster",
                "expertise_level",
            ],
        )
        parser = StrOutputParser()
        llm_chain = prompt | llm | parser
        self.llm = llm_chain

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    # async def on_valves_updated(self):
    #     # This function is called when the valves are updated.
    #     pass

    # async def inlet(self, body: dict, user: dict) -> dict:
    #     # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
    #     print(f"inlet:{__name__}")
    #     print(body)
    #     print(user)
    #     return body

    # async def outlet(self, body: dict, user: dict) -> dict:
    #     # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
    #     print(f"outlet:{__name__}")
    #     print(body)
    #     print(user)
    #     return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator[str, None, None]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")

        docs_with_score = wrap_search(self.store, user_message, 1000)

        docs = [doc.page_content for doc, score in docs_with_score]

        re_ranked_scores = get_rerank(user_message, docs)
        re_ranked_index = sorted(
            re_ranked_scores, key=lambda x: x["relevance_score"], reverse=True
        )
        top_k = 75
        re_ranked_docs = [docs[row["index"]] for row in re_ranked_index[:top_k]]
        re_ranked_doc_names = [
            docs_with_score[row["index"]][0].metadata["name"]
            for row in re_ranked_index[:top_k]
        ]

        depth = 5
        after_graph = (
            graph_expansion(self.URI, self.AUTH, re_ranked_doc_names, depth)
            + re_ranked_docs
        )
        print(f"From {len(re_ranked_doc_names)} passages to {len(after_graph)=}")

        re_ranked_scores = get_rerank(user_message, after_graph)
        re_ranked_index = sorted(
            re_ranked_scores, key=lambda x: x["relevance_score"], reverse=True
        )
        top_k = 15
        re_ranked_docs = [after_graph[row["index"]] for row in re_ranked_index[:top_k]]

        context = "\n ".join(re_ranked_docs)

        llm_input = {
            "context": context,
            "post": user_message,
            "expertise_level": "expert",
        }
        print(f'Context Length: {len(llm_input["context"])}')


        def add_sources(llm):
            msg = "Start: From the following Notes: [1]... \n\n"
            yield msg
            yield from llm.stream(llm_input)
            msg = "End: From the following Notes: [2]... Thank you!"
            yield msg

        # print(messages)
        # print(user_message)
        # print(body)
        # llm_results = self.llm_chain.invoke(llm_input)
        # print(user_message)
        # print(body)

        return add_sources(self.llm)