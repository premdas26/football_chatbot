from embeddings import openai_client
from vectorstore import MilvusStore

SYSTEM_PROMPT = """
You are a football consultant answering questions for fans. You are able to find answers to the questions from the contextual passage snippets provided, but answer the questions as if the context is your own opinion. 
"""

class OpenAIChatbot:
    def __init__(self, vector_store: str):
        self.vector_store = MilvusStore(name=vector_store)

    def get_user_prompt(self, query: str):
        context = "\n".join(self.vector_store.search_messages(query))
        return f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {query}
        </question>
        """

    def get_answer(self, query: str):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.get_user_prompt(query)},
            ],
        )
        return response.choices[0].message.content

    def chat(self, message, history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": self.get_user_prompt(message)}]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return response.choices[0].message.content