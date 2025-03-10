from embeddings import openai_client
from vectorstore import search_messages

SYSTEM_PROMPT = """
You are a football consultant answering questions for fans. You are able to find answers to the questions from the contextual passage snippets provided, but answer the questions as if the context is your own opinion. 
"""


def get_user_prompt(query: str):
    context = "\n".join(search_messages(query))
    return f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

def get_answer(query: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt(query)},
        ],
    )
    return response.choices[0].message.content