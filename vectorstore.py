from pymilvus import MilvusClient
from typing import List

from embeddings import create_embeddings

milvus_client = MilvusClient(uri="./data/milvus_football_rag.db")

collection_name = "football_chatbot"


def setup_datastore():
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=1536,
        metric_type="IP",
        consistency_level="Strong",
    )


def insert_messages(messages: List[str]):
    data = []

    for i, message in enumerate(messages):
        data.append({"id": i, "vector": create_embeddings(message), "text": message})

    milvus_client.insert(collection_name=collection_name, data=data)


def search_messages(query: str) -> List[str]:
    search_resp = milvus_client.search(
        collection_name=collection_name,
        data=[
            create_embeddings(query)
        ],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )
    return [
        res["entity"]["text"] for res in search_resp[0]
    ]
