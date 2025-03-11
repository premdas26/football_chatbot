from pymilvus import MilvusClient
from typing import List

from embeddings import create_embeddings
from tqdm import tqdm

milvus_client = MilvusClient(uri="./data/milvus_football_rag.db")


class MilvusStore:
    def __init__(self, name: str):
        self.name = name
        self.setup_datastore()

    def setup_datastore(self):
        if milvus_client.has_collection(self.name):
            milvus_client.drop_collection(self.name)

        milvus_client.create_collection(
            collection_name=self.name,
            dimension=1536,
            metric_type="IP",
            consistency_level="Strong",
        )

    def insert_messages(self, messages: List[str]):
        data = []

        for i, message in enumerate(tqdm(messages)):
            data.append({"id": i, "vector": create_embeddings(message), "text": message})

        milvus_client.insert(collection_name=self.name, data=data)

    def search_messages(self, query: str) -> List[str]:
        search_resp = milvus_client.search(
            collection_name=self.name,
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
