from parser import parse_all_messages, split_messages
from vectorstore import MilvusStore

if __name__ == '__main__':
    message_chunks = split_messages(parse_all_messages())
    vector_store = MilvusStore(name="all_messages")
    vector_store.insert_messages(message_chunks)
