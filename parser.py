from collections import defaultdict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

SENDER_RE = r'\[\d{1,2}\/\d{1,2}\/\d{2},\s\d{1,2}:\d{2}:\d{2}.[AP]M\]\s(\w+\s\w+):\s(.+)'


def parse_messages_by_sender() -> dict:
    messages = defaultdict(list)
    last_sender = ''
    with open('data/_chat.txt', 'r') as file:
        for line in file:
            if 'image omitted' not in line:
                if match := re.match(SENDER_RE, line):
                    messages[match.group(1)].append(match.group(2))
                    last_sender = match.group(1)
                elif len(messages) > 0:
                    messages[last_sender][-1] += line
    return messages


def parse_all_messages() -> str:
    message_blob = ''
    with open('data/_chat.txt', 'r') as file:
        for line in file:
            if match := re.match(SENDER_RE, line):
                message_blob += f'{match.group(2)}\n'
            elif message_blob:
                message_blob += f'{line}\n'

    return message_blob


def split_messages(messages: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_text(messages)
