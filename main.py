import re
import json
from collections import defaultdict

from llm import get_answer

SENDER_RE = r'\[\d{1,2}\/\d{1,2}\/\d{2},\s\d{1,2}:\d{2}:\d{2}.[AP]M\]\s(\w+\s\w+):\s(.+)'


def parse_data() -> dict:
    messages = defaultdict(list)
    last_sender = ''
    with open('data/_chat.txt', 'r') as file:
        for line in file:
            if match := re.match(SENDER_RE, line):
                messages[match.group(1)].append(match.group(2))
                last_sender = match.group(1)
            elif len(messages) > 0:
                messages[last_sender][-1] += line
    return messages


def write_message_data():
    messages = parse_data()
    for sender in messages:
        print(f'{sender}: {len(messages[sender])}')
    with open('data/chat.json', 'w') as f:
        json.dump(messages, f, indent=4)


if __name__ == '__main__':
    query = input('Ask Premal a football question: \n')
    print("\n")
    print(get_answer(query))


