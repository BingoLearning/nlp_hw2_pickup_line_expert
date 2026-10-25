import json
import random

from config import CORPUS_FILE, SEED, TRAIN_FILE, VALID_FILE


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def expand(conversations):
    examples = []
    for conversation_id, row in enumerate(conversations):
        messages = row["messages"]
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                examples.append({
                    "conversation_id": conversation_id,
                    "prompt": messages[:index],
                    "completion": message["content"],
                })
    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare():
    conversations = read_jsonl(CORPUS_FILE)
    random.Random(SEED).shuffle(conversations)
    cut = max(1, round(len(conversations) * 0.9))
    train_conversations, valid_conversations = conversations[:cut], conversations[cut:]
    train, valid = expand(train_conversations), expand(valid_conversations)
    write_jsonl(TRAIN_FILE, train)
    write_jsonl(VALID_FILE, valid)
    print(f"Conversations train={len(train_conversations)}, valid={len(valid_conversations)}")
    print(f"SFT examples train={len(train)}, valid={len(valid)}")
    return train, valid


if __name__ == "__main__":
    prepare()

