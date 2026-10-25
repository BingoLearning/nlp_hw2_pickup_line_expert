import json

from chat import PickupLineExpert
from config import RESULTS_DIR


def main():
    bot = PickupLineExpert()
    prompts = [
        "教我如何透過長相自然地認識一位女生。",
        "如果她問長得像誰呢？",
    ]
    transcript = []
    for prompt in prompts:
        answer = bot.reply(prompt)
        transcript.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
        print(f"你：{prompt}\n專家：{answer}\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "multiturn_transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

