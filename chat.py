import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from config import OUTPUT_DIR, SYSTEM_PROMPT


class PickupLineExpert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            OUTPUT_DIR, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.messages = self.messages[:1]

    def reply(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        inputs = self.tokenizer.apply_chat_template(
            self.messages[-11:], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=96, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        reply = self.tokenizer.decode(output[0][length:], skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def main():
    bot = PickupLineExpert()
    print("土味情話專家已啟動；/reset 清除歷史，exit 結束。")
    while True:
        text = input("你：").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if text == "/reset":
            bot.reset()
            print("系統：已清除歷史。")
            continue
        if text:
            print("專家：" + bot.reply(text))


if __name__ == "__main__":
    main()

