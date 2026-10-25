import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import MODEL_ID, OUTPUT_DIR, SEED, TRAIN_FILE, VALID_FILE


class CompletionCollator:
    def __init__(self, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        encoded = []
        for row in features:
            prefix = self.tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            answer = self.tokenizer(
                row["completion"] + self.tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            input_ids = (prefix + answer)[: self.max_length]
            prompt_length = min(len(prefix), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            encoded.append({"input_ids": input_ids, "labels": labels})
        max_len = max(len(row["input_ids"]) for row in encoded)
        input_ids, labels, masks = [], [], []
        for row in encoded:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            masks.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }


class PeftTrainer(Trainer):
    """Transformers 5 workaround: reload PEFT safetensors, not pytorch_model.bin."""

    def _load_best_model(self):
        checkpoint = Path(self.state.best_model_checkpoint)
        adapter_file = checkpoint / "adapter_model.safetensors"
        state_dict = load_file(str(adapter_file), device="cpu")
        set_peft_model_state_dict(self.model, state_dict, adapter_name="default")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 訓練需要 NVIDIA CUDA GPU")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.add_adapter(LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    ))
    model.config.use_cache = False
    data = load_dataset(
        "json", data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)}
    )
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=float(os.getenv("NLP_EPOCHS", "5")),
        per_device_train_batch_size=int(os.getenv("NLP_TRAIN_BATCH", "1")),
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=int(os.getenv("NLP_GRAD_ACCUM", "16")),
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = PeftTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=CompletionCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"GPU={torch.cuda.get_device_name(0)}")
    trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

