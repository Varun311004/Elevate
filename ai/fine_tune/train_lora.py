"""Train a TinyLlama LoRA adapter for MCQ JSON generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def format_row(example: dict) -> str:
    prompt = str(example.get("prompt") or "").strip()
    completion = str(example.get("completion") or "").strip()
    return (
        "<|system|>\nYou are an expert educational AI that outputs valid MCQ JSON only.</s>\n"
        f"<|user|>\n{prompt}</s>\n"
        f"<|assistant|>\n{completion}</s>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL SFT dataset")
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize_fn(example: dict) -> dict:
        text = format_row(example)
        toks = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
