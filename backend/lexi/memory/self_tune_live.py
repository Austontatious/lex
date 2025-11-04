#!/usr/bin/env python3
"""
FRIDAY Self-Tuning Script — Live LoRA Adapter Generator
- Scans finalized memory threads
- Builds fine-tune-ready dataset
- Runs LoRA-based continual tuning
- Outputs to `loftq_live/` and logs results
"""
import os
import time
import json
from datetime import datetime
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
import torch

# ─── Config ──────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "friday/memory/default.jsonl"
OUTPUT_DIR = "models/friday_0.2.2/loftq_live"
MAX_TRAINING_SHARDS = 200  # cap to avoid drift


# ─── Build Fine-tune Dataset from Memory ─────────────────────────────────
def load_memory(file_path):
    samples = []
    seen_threads = set()
    with open(file_path, "r") as f:
        for line in f:
            shard = json.loads(line)
            if "tags" in shard and any(
                tag in shard["tags"] for tag in ["persona_shift", "symbolic", "novelty"]
            ):
                samples.append(
                    {
                        "messages": [
                            {"role": "user", "content": "What were we talking about?"},
                            {"role": "assistant", "content": shard["text"]},
                        ]
                    }
                )

            seen_threads.add(shard["thread_id"])
            samples.append(
                {
                    "messages": [
                        {"role": "user", "content": "What were we talking about?"},
                        {"role": "assistant", "content": shard["text"]},
                    ]
                }
            )
            if len(samples) >= MAX_TRAINING_SHARDS:
                break
    return samples


# ─── Save as JSONL Dataset ───────────────────────────────────────────────
def write_jsonl(data, out_path):
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# ─── Main Tuning Routine ─────────────────────────────────────────────────
def run_self_tune():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(OUTPUT_DIR) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Extracting memory threads...")
    samples = load_memory(DATA_PATH)
    dataset_path = run_dir / "memory_batch.jsonl"
    write_jsonl(samples, dataset_path)

    print("📦 Loading dataset...")
    raw_ds = load_dataset("json", data_files=str(dataset_path))["train"]

    print("🔧 Preparing model + LoRA config...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def tokenize(example):
        return tokenizer.apply_chat_template(
            example["messages"], return_tensors="pt", truncation=True, max_length=2048
        )

    tokenized_ds = raw_ds.map(tokenize)

    print("🚀 Starting training...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds,
        args=TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            fp16=True,
            save_strategy="no",
            logging_steps=10,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    print("💾 Saving adapter weights...")
    model.save_pretrained(run_dir / "adapter")
    print(f"✅ Self-tuning complete: {run_dir}")


if __name__ == "__main__":
    run_self_tune()
