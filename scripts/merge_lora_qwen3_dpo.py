import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = Path(os.environ.get("LEXI_MERGE_BASE", "/mnt/data/models/Qwen/Qwen3-30B-A3B-Instruct-2507"))
ADAPTER = Path(os.environ.get("LEXI_MERGE_ADAPTER", "/mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-final"))
OUT = Path(os.environ.get("LEXI_MERGE_OUT", "/mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged"))

print(f"Base: {BASE}\nAdapter: {ADAPTER}\nOut: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, use_fast=False)
print("Loaded tokenizer")

print("Loading base model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)
print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
print("Merging and unloading adapter...")
model = model.merge_and_unload()
print("Saving merged model...")
model.save_pretrained(OUT, safe_serialization=True)
tokenizer.save_pretrained(OUT)
print("Done.")
