"""
Quick smoke test for Tinker LoRA training on a small Lexi sample.

Requires:
- TINKER_API_KEY in env (no default).
- uvx available (`~/.local/bin/uvx`) to resolve Tinker + tinker-cookbook deps.

Override via env:
- LEXI_TINKER_DATA: path to JSONL with {"messages": [...]}
- LEXI_TINKER_LOG: log/output directory
- LEXI_TINKER_BASE: base model name (e.g., Qwen/Qwen3-32B-MoE)
- LEXI_TINKER_LR: learning rate
- LEXI_TINKER_BATCH: batch size
- LEXI_TINKER_EPOCHS: epochs
- LEXI_TINKER_LORA_RANK: LoRA rank
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config() -> train.Config:
    data_path = Path(
        os.environ.get("LEXI_TINKER_DATA", "/mnt/data/Lex/tmp/tinker_full.jsonl")
    )
    log_path = os.environ.get(
        "LEXI_TINKER_LOG", "/mnt/data/Lex/logs/tinker_qwen3_30b_a3b"
    )
    model_name = os.environ.get(
        "LEXI_TINKER_BASE", "Qwen/Qwen3-30B-A3B-Instruct-2507"
    )

    # Renderer: allow explicit override, else try model_info, else fall back to qwen3.
    renderer_override = os.environ.get("LEXI_TINKER_RENDERER")
    renderer_name = (
        renderer_override
        or model_info.get_recommended_renderer_name(model_name)
        or "qwen3"
    )

    batch_size = int(os.environ.get("LEXI_TINKER_BATCH", "2"))
    lr = float(os.environ.get("LEXI_TINKER_LR", "2e-4"))
    num_epochs = int(os.environ.get("LEXI_TINKER_EPOCHS", "1"))
    lora_rank = int(os.environ.get("LEXI_TINKER_LORA_RANK", "32"))

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(data_path),
        test_size=4,  # tiny holdout just to verify eval path
        shuffle_seed=0,
    )

    return train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=lr,
        lr_schedule="linear",
        num_epochs=num_epochs,
        eval_every=2,
        save_every=4,
        lora_rank=lora_rank,
    )


def main() -> None:
    cfg = build_config()
    print(
        f"[tinker-smoke] model={cfg.model_name} "
        f"data={cfg.dataset_builder.file_path} "
        f"log_path={cfg.log_path} "
        f"lr={cfg.learning_rate} batch={cfg.dataset_builder.common_config.batch_size} "
        f"epochs={cfg.num_epochs} rank={cfg.lora_rank}",
        flush=True,
    )
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
