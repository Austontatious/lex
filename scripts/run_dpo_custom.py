"""
Custom DPO launcher to reduce concurrency and batch size.
Uses HelpSteer3 comparisons and resumes from the SL checkpoint.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tinker_cookbook import model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import HelpSteer3ComparisonBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config() -> train_dpo.Config:
    base_model = os.environ.get("LEXI_DPO_BASE", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    renderer_name = os.environ.get(
        "LEXI_DPO_RENDERER", model_info.get_recommended_renderer_name(base_model)
    )
    log_path = os.environ.get(
        "LEXI_DPO_LOG", "/mnt/data/Lex/logs/dpo_qwen3_30b_a3b_v2"
    )
    load_path = os.environ.get(
        "LEXI_DPO_LOAD",
        "tinker://6f5f019b-f2fd-5f18-a05a-88467d2c4f32:train:0/weights/final",
    )

    batch_size = int(os.environ.get("LEXI_DPO_BATCH", "16"))
    max_length = int(os.environ.get("LEXI_DPO_MAXLEN", "4096"))
    lr = float(os.environ.get("LEXI_DPO_LR", "1e-5"))
    dpo_beta = float(os.environ.get("LEXI_DPO_BETA", "0.1"))
    lora_rank = int(os.environ.get("LEXI_DPO_RANK", "32"))
    num_epochs = int(os.environ.get("LEXI_DPO_EPOCHS", "1"))

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config, comparison_builder=HelpSteer3ComparisonBuilder()
    )

    return train_dpo.Config(
        log_path=log_path,
        model_name=base_model,
        dataset_builder=dataset_builder,
        load_checkpoint_path=load_path,
        learning_rate=lr,
        lr_schedule="linear",
        dpo_beta=dpo_beta,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        num_replicas=1,  # keep concurrency low to avoid sampler timeouts
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=50,
        eval_every=25,
    )


def main() -> None:
    cfg = build_config()
    print(
        f"[dpo] model={cfg.model_name} load={cfg.load_checkpoint_path} "
        f"log_path={cfg.log_path} batch={cfg.dataset_builder.common_config.batch_size} "
        f"lr={cfg.learning_rate} beta={cfg.dpo_beta} maxlen={cfg.dataset_builder.common_config.max_length}",
        flush=True,
    )
    asyncio.run(train_dpo.main(cfg))


if __name__ == "__main__":
    main()
