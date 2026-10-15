"""Produce small, explicit experiment grids without importing GPU dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def make_plan():
    defaults = dict(
        model_name="d8", learning_rate=0.003, num_train_sequences=600_000,
        batch_size=64, num_micro_batches=1, num_epochs=1.0,
        model_seed=42, data_seed=42, lr_schedule="linear",
        warmup_percent=0.01, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_norm=1.0, precision="mp", num_evals=100,
    )
    runs = [{
        "id": "a1-smoke", "group": "smoke", "purpose": "pipeline_only",
        "config": {**defaults, "num_train_sequences": 2048, "num_evals": 2,
                   "latest_checkpoint_frequency": None, "save_model": False},
    }]
    for lr in LEARNING_RATES:
        runs.append({
            "id": f"a1-lr-{lr:g}-s42", "group": "lr", "purpose": "A1-P1a",
            "config": {**defaults, "learning_rate": lr},
        })
    for seed in (43, 44):
        runs.append({
            "id": f"a1-baseline-s{seed}", "group": "repeat",
            "purpose": "pilot_combined_seed_variation_not_noise_decomposition",
            "config": {**defaults, "model_seed": seed, "data_seed": seed},
        })
    return {
        "version": 1, "assignment": "A1", "date_beijing": "2026-10-15",
        "upstream_commit": json.loads((ROOT / "sources.lock.json").read_text())["upstream"]["commit"],
        "context_length": 1024, "gpu_per_run": 1,
        "formal_input_tokens_per_run": 614_400_000,
        "formal_predicted_tokens_per_run": 613_800_000,
        "notes": [
            "Batch is sequences per optimizer step, not per microbatch.",
            "The smoke is a separate short recipe; its losses do not answer A1.",
            "Reserve three seeds for confirmation; this is not the full A1 variation study.",
            "Choose later paired sweeps after observing the first LR curve.",
        ],
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = make_plan()
    rendered = json.dumps(plan, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

