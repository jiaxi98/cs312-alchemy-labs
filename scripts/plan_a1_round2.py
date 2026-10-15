"""A1 controlled sweeps, scaling pilots, randomness and perturbation comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from plan_a1 import make_plan

ROOT = Path(__file__).resolve().parents[1]


def make_round2():
    initial = make_plan()
    default = next(r["config"] for r in initial["runs"] if r["id"] == "a1-lr-0.003-s42")
    runs, reused, seen = [], {}, {}
    initial_keys = {json.dumps(r["config"], sort_keys=True): r["id"]
                    for r in initial["runs"] if r["group"] != "smoke"}

    def add(group, purpose, label, changes, *, replica=None, priority=1):
        config = {**default, **changes}
        key = json.dumps(config, sort_keys=True)
        if replica is None and key in initial_keys:
            reused.setdefault(initial_keys[key], []).append(group)
            return
        identity = key + (f"/replica/{replica}" if replica is not None else "")
        if identity in seen:
            existing = seen[identity]
            if group not in existing["comparisons"]:
                existing["comparisons"].append(group)
            return
        number = len(runs) + 1
        row = dict(id=f"a1-r2-{number:03d}", group=group, purpose=purpose,
                   display_name=label, config=config, comparisons=[group], priority=priority)
        if replica is not None:
            row["replica"] = replica
            if replica.startswith("h20-"):
                row["required_host_index"] = int(replica.removeprefix("h20-")) - 1
        seen[identity] = row
        runs.append(row)

    # Batch 128/256 leave a smaller last batch. Upstream handles this with
    # num_micro_batches=1; do not drop rows or claim identical optimizer-step counts.
    for seed in (42, 43):
        seeds = dict(model_seed=seed, data_seed=seed)
        for value in (0, .001, .01, .05, .1, .2):
            add("p1-warmup", "P1a: sensitivity to warmup fraction", f"P1 warmup={value:g} · s={seed}",
                {**seeds, "warmup_percent": value})
        for value in (0, .01, .03, .1, .3, 1):
            add("p1-weight-decay", "P1a: sensitivity to masked weight decay", f"P1 wd={value:g} · s={seed}",
                {**seeds, "weight_decay": value})
        for batch in (16, 32, 64, 128, 256):
            for lr in (.0003, .001, .003, .006, .01, .03):
                add("p1-batch-lr", "P1b: LR and batch at fixed model and input tokens",
                    f"P1 batch×lr · b={batch} lr={lr:g} · s={seed}",
                    {**seeds, "batch_size": batch, "learning_rate": lr})
        for wd in (0, .01, .1, .3, 1):
            for lr in (.001, .003, .01):
                add("p1-wd-lr", "P1b: interaction between LR and weight decay",
                    f"P1 wd×lr · wd={wd:g} lr={lr:g} · s={seed}",
                    {**seeds, "weight_decay": wd, "learning_rate": lr})
        for warmup in (0, .01, .1, .2):
            for lr in (.001, .003, .01):
                add("p1-warmup-lr", "P1b: interaction between LR and warmup",
                    f"P1 warmup×lr · w={warmup:g} lr={lr:g} · s={seed}",
                    {**seeds, "warmup_percent": warmup, "learning_rate": lr})
        for schedule in ("linear", "cos", "constant", "wsd0.1", "wsd0.2", "wsd0.5"):
            for lr in (.001, .003, .01):
                add("p1-schedule", "P1c: each scheduler gets the same LR search",
                    f"P1 schedule · {schedule} lr={lr:g} · s={seed}",
                    {**seeds, "lr_schedule": schedule, "learning_rate": lr})

    for beta1 in (0.0, .5, .9, .95):
        for lr in (.001, .003, .01):
            add("p1-momentum", "P1c/P5: momentum, convergence and curve smoothness",
                f"P1 momentum · β1={beta1:g} lr={lr:g}", {"beta1": beta1, "learning_rate": lr})

    # Fit only these smaller scales. d9 is a genuine held-out scale. The existing
    # d8 baseline has already been seen, so it must not be called a blind forecast.
    recipes = [("linear", .003, 0), ("constant", .003, 0),
               ("linear", .003, .2), ("linear", .03, 0)]
    for depth in range(4, 8):
        for schedule, lr, dropout in recipes:
            for seed in (42, 43):
                add("p2-scaling-pilot", "P2a: d4-d7 fit; predict d9 before releasing larger models",
                    f"P2 d{depth} · {schedule} lr={lr:g} drop={dropout:g} · s={seed}",
                    dict(model_name=f"d{depth}", lr_schedule=schedule, learning_rate=lr,
                         dropout=dropout, model_seed=seed, data_seed=seed), priority=0)

    for seed in range(4, 14):
        for label, changes in (("init", dict(model_seed=seed)),
                               ("data", dict(data_seed=seed)),
                               ("joint", dict(model_seed=seed, data_seed=seed))):
            add("p3-" + label, "P3: isolate initialization vs data subset/order variability",
                f"P3 {label} · seed={seed}", changes, priority=0)
    for replica in range(8):
        add("p3-hardware", "P3: fixed seeds across different H20 hosts; not H100/A100 comparison",
            f"P3 same seeds · host repeat={replica + 1}", {},
            replica=f"h20-{replica + 1}", priority=0)
    for seed in range(4, 9):
        for label, changes in (("batch16", dict(batch_size=16)), ("lr0.009", dict(learning_rate=.009))):
            add("p3-recipe-variance", "P3c: compare variability after independently changing recipe",
                f"P3 variance · {label} · s={seed}", dict(model_seed=seed, data_seed=seed, **changes))

    for seed in (42, 43, 44, 45, 46, 47):
        for perturb in (False, True):
            add("p4-token", "P4a: deterministic baseline vs one changed token at row0,pos100",
                f"P4 token={'changed' if perturb else 'control'} · s={seed}",
                dict(model_seed=seed, data_seed=seed, deterministic=True, perturb_one_token=perturb),
                priority=0)

    # Round-robin thematic groups so the first 160 slots cover all questions.
    groups = {}
    for row in runs:
        groups.setdefault(row["group"], []).append(row)
    ordered = []
    while any(groups.values()):
        for values in groups.values():
            if values:
                ordered.append(values.pop(0))
    return dict(version=1, assignment="A1", batch="round2", date_beijing="2026-10-15",
        upstream_commit=initial["upstream_commit"], context_length=1024,
        input_tokens_per_run=614400000, gpu_per_run=1, max_parallel=160,
        reused_initial_runs=reused, runs=ordered,
        held_out=dict(depths=[8, 9], released=False,
            note="d8 default already observed; only d9 is blind. Release after smaller-scale predictions."),
        limitations=["constant scheduler ignores warmup in upstream; this is the course's literal recipe",
                    "P1 fixes parameter count × input tokens, not wall time or steps",
                    "batch128/256 has a smaller final batch; accumulated input-token axis is capped",
                    "hardware repeats all use H20; no claim of H100/A100 cross-architecture coverage",
                    "P4 tests only first-token perturbation, not the entire later-time intervention task",
                    "P5/P6 reuse all recorded curves and RMS diagnostics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/a1/round2.json")
    args = parser.parse_args()
    plan = make_round2()
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(total=len(plan["runs"]), groups=Counter(r["group"] for r in plan["runs"])), indent=2))
