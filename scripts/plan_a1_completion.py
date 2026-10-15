"""Explicit remaining A1 comparisons; deduplicate against completed initial/R2 work."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from plan_a1 import make_plan
from plan_a1_round2 import make_round2

ROOT = Path(__file__).resolve().parents[1]


def make_completion():
    initial, r2 = make_plan(), make_round2()
    default = next(r["config"] for r in initial["runs"] if r["id"] == "a1-lr-0.003-s42")
    defaults = {**default, "dropout": 0.0, "deterministic": False, "perturb_one_token": False,
                "qk_norm": True, "tie_word_embeddings": False}
    def key(config, interventions=None):
        return json.dumps(dict(config={**defaults, **config}, interventions=interventions or {}), sort_keys=True)
    known = {key(r["config"]): r["id"] for r in initial["runs"] if r["group"] != "smoke"}
    for row in r2["runs"]:
        if "replica" not in row:
            known[key(row["config"])] = row["id"] + ("-fix1" if row["config"]["beta1"] == 0 else "")
    rows, reused, requested, seen = [], {}, [], {}

    def add(group, purpose, label, changes, *, interventions=None, host=None, gpu=None, repeat=None):
        config = {**default, **changes}
        sig = key(config, interventions)
        request = dict(group=group, label=label, config=config, interventions=interventions or {})
        if repeat is None and host is None and sig in known:
            reused.setdefault(known[sig], []).append(group)
            request["run_id"] = known[sig]
        elif repeat is None and host is None and sig in seen:
            row = seen[sig]
            if group not in row["comparisons"]:
                row["comparisons"].append(group)
            request["run_id"] = row["id"]
        else:
            row = dict(id=f"a1-c-{len(rows)+1:03d}", group=group, purpose=purpose,
                display_name=label, config=config, comparisons=[group])
            if interventions:
                row["interventions"] = interventions
            if host is not None:
                row["required_host_index"] = host
            if gpu is not None:
                row["required_gpu"] = gpu
            if repeat is not None:
                row["replica"] = repeat
            rows.append(row)
            if host is None and repeat is None:
                seen[sig] = row
            request["run_id"] = row["id"]
        requested.append(request)

    # Literal P1 quiz grids; 0.009 is deliberately not rounded to 0.01.
    for seed in (42, 43):
        for batch in (64, 128):
            for lr in (.001, .003, .009):
                add("p1-example-batch", "PDF p6: exact batch64/128 × LR0.001/0.003/0.009",
                    f"P1 example · b={batch} lr={lr:g} · s={seed}",
                    dict(batch_size=batch, learning_rate=lr, model_seed=seed, data_seed=seed))
        for schedule, warmup in (("wsd0.2", 0.), ("wsd0.2", .2), ("linear", .1)):
            add("p1-example-schedule", "PDF p7: LR0.009 WD1 schedule/warmup comparison",
                f"P1 example · {schedule} w={warmup:g} · s={seed}",
                dict(learning_rate=.009, weight_decay=1.0, lr_schedule=schedule,
                     warmup_percent=warmup, model_seed=seed, data_seed=seed))

    for depth in (8, 9):
        for schedule, lr, dropout in (("linear", .003, 0.), ("constant", .003, 0.),
                                     ("linear", .003, .2), ("linear", .03, 0.)):
            for seed in (42, 43):
                add("p2-heldout", "P2a: release after d4-d7 prediction registration",
                    f"P2 verify · d{depth} {schedule} lr={lr:g} drop={dropout:g} · s={seed}",
                    dict(model_name=f"d{depth}", lr_schedule=schedule, learning_rate=lr,
                         dropout=dropout, model_seed=seed, data_seed=seed))

    # Open P2b/c questions become these finite, declared comparisons.
    for depth in range(4, 10):
        for seed in (42, 43):
            for label, change in (("slow-lr", {"learning_rate": .0001}),
                                  ("drop0.5", {"dropout": .5}),
                                  ("aggressive", {"learning_rate": .1}),
                                  ("reuse40k", {"num_train_sequences": 40000, "num_epochs": 15.0})):
                add("p2-slope-break", "P2b/c: controlled model-scaling interventions, fixed 614.4M tokens",
                    f"P2 slope · d{depth} {label} · s={seed}",
                    dict(model_name=f"d{depth}", model_seed=seed, data_seed=seed, **change))
    for depth in (4, 8):
        for sequences in (37504, 75008, 150016, 300032, 600000, 1200000):
            for seed in (42, 43):
                for lr in (.003, .03):
                    add("p2-data-scaling", "P2b/c: input-data horizon scaling; LR is explicitly controlled",
                        f"P2 data · d{depth} n={sequences} lr={lr:g} · s={seed}",
                        dict(model_name=f"d{depth}", num_train_sequences=sequences,
                             model_seed=seed, data_seed=seed, learning_rate=lr))

    # Fill the parallel 10-seed grids, and separately obey the example's SAME GPU
    # requirement with all thirty exact runs on host0/GPU0 in one serial lane.
    for seed in range(4, 14):
        for label, change in (("baseline", {}), ("batch16", {"batch_size": 16}),
                              ("lr0.009", {"learning_rate": .009})):
            cfg = dict(model_seed=seed, data_seed=seed, **change)
            add("p3-variance-ten", "P3c: ten paired seeds per recipe; H20 fleet comparison",
                f"P3 10seed · {label} · s={seed}", cfg)
            add("p3-example-same-gpu", "PDF p14: all thirty runs on the identical physical GPU",
                f"P3 exact GPU · {label} · s={seed}", cfg, host=0, gpu=0,
                repeat=f"same-gpu-{label}-{seed}")
    for depth, sequences in ((9, 600000), (8, 1200000)):
        for seed in range(4, 14):
            add("p3-scale-variance", "P3c: variability at larger model or longer horizon",
                f"P3 scale · d{depth} n={sequences} · s={seed}",
                dict(model_name=f"d{depth}", num_train_sequences=sequences, model_seed=seed, data_seed=seed))

    # A fixed same-GPU repeat establishes the deterministic noise floor.
    for replica in (1, 2):
        add("p3-determinism-repeat", "PDF p13: identical deterministic recipe, same physical GPU",
            f"P3 deterministic · repeat={replica}", dict(deterministic=True), host=1, gpu=0,
            repeat=f"determinism-{replica}")

    # Later rows are first consumed at the specified optimizer step. Edit after
    # global shuffle, before batching; this leaves all earlier examples unchanged.
    # Each seed's control and interventions use the same host/GPU.
    for seed in (42, 43):
        for step in (0, 1000, 5000, 8000):
            for count in (1, 4, 16):
                add("p4-time-magnitude", "P4b: exact number of token edits at a controlled first-consumption step",
                    f"P4 time · t={step} tokens={count} · s={seed}",
                    dict(deterministic=True, model_seed=seed, data_seed=seed),
                    interventions={"token_edit": dict(after_steps=step, count=count, position=100)},
                    host=seed-40, gpu=0)
        add("p4-time-control", "P4b: same-GPU unperturbed control", f"P4 time · control · s={seed}",
            dict(deterministic=True, model_seed=seed, data_seed=seed), host=seed-40, gpu=0,
            repeat=f"time-control-{seed}")

    for seed in (42, 43):
        for label, change in (("baseline", {}), ("batch256", {"batch_size": 256}),
                              ("lr0.0003", {"learning_rate": .0003}),
                              ("lr0.009", {"learning_rate": .009})):
            cfg = dict(model_seed=seed, data_seed=seed, **change)
            add("p5-example", "PDF p16: four literal curve-diagnosis candidates",
                f"P5 curve · {label} · s={seed}", cfg)
            add("p6-fixed-val-rms", "PDF p18: post-block RMS at layers1/3/6 on fixed eight validation sequences",
                f"P6 fixed val · {label} · s={seed}", cfg,
                interventions={"fixed_val_rms": True})

    # Mix topics across launch slots. GPU-reserved rows stay ordered within lane.
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    return dict(version=1, assignment="A1", batch="completion", max_parallel=160,
        upstream_commit=initial["upstream_commit"], context_length=1024,
        runs=ordered, requested_comparisons=requested, reused_runs=reused,
        predictions="reports/a1_scaling_predictions.json",
        exclusions=["PDF p10 explicitly says not to run the changing-context illustrative example.",
                    "d20 is a prediction request; no d20 training is required.",
                    "H100/A100 hardware comparison unavailable in the H20 allocation.",
                    "P7 is an optional learner-designed question, with no fixed recipe to exhaust."])


if __name__ == "__main__":
    plan = make_completion()
    (ROOT / "experiments/a1/completion.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"new_runs": len(plan["runs"]), "groups": Counter(r["group"] for r in plan["runs"]),
                      "reused_runs": len(plan["reused_runs"])}, indent=2))
