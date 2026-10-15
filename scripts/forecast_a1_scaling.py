"""Freeze d8/d9/d20 forecasts using only completed d4-d7 pilot losses."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fit(depths, losses):
    c = (np.array(depths, dtype=float) / 8) ** 3
    y = np.array(losses, dtype=float)
    candidates = []
    # At each exponent, solve the linear least-squares problem for a and epsilon.
    for alpha in np.linspace(0.01, 2.0, 12000):
        design = np.column_stack([c ** -alpha, np.ones_like(c)])
        a, epsilon = np.linalg.lstsq(design, y, rcond=None)[0]
        if a >= 0 and 0 <= epsilon <= min(y):
            sse = float(np.square(design @ [a, epsilon] - y).sum())
            candidates.append((sse, alpha, a, epsilon))
    if not candidates:
        raise RuntimeError("No positive power-law fit")
    sse, alpha, a, epsilon = min(candidates)
    return dict(a=float(a), alpha=float(alpha), epsilon=float(epsilon), sse=sse,
        predictions={f"d{d}": float(a * ((d / 8) ** 3) ** -alpha + epsilon) for d in (8, 9, 20)})


def main():
    plan = json.loads((ROOT / "experiments/a1/round2.json").read_text())
    source = ROOT / "reports/a1_campaigns_latest.local.json"
    snapshots = json.loads(source.read_text())
    results = {r["id"]: r for c in snapshots["campaigns"].values() for r in c["runs"]}
    grouped = {}
    for row in plan["runs"]:
        if row["group"] != "p2-scaling-pilot":
            continue
        c = row["config"]
        depth = int(c["model_name"][1:])
        assert depth in range(4, 8)
        result = results[row["id"]]
        assert result["state"] == "complete" and result["final"]["completed_steps"] == 9375
        key = (c["lr_schedule"], c["learning_rate"], c.get("dropout", 0), c["model_seed"])
        grouped.setdefault(key, []).append(dict(depth=depth, loss=result["final"]["val_loss"], id=row["id"]))
    output = []
    for (schedule, lr, dropout, seed), points in grouped.items():
        points.sort(key=lambda p: p["depth"])
        output.append(dict(schedule=schedule, learning_rate=lr, dropout=dropout, seed=seed,
            points=points, fit=fit([p["depth"] for p in points], [p["loss"] for p in points])))
    report = dict(created_at_beijing=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formula="L=a*C^(-alpha)+epsilon; C=(depth/8)^3 at fixed token budget",
        fit_inputs="Only d4-d7. Source losses rounded to 6 decimal places by the training transcript.",
        limitations=["Three-parameter fit to four scales is weakly constrained.",
            "d20 is an illustrative extrapolation, not a calibrated confidence statement.",
            "d8 baseline/high-LR/constant results were previously observed; d9 is genuinely held out.",
            "These are assistant preregistrations, not the learner's own predictions."], fits=output)
    target = ROOT / "reports/a1_scaling_predictions.json"
    if target.exists():
        raise FileExistsError("Do not overwrite preregistered forecasts")
    target.write_text(json.dumps(report, indent=2) + "\n")
    for row in output:
        print(row["schedule"], row["learning_rate"], row["dropout"], row["seed"], row["fit"]["predictions"])


if __name__ == "__main__":
    main()
