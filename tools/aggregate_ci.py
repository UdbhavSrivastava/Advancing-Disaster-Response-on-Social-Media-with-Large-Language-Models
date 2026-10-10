import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["accuracy", "precision", "recall", "f1"]


def metric_value(row, metric):
    for key in (f"test_{metric}", f"eval_{metric}", metric):
        value = row.get(key)
        if value is not None and not pd.isna(value):
            return value
    return np.nan


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) <= 1:
        return 0.0
    return 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))


def load_rows(root):
    rows = []
    for path in Path(root).rglob("metrics.json"):
        with path.open() as f:
            row = json.load(f)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {root}")
    return rows


def summarize(per_run):
    rows = []
    for task, group in per_run.groupby("task"):
        for metric in METRICS:
            vals = group[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci = float(ci95(vals))
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "n": int(len(vals)),
                    "mean": mean,
                    "std": std,
                    "ci95": ci,
                    "ci95_low": mean - ci,
                    "ci95_high": mean + ci,
                    "mean_std": f"{mean:.6f}+/-{std:.6f}",
                    "ci95_interval": f"[{mean - ci:.6f}, {mean + ci:.6f}]",
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate seed metrics and compute 95% CIs.")
    parser.add_argument("--root", required=True, help="Root containing per-run metrics.json files.")
    parser.add_argument("--out_csv", required=True, help="Summary CSV path.")
    parser.add_argument("--per_run_csv", default=None, help="Optional normalized per-run CSV path.")
    args = parser.parse_args()

    raw_rows = load_rows(args.root)
    per_run_rows = []
    for row in raw_rows:
        normalized = {
            "task": row.get("task"),
            "seed": row.get("seed"),
            "output_dir": row.get("output_dir"),
        }
        for metric in METRICS:
            normalized[metric] = metric_value(row, metric)
        per_run_rows.append(normalized)

    per_run = pd.DataFrame(per_run_rows).sort_values(["task", "seed"])
    summary = summarize(per_run)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    if args.per_run_csv:
        per_run_csv = Path(args.per_run_csv)
        per_run_csv.parent.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(per_run_csv, index=False)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
