#!/usr/bin/env python3
"""Summarize a benchmark directory in the paper's format (Tables 1 and 7).

    python scripts/summary.py <out dir> [--concurrency 1]

tok/s             per request i: n_i / L_i (output tokens / request latency incl. prefill);
                  a run reports the median over its requests (paper App. C)
acceptance length spec_decode_acceptance_length: 1 + accepted/drafts from the server's
                  /metrics counters around the run (absent for a non-speculative server)
T=0 is one deterministic run; T=1 is mean +- std over seeds 42/43/44, each seed's statistic
computed first. SPEED-Bench is the mean of its 11 categories. AVG weights the 10 suites
equally. Averages over missing cells are marked partial.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from suite import CONFIGS, DATASETS, TEMPERATURES, result_files

METRICS = ("tps", "al")


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def request_rate_median(result: dict) -> float:
    """Median over successful requests of output tokens / request latency."""
    rates = [n / latency for n, latency in zip(result["output_lens"], result["latencies"], strict=True)
             if n > 0 and latency > 0]
    return statistics.median(rates)


def run_metrics(files: list[Path]) -> dict[str, float | None] | None:
    """tok/s and acceptance length of one dataset x config (mean over SPEED-Bench categories)."""
    if not files:
        return None
    results = [json.loads(f.read_text()) for f in files]
    return {"tps": statistics.mean(request_rate_median(r) for r in results),
            "al": mean_or_none(r.get("spec_decode_acceptance_length") for r in results)}


def dataset_cells(out: Path, dataset: str, concurrency: int) -> dict[str, tuple]:
    """{"tps T=0": (mean, std), ...}; std is over the T=1 seeds, None for T=0."""
    runs = {c.tag: run_metrics(result_files(out, dataset, c.tag, concurrency)) for c in CONFIGS}
    cells = {}
    for metric in METRICS:
        for temp, tags in TEMPERATURES.items():
            values = [runs[t][metric] for t in tags if runs[t] and runs[t][metric] is not None]
            complete = len(values) == len(tags)
            mean = statistics.mean(values) if values else None
            std = statistics.stdev(values) if complete and len(values) > 1 else None
            cells[f"{metric} {temp}"] = (mean, std, complete)
    return cells


def summarize(out: Path, concurrency: int = 1, title: str = "") -> str:
    rows = {d: dataset_cells(out, d, concurrency) for d in DATASETS}
    columns = list(next(iter(rows.values())))
    average = {}
    for col in columns:
        values = [rows[d][col][0] for d in DATASETS]
        partial = any(v is None or not rows[d][col][2] for d, v in zip(DATASETS, values, strict=True))
        average[col] = (mean_or_none(values), None, not partial)
    rows["AVG (10 suites)"] = average
    done = sum(bool(result_files(out, d, c.tag, concurrency)) for d in DATASETS for c in CONFIGS)

    def fmt(cell, decimals):
        mean, std, complete = cell
        if mean is None:
            return ""
        text = f"{mean:.{decimals}f}" + (f" ±{std:.{decimals}f}" if std is not None else "")
        return text if complete else text + "*"

    header = ["dataset", "tok/s T=0", "tok/s T=1", "accept len T=0", "accept len T=1"]
    lines = [f"# {title or out.name}", "",
             f"concurrency {concurrency}; dataset x config runs done {done}/"
             f"{len(DATASETS) * len(CONFIGS)}; * = incomplete", "",
             "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for name, cells in rows.items():
        lines.append(f"| {name} | " + " | ".join(
            fmt(cells[col], 1 if col.startswith("tps") else 3) for col in columns) + " |")
    table = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(table)
    (out / "summary.json").write_text(json.dumps(
        {"concurrency": concurrency, "runs_done": done,
         "rows": {name: {col: {"mean": c[0], "std": c[1], "complete": c[2]}
                         for col, c in cells.items()} for name, cells in rows.items()}},
        indent=2))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    print(summarize(args.out, args.concurrency))
