#!/usr/bin/env python3
"""Benchmark a running vLLM endpoint on the DCOCO paper suite with `vllm bench serve`.

    python scripts/benchmark.py --endpoint http://localhost:8000 --out results/q3-8b

Runs the 10 suites x 4 configs of scripts/suite.py (one `vllm bench serve` per prompt file),
then writes summary.md / summary.json in the paper's format (scripts/summary.py). The default
prompt selection is the paper's (128 rows per suite, seed 1234); --rows all uses every row.
The selected prompts and their indices are written to <out>/prompts/.

Resumable: finished runs are skipped. To split the suite over several servers of the same
model, give each a subset with --datasets and the same --out.

Acceptance length comes from the server's /metrics counters, which are server-wide: nothing
else may send requests to the endpoint while it is benchmarked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from suite import (CONFIGS, DATASETS, ROOT, SPEEDBENCH, SPEEDBENCH_CATEGORIES, jobs_for,
                   write_prompt_files)
from summary import summarize
from vllm_bench import VllmCli, run_job

def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def served_model(endpoint: str) -> str:
    try:
        urllib.request.urlopen(f"{endpoint}/health", timeout=10)  # noqa: S310
        with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as r:  # noqa: S310
            return json.load(r)["data"][0]["id"]
    except OSError as e:
        sys.exit(f"endpoint {endpoint} is not ready: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True, help="e.g. http://localhost:8000")
    p.add_argument("--out", required=True, type=Path, help="results directory")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--configs", nargs="+", default=[c.tag for c in CONFIGS],
                   choices=[c.tag for c in CONFIGS])
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--tokenizer", help="for client-side prompt lengths (default: served model)")
    p.add_argument("--rows", default="128",
                   help="prompts per suite: a number (default 128, the paper's) or 'all'")
    p.add_argument("--seed", type=int, default=1234,
                   help="row-sampling seed (default 1234, the paper's)")
    p.add_argument("--speedbench-dir", type=Path, default=ROOT / "data" / "speedbench_data",
                   help="SPEED-Bench <category>.jsonl files (default: %(default)s)")
    runner = p.add_mutually_exclusive_group()
    runner.add_argument("--image", default="dcoco:vllm-v0.31.0",
                        help="run `vllm bench serve` in this image (default)")
    runner.add_argument("--vllm-bin", help="use this local `vllm` executable instead")
    args = p.parse_args()
    args.endpoint = args.endpoint.rstrip("/").removesuffix("/v1")
    args.out = args.out.resolve()
    args.speedbench_dir = args.speedbench_dir.resolve()
    args.rows = None if args.rows == "all" else int(args.rows)
    return args


def main() -> None:
    args = parse_args()
    model = served_model(args.endpoint)
    missing = [c for c in SPEEDBENCH_CATEGORIES if SPEEDBENCH in args.datasets
               and not (args.speedbench_dir / f"{c}.jsonl").exists()]
    if missing:
        sys.exit(f"SPEED-Bench files missing from {args.speedbench_dir}: {', '.join(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt_dir = write_prompt_files(args.out, args.rows, args.seed)
    cli = VllmCli(None if args.vllm_bin else args.image, args.vllm_bin,
                  mounts=[args.speedbench_dir, args.out])
    bench_kwargs = dict(endpoint=args.endpoint, model=model,
                        tokenizer=args.tokenizer or model, concurrency=args.concurrency)

    rows = "all rows" if args.rows is None else f"{args.rows} rows/suite, seed {args.seed}"
    log(f"benchmarking {model} at {args.endpoint} (c={args.concurrency}, {rows}) -> {args.out}")
    failed = []
    for config in [c for c in CONFIGS if c.tag in args.configs]:
        for dataset in args.datasets:  # config-major: all of T=0 before any T=1 seed
            jobs = jobs_for(dataset, config, args.out, args.concurrency, prompt_dir,
                            args.speedbench_dir)
            bad = [job for job in jobs if not run_job(cli, job, **bench_kwargs)]
            failed += bad
            log(f"{dataset:14s} {config.tag:7s} "
                + ("ok" if not bad else f"FAILED, see {bad[0].result.with_suffix('.log')}"))

    print(summarize(args.out, args.concurrency, title=f"{model} @ {args.endpoint}"))
    if failed:
        sys.exit(f"{len(failed)} runs failed")


if __name__ == "__main__":
    main()
