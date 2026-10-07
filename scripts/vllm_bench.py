"""Run `vllm bench serve` for one suite job, with a local vllm or inside the DCOCO image."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from suite import MAX_TOKENS, Job


class VllmCli:
    """Invokes the `vllm` CLI, either a given executable or `docker run <image>`."""

    def __init__(self, image: str | None, vllm_bin: str | None, mounts: list[Path]):
        self.image = image
        self.vllm_bin = vllm_bin
        self.mounts = sorted({str(m.resolve()) for m in mounts})

    def command(self, args: list[str]) -> list[str]:
        if self.vllm_bin:
            return [self.vllm_bin, *args]
        hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        docker = ["docker", "run", "--rm", "--network", "host",
                  "--user", f"{os.getuid()}:{os.getgid()}",
                  "-e", "HOME=/tmp", "-e", "HF_TOKEN",
                  "-v", f"{hf_cache}:/tmp/.cache/huggingface"]
        for path in self.mounts:
            docker += ["-v", f"{path}:{path}"]
        return [*docker, "--entrypoint", "vllm", self.image, *args]


def bench_serve_args(job: Job, endpoint: str, model: str, tokenizer: str,
                     concurrency: int) -> list[str]:
    """`vllm bench serve` arguments for one job: every prompt once, in file order."""
    num_prompts = sum(1 for _ in job.prompts.open())
    return [
        "bench", "serve",
        "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
        "--base-url", endpoint, "--model", model, "--tokenizer", tokenizer,
        "--dataset-name", "custom", "--dataset-path", str(job.prompts),
        "--num-prompts", str(num_prompts), "--no-oversample", "--disable-shuffle",
        # untemplated user message; the server applies the chat template
        "--skip-chat-template",
        "--custom-output-len", str(MAX_TOKENS),
        "--temperature", str(job.config.temperature),
        "--extra-body", json.dumps(job.config.extra_body()),
        "--max-concurrency", str(concurrency), "--request-rate", "inf",
        # the paper discards no warmup requests
        "--num-warmups", "0",
        # per-request output lengths, needed for the per-request rate
        "--save-result", "--save-detailed",
        "--result-dir", str(job.result.parent), "--result-filename", job.result.name,
    ]


def run_job(cli: VllmCli, job: Job, **bench_kwargs) -> bool:
    """Run one job unless its result exists; the log goes next to the result."""
    if job.result.exists():
        return True
    job.result.parent.mkdir(parents=True, exist_ok=True)
    cmd = cli.command(bench_serve_args(job, **bench_kwargs))
    with job.result.with_suffix(".log").open("w") as log:
        ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0
    if not ok:
        job.result.unlink(missing_ok=True)
    return ok
