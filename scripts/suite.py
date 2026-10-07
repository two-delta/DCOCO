"""The DCOCO paper's benchmark suite (App. C): 10 suites x 4 request configs.

Prompts come from DeepSeek's DeepSpec eval files, vendored unmodified in data/ (the
paper's prompt pool). Each suite is subsampled exactly as td-speculators'
`prepare_evalsets.py --source deepspec` did for the paper: first user turn, stripped, empty
rows dropped, then `sorted(random.Random(seed).sample(range(n), rows))` when the suite has more
than `rows` prompts. The paper uses rows=128, seed=1234 (aime25 30 and mt-bench 80 stay whole).
SPEED-Bench qualitative (11 categories x 80) is a separate, locally built prompt set.

Every request: max_tokens 2048, thinking off, sent as an untemplated user message to
/v1/chat/completions (the server applies the chat template).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MAX_TOKENS = 2048
EVALSET_NAMES = ["math500", "aime25", "gsm8k", "humaneval", "mbpp", "livecodebench",
                 "mt-bench", "arena-hard-v2", "alpaca"]
SPEEDBENCH = "speedbench"  # one run per category; the suite score is the category mean
SPEEDBENCH_CATEGORIES = ["coding", "humanities", "math", "multilingual", "qa", "rag",
                         "reasoning", "roleplay", "stem", "summarization", "writing"]
DATASETS = EVALSET_NAMES + [SPEEDBENCH]


@dataclass(frozen=True)
class Config:
    """One request setting: T=0 is a single deterministic run, T=1 is run with 3 seeds."""
    tag: str
    temperature: float
    seed: int | None = None

    def extra_body(self) -> dict:
        body: dict = {"chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            body["seed"] = self.seed
        return body


CONFIGS = [Config("t0", 0), Config("t1_s42", 1, 42), Config("t1_s43", 1, 43), Config("t1_s44", 1, 44)]
TEMPERATURES = {"T=0": ["t0"], "T=1": ["t1_s42", "t1_s43", "t1_s44"]}


@dataclass(frozen=True)
class Job:
    """One `vllm bench serve` run over one prompt file, saved to one result file."""
    dataset: str
    config: Config
    prompts: Path
    result: Path


def deepspec_prompts(suite: str) -> list[str]:
    """All prompts of a DeepSpec suite: first user turn of each row, stripped, empties dropped.

    Rows are split on "\\n", never str.splitlines(): arena-hard-v2 contains U+2028/U+2029
    inside JSON strings, which splitlines() would tear apart.
    """
    text = (DATA / f"{suite}.jsonl").read_text()
    prompts = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        turns = json.loads(line)["turns"]
        prompt = (turns[0] if isinstance(turns, list) else turns).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def select_rows(num_prompts: int, rows: int | None, seed: int) -> list[int]:
    """Indices to benchmark: every row, or a sorted seeded sample of `rows` of them."""
    if rows is None or num_prompts <= rows:
        return list(range(num_prompts))
    return sorted(random.Random(seed).sample(range(num_prompts), rows))


def write_prompt_files(out: Path, rows: int | None, seed: int) -> Path:
    """Write the selected prompts of every evalset to <out>/prompts/, with a manifest.

    Written once: a resumed run, or a second server splitting the suite into the same <out>,
    reuses the files after checking they describe the same selection and are unchanged.
    """
    prompt_dir = out / "prompts"
    manifest_path = prompt_dir / "manifest.json"
    selection = {"source": "data", "rows": rows, "seed": seed}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["selection"] != selection:
            raise SystemExit(f"{out} holds results for {manifest['selection']}, not {selection}")
        for suite, entry in manifest["suites"].items():
            digest = hashlib.sha256((prompt_dir / f"{suite}.jsonl").read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise SystemExit(f"{prompt_dir / suite}.jsonl changed since it was written")
        return prompt_dir
    prompt_dir.mkdir(parents=True, exist_ok=True)
    suites = {}
    for suite in EVALSET_NAMES:
        prompts = deepspec_prompts(suite)
        indices = select_rows(len(prompts), rows, seed)
        path = prompt_dir / f"{suite}.jsonl"
        path.write_text("".join(json.dumps({"prompt": prompts[i]}) + "\n" for i in indices))
        suites[suite] = {"available": len(prompts), "selected": len(indices),
                         "indices": indices,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # manifest last: its presence means every prompt file is complete
    manifest_path.write_text(json.dumps({"selection": selection, "suites": suites}, indent=1))
    return prompt_dir


def result_dir(out: Path, dataset: str, concurrency: int) -> Path:
    return out / dataset / f"c{concurrency}"


def jobs_for(dataset: str, config: Config, out: Path, concurrency: int,
             prompt_dir: Path, speedbench_dir: Path) -> list[Job]:
    """Evalsets are one run each; SPEED-Bench is one run per category."""
    base = result_dir(out, dataset, concurrency)
    if dataset == SPEEDBENCH:
        return [Job(dataset, config, speedbench_dir / f"{category}.jsonl",
                    base / config.tag / f"{category}.json")
                for category in SPEEDBENCH_CATEGORIES]
    return [Job(dataset, config, prompt_dir / f"{dataset}.jsonl", base / f"{config.tag}.json")]


def result_files(out: Path, dataset: str, tag: str, concurrency: int) -> list[Path]:
    """Result files of one dataset x config; empty unless all of its runs finished."""
    base = result_dir(out, dataset, concurrency)
    if dataset == SPEEDBENCH:
        files = sorted((base / tag).glob("*.json"))
        return files if len(files) == len(SPEEDBENCH_CATEGORIES) else []
    file = base / f"{tag}.json"
    return [file] if file.exists() else []
