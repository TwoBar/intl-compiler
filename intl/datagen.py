"""INTL Training Data Generator (§5.1–5.4).

Generates JSONL training pairs for a given adapter using Claude.
- Category A (~60%): Fresh INTL → target code
- Category B (~30%): PATCH compilation
- Category C (~10%): Error correction

Output: /workspace/data/<adapter>/train.jsonl
        /workspace/data/<adapter>/validation.jsonl

Usage:
    python3 -m intl.datagen --adapter python_fastapi --count 3000
    python3 -m intl.datagen --adapter python_fastapi --count 3000 --val 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator

import anthropic

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

# ── INTL constructs that must be covered (§5.4) ───────────────────────────────
CONSTRUCTS = [
    "QUERY", "PERSIST", "IF/THEN/ELSE", "FAIL", "RETURN",
    "LOOP", "EMIT", "SEQUENCE", "PARALLEL", "TRANSACTION",
    "LOCK", "FALLBACK", "CACHE GET", "CACHE SET",
    "VALIDATE", "PAGINATE", "AGGREGATE", "TRANSFORM",
    "OBSERVABLE", "CONFIDENCE", "TIMEOUT", "PATCH",
]

# Constructs requiring ≥100 pairs (§5.4)
HIGH_FREQ_CONSTRUCTS = {"PARALLEL", "FALLBACK", "TRANSACTION"}

# ── Category weights ──────────────────────────────────────────────────────────
CATEGORY_WEIGHTS = [("A", 0.60), ("B", 0.30), ("C", 0.10)]


def _pick_category() -> str:
    r = random.random()
    cumulative = 0.0
    for cat, weight in CATEGORY_WEIGHTS:
        cumulative += weight
        if r < cumulative:
            return cat
    return "A"


# ── Prompt templates ──────────────────────────────────────────────────────────
def _system_prompt(adapter: str) -> str:
    lang = adapter.replace("_", " ").title()
    return (
        f"You are the INTL compiler for {lang}. "
        f"Your task is to translate INTL (Intent Language) blocks into idiomatic, "
        f"production-ready {lang} code. "
        f"Output ONLY the compiled code — no explanations, no markdown fences. "
        f"Wrap the output in INTL:BEGIN [id=<id>] and INTL:END [id=<id>] comment sentinels."
    )


def _generation_prompt_A(adapter: str, construct: str, idx: int) -> str:
    """Category A: fresh INTL → target code."""
    return f"""Generate ONE complete INTL training pair for the {adapter} adapter.

The INTL block MUST use the '{construct}' construct prominently.
Use a realistic business domain scenario (auth, payments, orders, inventory, users, etc.).
Assign unique ids (m{idx:04d}, f{idx:04d}, etc.).

Respond with a JSON object with exactly these fields:
{{
  "system": "<system prompt for the {adapter} compiler>",
  "prompt": "<complete INTL function/pipeline block>",
  "completion": "<idiomatic {adapter} code output with INTL:BEGIN/END sentinels>"
}}

Requirements for the completion:
- Must include INTL:BEGIN [id=f{idx:04d}] and INTL:END [id=f{idx:04d}] sentinels
- Must be idiomatic, production-ready {adapter} code
- Must implement ALL PRECONDITION guards as explicit checks
- Must implement MUTATES as actual write operations
- Must implement OBSERVABLE as logging decorator/call
- No TODO, FIXME, or placeholder code
"""


def _generation_prompt_B(adapter: str, construct: str, idx: int) -> str:
    """Category B: PATCH compilation."""
    return f"""Generate ONE PATCH training pair for the {adapter} adapter.

The pair should show: (existing compiled code) + (INTL PATCH block) → (minimally modified output).
Use the '{construct}' construct as the patch target.
Use a realistic business scenario.

Respond with a JSON object with exactly these fields:
{{
  "system": "<system prompt for the {adapter} compiler — PATCH mode>",
  "prompt": "<existing compiled code snippet + INTL PATCH block>",
  "completion": "<minimally modified {adapter} code with INTL:BEGIN/END sentinels>"
}}

The prompt field should contain:
1. The existing compiled code (wrapped in INTL:BEGIN/END sentinels)
2. The INTL PATCH block requesting a specific change

The completion should only change what the PATCH block specifies — minimal diff.
"""


def _generation_prompt_C(adapter: str, construct: str, idx: int) -> str:
    """Category C: error correction."""
    return f"""Generate ONE error-correction training pair for the {adapter} adapter.

Show: (INTL block) + (failed/incorrect compiled output) + (error message) → (corrected output).
Use the '{construct}' construct.

Respond with a JSON object with exactly these fields:
{{
  "system": "<system prompt for the {adapter} compiler — error correction mode>",
  "prompt": "<INTL block + failed output + validation error message>",
  "completion": "<corrected, complete {adapter} code with INTL:BEGIN/END sentinels>"
}}

The prompt should clearly show what went wrong. The completion fixes it completely.
"""


# ── Pair generation ───────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a Claude response."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { ... } block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Could not extract balanced JSON from response")


def _generate_pair(
    client: anthropic.Anthropic,
    adapter: str,
    category: str,
    construct: str,
    idx: int,
    retries: int = 3,
) -> dict | None:
    """Ask Claude to generate a single training pair. Returns dict or None on failure."""
    prompt_fn = {"A": _generation_prompt_A, "B": _generation_prompt_B,
                 "C": _generation_prompt_C}[category]
    user_msg = prompt_fn(adapter, construct, idx)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=3000,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            pair = _extract_json(raw)

            # Validate structure
            if not all(k in pair for k in ("system", "prompt", "completion")):
                logger.warning("Missing fields in pair (attempt %d)", attempt + 1)
                continue

            pair["category"] = category
            pair["construct"] = construct
            pair["adapter"] = adapter
            return pair

        except Exception as e:
            logger.warning("Generation attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)

    return None


def _construct_schedule(total: int) -> Iterator[str]:
    """Yield constructs ensuring coverage requirements (§5.4)."""
    # First, ensure high-freq constructs get ≥100 pairs each
    schedule = []
    for c in HIGH_FREQ_CONSTRUCTS:
        schedule.extend([c] * 100)
    # Fill remainder with random constructs
    remaining = max(0, total - len(schedule))
    pool = CONSTRUCTS * ((remaining // len(CONSTRUCTS)) + 2)
    random.shuffle(pool)
    schedule.extend(pool[:remaining])
    random.shuffle(schedule)
    for c in schedule[:total]:
        yield c


# ── Main generation loop ──────────────────────────────────────────────────────
def generate(
    adapter: str,
    count: int = 1000,
    val_count: int = 200,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Generate *count* training pairs + *val_count* validation pairs.

    Returns (train_path, val_path).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    out_dir = output_dir or (DATA_DIR / adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"

    if dry_run:
        logger.info("[DRY RUN] Would generate %d train + %d val pairs for %s",
                    count, val_count, adapter)
        return train_path, val_path

    client = anthropic.Anthropic(api_key=api_key)
    total = count + val_count
    construct_gen = _construct_schedule(total)
    pairs: list[dict] = []

    logger.info("Generating %d pairs for adapter '%s' …", total, adapter)
    for i in range(total):
        construct = next(construct_gen)
        category = _pick_category()
        pair = _generate_pair(client, adapter, category, construct, idx=i)
        if pair is not None:
            pairs.append(pair)
        if (i + 1) % 50 == 0:
            logger.info("  Progress: %d/%d pairs generated (%d failed)",
                        len(pairs), i + 1, (i + 1) - len(pairs))

    # Shuffle and split
    random.shuffle(pairs)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    _write_jsonl(train_path, train_pairs)
    _write_jsonl(val_path, val_pairs)

    logger.info("Wrote %d train pairs → %s", len(train_pairs), train_path)
    logger.info("Wrote %d validation pairs → %s", len(val_pairs), val_path)
    return train_path, val_path


def _write_jsonl(path: Path, pairs: list[dict]):
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ── HuggingFace upload ─────────────────────────────────────────────────────────
def push_to_hub(adapter: str, data_dir: Path | None = None) -> str:
    """Upload generated data to HuggingFace dataset repo. Returns URL."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    hf_user = os.environ.get("HF_USERNAME")
    if not token or not hf_user:
        raise RuntimeError("HF_TOKEN and HF_USERNAME must be set")

    folder = data_dir or (DATA_DIR / adapter)
    repo_id = f"{hf_user}/intl-training-pairs"

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        path_in_repo=adapter,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{adapter}"
    logger.info("Pushed %s data → %s", adapter, url)
    return url


# ── CLI entry point ───────────────────────────────────────────────────────────
def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="INTL Training Data Generator")
    parser.add_argument("--adapter", required=True, help="Adapter name (e.g. python_fastapi)")
    parser.add_argument("--count", type=int, default=1000, help="Number of training pairs")
    parser.add_argument("--val", type=int, default=200, help="Validation pairs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace after generation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path, val_path = generate(
        args.adapter, args.count, args.val, args.output_dir, args.dry_run
    )
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")

    if args.push and not args.dry_run:
        url = push_to_hub(args.adapter, args.output_dir)
        print(f"HF:    {url}")


if __name__ == "__main__":
    _cli()
