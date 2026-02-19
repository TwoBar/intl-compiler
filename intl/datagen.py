"""INTL Training Data Generator (§5.1–5.4).

Generates JSONL training pairs for a given adapter using Claude.
- Category A (~60%): Fresh INTL → target code
- Category B (~30%): PATCH compilation
- Category C (~10%): Error correction

Output: /workspace/data/<adapter>/train.jsonl
        /workspace/data/<adapter>/validation.jsonl

Usage:
    python3 -m intl.datagen --adapter python_fastapi --count 3000 --val 200
    python3 -m intl.datagen --adapter sql_postgres   --count 1500 --val 200
    python3 -m intl.datagen --adapter python_fastapi --count 20          # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2000
CONCURRENCY = 2        # conservative — haiku has tighter rate limits
MAX_RETRIES = 3
RETRY_DELAY = 8

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

# ── INTL constructs (§5.4) ────────────────────────────────────────────────────
CONSTRUCTS = [
    "QUERY", "PERSIST", "IF/THEN/ELSE", "FAIL", "RETURN",
    "LOOP", "EMIT", "SEQUENCE", "PARALLEL", "TRANSACTION",
    "LOCK", "FALLBACK", "CACHE GET", "CACHE SET",
    "VALIDATE", "PAGINATE", "AGGREGATE", "TRANSFORM",
    "OBSERVABLE", "CONFIDENCE", "TIMEOUT", "PATCH",
]
HIGH_FREQ_CONSTRUCTS = {"PARALLEL", "FALLBACK", "TRANSACTION"}

CATEGORY_WEIGHTS = [("A", 0.60), ("B", 0.30), ("C", 0.10)]

DOMAINS = [
    "authentication", "payments", "orders", "inventory", "users",
    "notifications", "reporting", "search", "products", "subscriptions",
    "shipping", "reviews", "coupons", "analytics", "sessions",
    "audit logs", "file uploads", "webhooks", "api keys", "permissions",
]


# ── INTL block examples for grounding ─────────────────────────────────────────
INTL_EXAMPLE = """\
FUNCTION login [id=f001]
  INTENT       "validate credentials and return session token"
  PRECONDITION email.length > 0
  PRECONDITION password.length >= 8
  POSTCONDITION result.token IS NOT NULL
  READS        users_table
  MUTATES      sessions_table
  OBSERVABLE

  user = QUERY users_table WHERE email == email LIMIT 1
  IF user IS NULL THEN FAIL AuthError("user_not_found")
  IF NOT verify_hash(password, user.password_hash) THEN FAIL AuthError("invalid_password")
  token = generate_token(user.id)
  PERSIST sessions_table (user_id: user.id, token: token, expires_at: now()+24h)
  RETURN SessionToken(token: token)
END FUNCTION login [id=f001]"""

FASTAPI_EXAMPLE = """\
# ═══ INTL:BEGIN [id=f001] login ═══
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

async def login(email: str, password: str) -> SessionToken:
    assert len(email) > 0, "email must not be empty"
    assert len(password) >= 8, "password must be at least 8 chars"
    user = await db.query(users_table).filter(email=email).first()
    if user is None:
        raise AuthError("user_not_found")
    if not verify_hash(password, user.password_hash):
        raise AuthError("invalid_password")
    token = generate_token(user.id)
    await db.insert(sessions_table, user_id=user.id, token=token,
                    expires_at=datetime.utcnow() + timedelta(hours=24))
    logger.info("login: user=%s", user.id)
    result = SessionToken(token=token)
    assert result.token is not None
    return result
# ═══ INTL:END   [id=f001] login ═══"""


def _adapter_display(adapter: str) -> str:
    parts = adapter.split("_")
    lang = parts[0].title()
    framework = " ".join(p.title() for p in parts[1:]) if len(parts) > 1 else ""
    return f"{lang} {framework}".strip() if framework else lang


def _pick_category() -> str:
    r = random.random()
    cumulative = 0.0
    for cat, weight in CATEGORY_WEIGHTS:
        cumulative += weight
        if r < cumulative:
            return cat
    return "A"


def _construct_schedule(n: int) -> list[str]:
    schedule: list[str] = []
    for c in HIGH_FREQ_CONSTRUCTS:
        schedule.extend([c] * min(100, n // len(CONSTRUCTS)))
    for c in CONSTRUCTS:
        if c not in HIGH_FREQ_CONSTRUCTS:
            schedule.extend([c] * 50)
    while len(schedule) < n:
        schedule.append(random.choice(CONSTRUCTS))
    random.shuffle(schedule)
    return schedule[:n]


# ── Delimiter-based prompt format ─────────────────────────────────────────────
DELIM_SYSTEM    = "<<<SYSTEM>>>"
DELIM_PROMPT    = "<<<PROMPT>>>"
DELIM_COMPLETION = "<<<COMPLETION>>>"
DELIM_END       = "<<<END>>>"


def _parse_delimited(text: str) -> dict | None:
    """Parse delimiter-separated response into {system, prompt, completion}."""
    try:
        sys_part  = _extract(text, DELIM_SYSTEM,     DELIM_PROMPT)
        prompt    = _extract(text, DELIM_PROMPT,      DELIM_COMPLETION)
        completion = _extract(text, DELIM_COMPLETION, DELIM_END)
        if not sys_part or not prompt or not completion:
            return None
        return {"system": sys_part, "prompt": prompt, "completion": completion}
    except Exception:
        return None


def _extract(text: str, start_tag: str, end_tag: str) -> str:
    try:
        start = text.index(start_tag) + len(start_tag)
        end   = text.index(end_tag, start)
        return text[start:end].strip()
    except ValueError:
        return ""


# ── Prompt builders ───────────────────────────────────────────────────────────
def _prompt_A(adapter: str, construct: str, idx: int, domain: str) -> str:
    display = _adapter_display(adapter)
    sys_content = (
        f"You are the INTL compiler for {display}. "
        f"Given an INTL block, produce idiomatic, production-ready {display} code. "
        f"Wrap output in INTL:BEGIN / INTL:END sentinels with the correct block ID."
    )
    return f"""You are generating training data for the INTL compiler system.

INTL (Intent Language) is a structured specification language. Here is an example INTL block:

{INTL_EXAMPLE}

And the corresponding {display} compilation:

{FASTAPI_EXAMPLE}

---

YOUR TASK: Generate ONE training pair for the {display} adapter.

Requirements:
- INTL block MUST use the '{construct}' construct prominently
- Domain: {domain}
- Block ID: f{idx:04d}
- No TODO, FIXME, placeholder, or stub code in completion
- All PRECONDITION → explicit guard clause in completion
- All MUTATES → actual write operation
- OBSERVABLE (if present) → logging call
- INTL:BEGIN sentinel format: # ═══ INTL:BEGIN [id=f{idx:04d}] <name> ═══
- INTL:END sentinel format:   # ═══ INTL:END   [id=f{idx:04d}] <name> ═══

Respond using EXACTLY this format (no other text):

{DELIM_SYSTEM}
{sys_content}
{DELIM_PROMPT}
<write a complete INTL FUNCTION block here using the {construct} construct>
{DELIM_COMPLETION}
<write idiomatic {display} code here with proper INTL:BEGIN/END sentinels>
{DELIM_END}"""


def _prompt_B(adapter: str, construct: str, idx: int, domain: str) -> str:
    display = _adapter_display(adapter)
    sys_content = (
        f"You are the INTL compiler for {display} in PATCH mode. "
        f"Apply the PATCH block to the existing code with minimal diff. "
        f"Wrap output in INTL:BEGIN / INTL:END sentinels."
    )
    return f"""You are generating PATCH training data for the INTL compiler.

A PATCH pair shows: existing compiled code + INTL PATCH block → minimally modified output.

INTL PATCH syntax example:
PATCH f{idx:04d} [target=f{idx:04d}]
  INTENT "add rate limiting to the function"
  ADD BEFORE: rate_limit_check(user_id)
END PATCH f{idx:04d}

Domain: {domain}
Construct to patch: {construct}
Block ID: f{idx:04d}
Target language: {display}

Respond using EXACTLY this format (no other text):

{DELIM_SYSTEM}
{sys_content}
{DELIM_PROMPT}
<existing {display} code in INTL:BEGIN/END sentinels>

<INTL PATCH block requesting a {construct}-related change>
{DELIM_COMPLETION}
<minimally modified {display} code — only the PATCH change applied — in INTL:BEGIN/END sentinels>
{DELIM_END}"""


def _prompt_C(adapter: str, construct: str, idx: int, domain: str) -> str:
    display = _adapter_display(adapter)
    sys_content = (
        f"You are the INTL escalation compiler for {display}. "
        f"Fix the failed output. Return ONLY code wrapped in INTL sentinels."
    )
    return f"""You are generating error-correction training data for the INTL compiler.

A Category C pair shows: INTL block + failed output + error → corrected output.

Domain: {domain}
Construct: {construct}
Block ID: f{idx:04d}
Target language: {display}

The failed code must have 1-2 realistic errors (missing guard, wrong type, TODO stub, or missing sentinel).
The correction must fix ALL errors completely.

Respond using EXACTLY this format (no other text):

{DELIM_SYSTEM}
{sys_content}
{DELIM_PROMPT}
<INTL block with {construct}>

# PREVIOUS ATTEMPT (failed):
<broken {display} code with 1-2 realistic errors>

# FAILING CHECKS: <check name(s) and brief error description>
{DELIM_COMPLETION}
<corrected {display} code with all issues fixed, in proper INTL:BEGIN/END sentinels>
{DELIM_END}"""


def _build_prompt(adapter: str, category: str, construct: str, idx: int) -> str:
    domain = random.choice(DOMAINS)
    if category == "A":
        return _prompt_A(adapter, construct, idx, domain)
    if category == "B":
        return _prompt_B(adapter, construct, idx, domain)
    return _prompt_C(adapter, construct, idx, domain)


# ── Core async generator ──────────────────────────────────────────────────────
async def _generate_one(
    client: anthropic.AsyncAnthropic,
    adapter: str,
    category: str,
    construct: str,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    prompt = _build_prompt(adapter, category, construct, idx)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = resp.content[0].text
            pair = _parse_delimited(raw)

            if pair is None:
                logger.warning("[%d/%d] idx=%d delimiter parse failed — raw[:200]: %s",
                               attempt, MAX_RETRIES, idx, raw[:200])
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None

            # Basic quality gate: prompt must look like INTL, completion must have sentinel
            if "FUNCTION" not in pair["prompt"] and "PIPELINE" not in pair["prompt"] and "PATCH" not in pair["prompt"]:
                logger.warning("[%d/%d] idx=%d prompt doesn't contain INTL construct", attempt, MAX_RETRIES, idx)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None

            if "INTL:BEGIN" not in pair["completion"] or "INTL:END" not in pair["completion"]:
                logger.warning("[%d/%d] idx=%d completion missing sentinels", attempt, MAX_RETRIES, idx)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None

            pair["metadata"] = {
                "category": category,
                "adapter": adapter,
                "construct": construct,
                "idx": idx,
            }
            return pair

        except anthropic.RateLimitError:
            wait = RETRY_DELAY * attempt * 2
            logger.warning("[%d/%d] idx=%d rate limited — sleeping %ds", attempt, MAX_RETRIES, idx, wait)
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            logger.warning("[%d/%d] idx=%d API error: %s", attempt, MAX_RETRIES, idx, e)
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning("[%d/%d] idx=%d unexpected: %s", attempt, MAX_RETRIES, idx, e)
            await asyncio.sleep(RETRY_DELAY)

    logger.error("idx=%d failed after %d attempts", idx, MAX_RETRIES)
    return None


async def _generate_all(
    adapter: str,
    total: int,
    offset: int,
    checkpoint_path: Path,
) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client  = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    constructs = _construct_schedule(total)
    categories = [_pick_category() for _ in range(total)]

    tasks = [
        _generate_one(client, adapter, categories[i], constructs[i], offset + i, semaphore)
        for i in range(total)
    ]

    pairs: list[dict] = []
    failed = 0
    done   = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        done  += 1
        if result is not None:
            pairs.append(result)
            if len(pairs) % 10 == 0:
                _write_jsonl(checkpoint_path, pairs)
                logger.info("  checkpoint: %d/%d — %d good, %d failed",
                            done, total, len(pairs), failed)
        else:
            failed += 1

        if done % 10 == 0 or done == total:
            logger.info("  progress: %d/%d — %d good, %d failed", done, total, len(pairs), failed)

    logger.info("Done: %d good, %d failed / %d total", len(pairs), failed, total)
    return pairs


# ── Public API ────────────────────────────────────────────────────────────────
def generate(
    adapter: str,
    count: int = 1000,
    val_count: int = 200,
    output_dir: Path | None = None,
    dry_run: bool = False,
    append: bool = False,
) -> tuple[Path, Path]:
    out_dir = output_dir or (DATA_DIR / adapter)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "validation.jsonl"
    checkpoint = out_dir / "train.jsonl.tmp"

    existing_train = _read_jsonl(train_path) if (append and train_path.exists()) else []
    existing_val   = _read_jsonl(val_path)   if (append and val_path.exists()) else []

    need_train = max(0, count     - len(existing_train))
    need_val   = max(0, val_count - len(existing_val))
    total_needed = need_train + need_val

    if dry_run:
        est = (total_needed / CONCURRENCY) * 10 / 60
        logger.info(
            "[DRY RUN] Would generate %d train + %d val pairs for %s "
            "(model=%s, workers=%d, est=%.0f min)",
            need_train, need_val, adapter, MODEL, CONCURRENCY, est,
        )
        return train_path, val_path

    if total_needed == 0:
        logger.info("Already have %d train + %d val — nothing to do.",
                    len(existing_train), len(existing_val))
        return train_path, val_path

    est = (total_needed / CONCURRENCY) * 10 / 60
    logger.info(
        "Generating %d pairs for %s — model=%s, workers=%d, est=%.0f min",
        total_needed, adapter, MODEL, CONCURRENCY, est,
    )

    offset = len(existing_train) + len(existing_val)
    new_pairs = asyncio.run(_generate_all(adapter, total_needed, offset, checkpoint))

    random.shuffle(new_pairs)
    new_val   = new_pairs[:need_val]
    new_train = new_pairs[need_val:]

    all_train = existing_train + new_train
    all_val   = existing_val + new_val

    _write_jsonl(train_path, all_train)
    _write_jsonl(val_path,   all_val)

    if checkpoint.exists():
        checkpoint.unlink()

    logger.info("Wrote %d train → %s", len(all_train), train_path)
    logger.info("Wrote %d val   → %s", len(all_val),   val_path)
    return train_path, val_path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines


def _write_jsonl(path: Path, pairs: list[dict]) -> None:
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ── HuggingFace upload ────────────────────────────────────────────────────────
def push_to_hub(adapter: str, data_dir: Path | None = None) -> str:
    from huggingface_hub import HfApi

    token   = os.environ.get("HF_TOKEN", "")
    hf_user = os.environ.get("HF_USERNAME", "")
    if not token or not hf_user:
        raise RuntimeError("HF_TOKEN and HF_USERNAME must be set")

    folder  = data_dir or (DATA_DIR / adapter)
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
    logger.info("Pushed %s → %s", adapter, url)
    return url


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="INTL Training Data Generator")
    p.add_argument("--adapter",    required=True)
    p.add_argument("--count",      type=int,  default=1000, help="Target training pairs")
    p.add_argument("--val",        type=int,  default=200,  help="Target validation pairs")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--push",       action="store_true")
    p.add_argument("--append",     action="store_true")
    p.add_argument("--dry-run",    action="store_true")
    args = p.parse_args()

    train_path, val_path = generate(
        args.adapter, args.count, args.val, args.output_dir, args.dry_run, args.append,
    )
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")

    if args.push and not args.dry_run:
        url = push_to_hub(args.adapter, args.output_dir)
        print(f"HF:    {url}")


if __name__ == "__main__":
    _cli()
