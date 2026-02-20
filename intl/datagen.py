"""INTL Training Data Generator — Two-Step Pipeline (§5.1–5.4).

Two-step generation per pair:
  Step 1: Frontier model writes syntactically valid INTL (spec in system prompt)
  Step 2: Frontier model compiles INTL → production target-language code

Categories (§5.3 split):
  A (60%): Fresh FUNCTION/PIPELINE → compiled target code       [2 API calls]
  B (30%): Base INTL + PATCH block → patched compiled code      [4 API calls]
  C (10%): INTL + broken compile + failing checks → correction  [3 API calls]

Training pair format (§5.1):
  {"system": "<adapter compiler sys>",
   "prompt": "<INTL source block>",
   "completion": "<compiled code with INTL:BEGIN/END sentinels>"}

Usage:
    # Smoke test (fast)
    python3 -m intl.datagen --adapter python_fastapi --count 20 --val 5

    # Full run — always in background
    nohup python3 -m intl.datagen --adapter python_fastapi --count 3000 --val 200 \\
      > /workspace/data/python_fastapi/datagen.log 2>&1 &

    # Resume interrupted run
    python3 -m intl.datagen --adapter python_fastapi --count 3000 --val 200 --append
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
MODEL        = "claude-sonnet-4-6"
MAX_TOKENS   = 4096   # compiled functions can be long; 2048 was truncating END sentinel
CONCURRENCY  = 1      # 1 pair at a time — datagen shares API key with the Telegram bot
MAX_RETRIES  = 5
RETRY_DELAY  = 20     # base delay — doubles on rate limit
PAIR_DELAY   = 3.0    # seconds between completed pairs
BATCH_SIZE   = 3      # INTL functions written per API call (Cat A only)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
SPEC_PATH = Path(__file__).parent.parent / "docs" / "INTL_Specification.md"

# ── INTL constructs ───────────────────────────────────────────────────────────
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

# ── Compact INTL syntax reference (inlined to avoid file dependency) ───────────
INTL_SYNTAX_REF = """\
INTL (Intent Language) Syntax Reference

BLOCK TYPES:
  FUNCTION <name> [id=<id>]
    INTENT "<description>"
    [PRECONDITION <expr>]
    [POSTCONDITION <expr>]
    [READS <table>]
    [MUTATES <table>]
    [REQUIRES <module_id>]
    [OBSERVABLE]
    [TIMEOUT <ms>]
    <body constructs>
  END FUNCTION <name> [id=<id>]

  PIPELINE <name> [id=<id>]
    INTENT "<description>"
    STEP <name>: <construct>
    [TIMEOUT <ms>]
  END PIPELINE <name> [id=<id>]

  PATCH <id> [target=<id>]
    INTENT "<description>"
    ADD BEFORE: <construct>
    ADD AFTER: <construct>
    REPLACE: <old> WITH: <new>
  END PATCH <id>

BODY CONSTRUCTS:
  QUERY <table> WHERE <cond> [LIMIT n] [ORDER BY <field>]
  PERSIST <table> (<field>: <val>, ...)
  IF <cond> THEN <stmt> [ELSE <stmt>]
  FAIL <ErrorType>("<msg>")
  RETURN <expr>
  LOOP <var> IN <collection> DO ... END LOOP
  EMIT <EventType>(<field>: <val>, ...)
  SEQUENCE [<stmt>, ...]
  PARALLEL [<stmt>, ...]
  TRANSACTION DO ... END TRANSACTION
  LOCK <resource> DO ... END LOCK
  FALLBACK TRY <stmt> CATCH <stmt>
  CACHE GET <key> INTO <var>
  CACHE SET <key> = <val> [TTL <seconds>]
  VALIDATE <expr> OR FAIL <ErrorType>("<msg>")
  PAGINATE <query> PAGE <n> SIZE <m>
  AGGREGATE <collection> BY <field> [SELECT <exprs>]
  TRANSFORM <collection> USING <func>
  CONFIDENCE <expr> THRESHOLD <n>

SENTINEL FORMAT (required in compilation output):
  # ═══ INTL:BEGIN [id=<id>] <name> ═══
  <compiled code>
  # ═══ INTL:END   [id=<id>] <name> ═══

RULES:
  - Every block MUST have INTENT
  - Block IDs are globally unique (e.g., f0001, p0001)
  - PRECONDITION → explicit guard clause in compiled code
  - POSTCONDITION → assertion in compiled code
  - MUTATES → actual write/insert/update operation
  - OBSERVABLE → logging call at entry and key decision points
  - No TODO, FIXME, stub, or placeholder in compiled output
"""


def _load_spec() -> str:
    """Load INTL spec from file if available, else use compact inline reference."""
    if SPEC_PATH.exists():
        text = SPEC_PATH.read_text(encoding="utf-8")
        # Use first 6000 chars to keep system prompt manageable
        if len(text) > 6000:
            text = text[:6000] + "\n\n[spec truncated — key syntax above covers required constructs]"
        return text
    return INTL_SYNTAX_REF


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


# ── System prompts ────────────────────────────────────────────────────────────

def _sys_intl_writer(spec: str) -> str:
    return (
        "You are an INTL (Intent Language) architect. "
        "Your job is to write syntactically valid, realistic INTL blocks. "
        "Follow the specification precisely. "
        "Every block must have INTENT. Use correct END markers. "
        "Write complete, production-realistic logic — not toy examples.\n\n"
        + spec
    )


def _sys_compiler(adapter: str) -> str:
    display = _adapter_display(adapter)
    return (
        f"You are the INTL compiler for {display}. "
        f"Convert INTL blocks to idiomatic, production-ready {display} code. "
        "Rules:\n"
        "- Wrap ALL output in INTL:BEGIN / INTL:END sentinels matching the block ID\n"
        "- PRECONDITION → explicit guard/assertion\n"
        "- POSTCONDITION → assertion before return\n"
        "- MUTATES → actual write operation (insert/update/delete)\n"
        "- OBSERVABLE → structured logging at entry and decision points\n"
        "- No TODO, FIXME, stub, placeholder, or pass statements\n"
        "- Return ONLY the compiled code with sentinels — no explanation"
    )


def _sys_patch_compiler(adapter: str) -> str:
    display = _adapter_display(adapter)
    return (
        f"You are the INTL compiler for {display} in PATCH mode. "
        "Given existing compiled code and an INTL PATCH block, apply the patch with minimal diff. "
        "Rules:\n"
        "- Preserve existing code structure — only apply the described change\n"
        "- Wrap output in INTL:BEGIN / INTL:END sentinels\n"
        "- No TODO, stub, or placeholder\n"
        "- Return ONLY the patched code — no explanation"
    )


def _sys_escalation(adapter: str) -> str:
    display = _adapter_display(adapter)
    return (
        f"You are the INTL escalation compiler for {display}. "
        "Fix the failed compilation output. "
        "Rules:\n"
        "- Fix ALL listed failing checks completely\n"
        "- Wrap output in INTL:BEGIN / INTL:END sentinels\n"
        "- No TODO, stub, or placeholder\n"
        "- Return ONLY corrected code — no explanation"
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _extract_intl_block(text: str) -> str:
    """Strip markdown fences and leading prose, returning just the INTL block."""
    # Remove opening/closing code fences (```intl, ```INTL, ```, etc.)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    text = text.strip()
    # Find the first INTL keyword and return from there
    for keyword in ("FUNCTION ", "PIPELINE ", "PATCH ", "MODULE "):
        idx = text.find(keyword)
        if idx != -1:
            return text[idx:].strip()
    return text


def _validate_intl(text: str) -> tuple[bool, str]:
    """Check that text looks like a valid INTL block."""
    text = text.strip()
    has_block = any(
        kw in text for kw in ("FUNCTION ", "PIPELINE ", "PATCH ", "MODULE ")
    )
    if not has_block:
        return False, "missing FUNCTION/PIPELINE/PATCH block header"
    if "INTENT" not in text:
        return False, "missing INTENT field"
    if not any(f"END {kw}" in text for kw in ("FUNCTION", "PIPELINE", "PATCH", "MODULE")):
        return False, "missing END marker"
    return True, "ok"


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from compiled output."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def _validate_compiled(text: str) -> tuple[bool, str]:
    """Check that compiled output has sentinels and no stubs."""
    text = text.strip()
    if "INTL:BEGIN" not in text:
        return False, "missing INTL:BEGIN sentinel"
    if "INTL:END" not in text:
        return False, "missing INTL:END sentinel"
    stub_patterns = ["TODO", "FIXME", "pass  #", "raise NotImplementedError", "stub"]
    for pat in stub_patterns:
        if pat in text:
            return False, f"contains stub/placeholder: {pat!r}"
    return True, "ok"


# ── API call helper ───────────────────────────────────────────────────────────

async def _call(
    client: anthropic.AsyncAnthropic,
    system: str,
    user: str,
    semaphore: asyncio.Semaphore,
    context: str = "",
) -> str | None:
    """Single API call with retry on rate limit. Returns text or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            return resp.content[0].text.strip()
        except anthropic.RateLimitError:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning("%s rate limit — sleeping %ds (attempt %d/%d)", context, delay, attempt, MAX_RETRIES)
            await asyncio.sleep(delay)
        except anthropic.APIError as e:
            logger.warning("%s API error: %s (attempt %d/%d)", context, e, attempt, MAX_RETRIES)
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning("%s unexpected error: %s (attempt %d/%d)", context, e, attempt, MAX_RETRIES)
            await asyncio.sleep(RETRY_DELAY)
    logger.error("%s failed after %d attempts", context, MAX_RETRIES)
    return None


# ── Category generators ───────────────────────────────────────────────────────

async def _generate_pair_A(
    client: anthropic.AsyncAnthropic,
    adapter: str,
    construct: str,
    idx: int,
    domain: str,
    semaphore: asyncio.Semaphore,
    spec: str,
    prefetched_intl: str | None = None,
) -> dict | None:
    """Category A: Write INTL → compile to target code.
    If prefetched_intl is provided (from batch), skip Step 1. (1–2 API calls)"""
    ctx = f"[A/idx={idx}]"
    display = _adapter_display(adapter)
    block_id = f"f{idx:04d}"

    if prefetched_intl is not None:
        intl_text = prefetched_intl
        logger.debug("%s using pre-fetched INTL block", ctx)
    else:
        # Step 1: Generate valid INTL block (fallback if batch missed this index)
        writer_prompt = (
            f"Write a realistic INTL FUNCTION block for the '{domain}' domain.\n"
            f"Block ID: {block_id}\n"
            f"Must use the '{construct}' construct. Include INTENT, PRECONDITION, READS or MUTATES.\n"
            f"Return ONLY the INTL block, nothing else."
        )
        intl_text = await _call(client, _sys_intl_writer(spec), writer_prompt, semaphore, ctx + " step1")
        if intl_text is None:
            return None
        intl_text = _extract_intl_block(intl_text)

        ok, reason = _validate_intl(intl_text)
        if not ok:
            logger.warning("%s INTL validation failed: %s", ctx, reason)
            return None

    # Step 2: Compile INTL → target code
    compile_prompt = (
        f"Compile this INTL block to {display} code:\n\n{intl_text}"
    )
    compiled_text = await _call(client, _sys_compiler(adapter), compile_prompt, semaphore, ctx + " step2")
    if compiled_text is None:
        return None
    compiled_text = _strip_fences(compiled_text)

    ok, reason = _validate_compiled(compiled_text)
    if not ok:
        logger.warning("%s compiled validation failed: %s", ctx, reason)
        return None

    return {
        "system": _sys_compiler(adapter),
        "prompt": intl_text,
        "completion": compiled_text,
        "metadata": {"category": "A", "adapter": adapter, "construct": construct, "idx": idx},
    }


async def _generate_pair_B(
    client: anthropic.AsyncAnthropic,
    adapter: str,
    construct: str,
    idx: int,
    domain: str,
    semaphore: asyncio.Semaphore,
    spec: str,
) -> dict | None:
    """Category B: Base INTL → compiled + PATCH block → patched code. (4 API calls)"""
    ctx = f"[B/idx={idx}]"
    display = _adapter_display(adapter)
    base_id = f"f{idx:04d}"
    patch_id = f"p{idx:04d}"

    # Step 1: Generate base INTL
    writer_prompt = (
        f"Write a realistic INTL FUNCTION block for the '{domain}' domain.\n"
        f"Block ID: {base_id}\n"
        f"Must use '{construct}' in the body. Include INTENT, PRECONDITION, READS or MUTATES.\n"
        f"Return ONLY the INTL block."
    )
    base_intl = await _call(client, _sys_intl_writer(spec), writer_prompt, semaphore, ctx + " step1")
    if base_intl is None:
        return None
    base_intl = _extract_intl_block(base_intl)
    ok, reason = _validate_intl(base_intl)
    if not ok:
        logger.warning("%s base INTL failed: %s", ctx, reason)
        return None

    # Step 2: Compile base INTL
    compiled_base = await _call(
        client, _sys_compiler(adapter),
        f"Compile this INTL block to {display} code:\n\n{base_intl}",
        semaphore, ctx + " step2",
    )
    if compiled_base is None:
        return None
    compiled_base = _strip_fences(compiled_base)
    ok, reason = _validate_compiled(compiled_base)
    if not ok:
        logger.warning("%s compiled base failed: %s", ctx, reason)
        return None

    # Step 3: Generate PATCH block
    patch_ideas = [
        "add rate limiting before the main logic",
        "add caching to avoid redundant reads",
        "add an audit log entry on mutation",
        "add input sanitisation for string fields",
        "add a retry with exponential backoff on transient failures",
    ]
    patch_intent = random.choice(patch_ideas)
    patch_prompt = (
        f"Write an INTL PATCH block that targets function {base_id}.\n"
        f"Patch ID: {patch_id}\n"
        f"INTENT: \"{patch_intent}\"\n"
        f"The patch must be specific and actionable — not vague.\n"
        f"Return ONLY the INTL PATCH block."
    )
    patch_intl = await _call(client, _sys_intl_writer(spec), patch_prompt, semaphore, ctx + " step3")
    if patch_intl is None:
        return None
    patch_intl = _extract_intl_block(patch_intl)
    if "PATCH" not in patch_intl or "INTENT" not in patch_intl:
        logger.warning("%s PATCH block missing PATCH/INTENT", ctx)
        return None

    # Step 4: Apply patch
    apply_prompt = (
        f"Apply this INTL PATCH to the existing {display} code.\n\n"
        f"EXISTING CODE:\n{compiled_base}\n\n"
        f"INTL PATCH:\n{patch_intl}"
    )
    patched_code = await _call(
        client, _sys_patch_compiler(adapter), apply_prompt, semaphore, ctx + " step4",
    )
    if patched_code is None:
        return None
    patched_code = _strip_fences(patched_code)
    ok, reason = _validate_compiled(patched_code)
    if not ok:
        logger.warning("%s patched code failed: %s", ctx, reason)
        return None

    # Training pair: prompt = existing code + PATCH block, completion = patched code
    training_prompt = f"{compiled_base}\n\n{patch_intl}"
    return {
        "system": _sys_patch_compiler(adapter),
        "prompt": training_prompt,
        "completion": patched_code,
        "metadata": {"category": "B", "adapter": adapter, "construct": construct, "idx": idx},
    }


async def _generate_pair_C(
    client: anthropic.AsyncAnthropic,
    adapter: str,
    construct: str,
    idx: int,
    domain: str,
    semaphore: asyncio.Semaphore,
    spec: str,
) -> dict | None:
    """Category C: INTL + broken compile + failing checks → corrected code. (3 API calls)"""
    ctx = f"[C/idx={idx}]"
    display = _adapter_display(adapter)
    block_id = f"f{idx:04d}"

    # Step 1: Generate INTL block
    writer_prompt = (
        f"Write a realistic INTL FUNCTION block for the '{domain}' domain.\n"
        f"Block ID: {block_id}\n"
        f"Must use '{construct}'. Include INTENT, PRECONDITION, MUTATES.\n"
        f"Return ONLY the INTL block."
    )
    intl_text = await _call(client, _sys_intl_writer(spec), writer_prompt, semaphore, ctx + " step1")
    if intl_text is None:
        return None
    intl_text = _extract_intl_block(intl_text)
    ok, reason = _validate_intl(intl_text)
    if not ok:
        logger.warning("%s INTL failed: %s", ctx, reason)
        return None

    # Step 2: Generate a realistically broken compilation
    error_types = [
        "missing INTL:BEGIN/END sentinels",
        "PRECONDITION not enforced — missing guard clause",
        "MUTATES field not implemented — no actual write operation",
        "OBSERVABLE field ignored — no logging call present",
        "contains a TODO or stub instead of real implementation",
    ]
    chosen_error = random.choice(error_types)
    broken_prompt = (
        f"Compile this INTL block to {display} code, but deliberately introduce this flaw: {chosen_error}\n"
        f"The output should look plausible but have that specific problem.\n\n"
        f"{intl_text}"
    )
    broken_code = await _call(
        client, _sys_compiler(adapter), broken_prompt, semaphore, ctx + " step2",
    )
    if broken_code is None:
        return None

    # Step 3: Fix the broken code
    fix_prompt = (
        f"The following {display} compilation of an INTL block has a flaw.\n\n"
        f"INTL BLOCK:\n{intl_text}\n\n"
        f"FAILED OUTPUT:\n{broken_code}\n\n"
        f"FAILING CHECK: {chosen_error}\n\n"
        f"Return the fully corrected {display} code with proper INTL:BEGIN/END sentinels."
    )
    corrected_code = await _call(
        client, _sys_escalation(adapter), fix_prompt, semaphore, ctx + " step3",
    )
    if corrected_code is None:
        return None
    corrected_code = _strip_fences(corrected_code)
    ok, reason = _validate_compiled(corrected_code)
    if not ok:
        logger.warning("%s corrected code failed: %s", ctx, reason)
        return None

    # Training pair: prompt = INTL + broken output + check, completion = corrected
    training_prompt = (
        f"{intl_text}\n\n"
        f"# PREVIOUS ATTEMPT (failed):\n{broken_code}\n\n"
        f"# FAILING CHECK: {chosen_error}"
    )
    return {
        "system": _sys_escalation(adapter),
        "prompt": training_prompt,
        "completion": corrected_code,
        "metadata": {"category": "C", "adapter": adapter, "construct": construct, "idx": idx},
    }


# ── Batch INTL writer (Cat A optimisation) ────────────────────────────────────

async def _generate_intl_batch(
    client: anthropic.AsyncAnthropic,
    constructs: list[str],
    idx_start: int,
    domains: list[str],
    semaphore: asyncio.Semaphore,
    spec: str,
) -> list[str]:
    """Write N INTL FUNCTION blocks in a single API call. Returns validated blocks (may be < N)."""
    n = len(constructs)
    ctx = f"[batch/idx={idx_start}-{idx_start+n-1}]"
    items = "\n".join(
        f"Block {i+1}: id=f{idx_start+i:04d}, construct='{constructs[i]}', domain='{domains[i]}'"
        for i in range(n)
    )
    writer_prompt = (
        f"Write {n} separate, realistic INTL FUNCTION blocks — one for each item below.\n"
        f"Separate blocks with a line containing only '---'.\n\n"
        f"{items}\n\n"
        f"Each block must:\n"
        f"- Use the specified block ID exactly\n"
        f"- Use the specified construct prominently in the body\n"
        f"- Include INTENT, at least one PRECONDITION, and READS or MUTATES\n"
        f"- Be complete and realistic — not a toy example\n"
        f"Return ONLY the {n} blocks separated by '---', nothing else."
    )
    raw = await _call(client, _sys_intl_writer(spec), writer_prompt, semaphore, ctx)
    if raw is None:
        return []

    blocks = []
    for chunk in raw.split("---"):
        chunk = chunk.strip()
        if not chunk:
            continue
        extracted = _extract_intl_block(chunk)
        ok, reason = _validate_intl(extracted)
        if ok:
            blocks.append(extracted)
        else:
            logger.warning("%s skipping block: %s", ctx, reason)
    return blocks


# ── Single pair dispatcher ────────────────────────────────────────────────────

async def _generate_one(
    client: anthropic.AsyncAnthropic,
    adapter: str,
    category: str,
    construct: str,
    idx: int,
    semaphore: asyncio.Semaphore,
    spec: str,
    prefetched_intl: str | None = None,
) -> dict | None:
    domain = random.choice(DOMAINS)
    if category == "A":
        return await _generate_pair_A(
            client, adapter, construct, idx, domain, semaphore, spec, prefetched_intl
        )
    if category == "B":
        return await _generate_pair_B(client, adapter, construct, idx, domain, semaphore, spec)
    return await _generate_pair_C(client, adapter, construct, idx, domain, semaphore, spec)


# ── Batch orchestrator ────────────────────────────────────────────────────────

CHECKPOINT_INTERVAL = 50  # write checkpoint every N successful pairs


async def _generate_all(
    adapter: str,
    total: int,
    offset: int,
    checkpoint_path: Path,
) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    spec = _load_spec()

    constructs = _construct_schedule(total)
    categories = [_pick_category() for _ in range(total)]

    # Pre-batch INTL generation for Cat A indices
    # Group consecutive Cat A indices into batches of BATCH_SIZE
    a_indices = [i for i, cat in enumerate(categories) if cat == "A"]
    intl_pool: dict[int, str] = {}  # idx → pre-generated INTL block

    for batch_start in range(0, len(a_indices), BATCH_SIZE):
        batch_idx = a_indices[batch_start: batch_start + BATCH_SIZE]
        batch_constructs = [constructs[i] for i in batch_idx]
        batch_domains = [random.choice(DOMAINS) for _ in batch_idx]
        abs_indices = [offset + i for i in batch_idx]

        blocks = await _generate_intl_batch(
            client, batch_constructs, abs_indices[0], batch_domains, semaphore, spec
        )
        for local_i, block in zip(batch_idx, blocks):
            intl_pool[local_i] = block

        await asyncio.sleep(PAIR_DELAY)

    pairs: list[dict] = []
    failed = 0
    done = 0
    t_start = time.time()

    for i in range(total):
        prefetched = intl_pool.get(i)
        result = await _generate_one(
            client, adapter, categories[i], constructs[i], offset + i, semaphore, spec, prefetched
        )
        done += 1
        if result is not None:
            pairs.append(result)
            if len(pairs) % CHECKPOINT_INTERVAL == 0:
                _write_jsonl(checkpoint_path, pairs)
                elapsed = time.time() - t_start
                rate = done / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    "checkpoint: %d/%d done — %d good, %d failed — %.1f pairs/min",
                    done, total, len(pairs), failed, rate,
                )
        else:
            failed += 1

        if done % 5 == 0 or done == total:
            logger.info("progress: %d/%d — %d good, %d failed", done, total, len(pairs), failed)

        await asyncio.sleep(PAIR_DELAY)

    logger.info("Done: %d good, %d failed / %d requested", len(pairs), failed, total)
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
    val_path = out_dir / "val.jsonl"
    checkpoint = out_dir / "train.jsonl.tmp"

    existing_train = _read_jsonl(train_path) if (append and train_path.exists()) else []
    existing_val = _read_jsonl(val_path) if (append and val_path.exists()) else []

    need_train = max(0, count - len(existing_train))
    need_val = max(0, val_count - len(existing_val))
    total_needed = need_train + need_val

    if dry_run:
        # A=2 calls, B=4 calls, C=3 calls; weighted avg ≈ 2.8 calls/pair
        avg_calls = 2 * 0.6 + 4 * 0.3 + 3 * 0.1
        est_calls = int(total_needed * avg_calls)
        logger.info(
            "[DRY RUN] %d train + %d val pairs for %s — ~%d API calls, model=%s, workers=%d",
            need_train, need_val, adapter, est_calls, MODEL, CONCURRENCY,
        )
        return train_path, val_path

    if total_needed == 0:
        logger.info("Already have %d train + %d val — nothing to do.", len(existing_train), len(existing_val))
        return train_path, val_path

    avg_calls = 2 * 0.6 + 4 * 0.3 + 3 * 0.1
    logger.info(
        "Generating %d pairs for %s — model=%s, workers=%d, ~%.0f API calls total",
        total_needed, adapter, MODEL, CONCURRENCY, total_needed * avg_calls,
    )

    offset = len(existing_train) + len(existing_val)
    new_pairs = asyncio.run(_generate_all(adapter, total_needed, offset, checkpoint))

    random.shuffle(new_pairs)
    new_val = new_pairs[:need_val]
    new_train = new_pairs[need_val:]

    all_train = existing_train + new_train
    all_val = existing_val + new_val

    _write_jsonl(train_path, all_train)
    _write_jsonl(val_path, all_val)

    if checkpoint.exists():
        checkpoint.unlink()

    logger.info("Wrote %d train → %s", len(all_train), train_path)
    logger.info("Wrote %d val   → %s", len(all_val), val_path)
    return train_path, val_path


# ── JSONL helpers ─────────────────────────────────────────────────────────────

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

    token = os.environ.get("HF_TOKEN", "")
    hf_user = os.environ.get("HF_USERNAME", "")
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
    logger.info("Pushed %s → %s", adapter, url)
    return url


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="INTL Training Data Generator (two-step pipeline)")
    p.add_argument("--adapter", required=True, help="Adapter name, e.g. python_fastapi")
    p.add_argument("--count", type=int, default=1000, help="Target training pairs")
    p.add_argument("--val", type=int, default=200, help="Target validation pairs")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--push", action="store_true", help="Push to HuggingFace after generation")
    p.add_argument("--append", action="store_true", help="Resume from existing output")
    p.add_argument("--dry-run", action="store_true", help="Estimate without generating")
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
