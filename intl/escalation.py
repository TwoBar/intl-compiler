"""INTL Escalation — Frontier Claude layer that corrects failed LoRA compilations.

When the validator exhausts the LoRA retry budget, escalate() is called with the
full structured escalation package. Claude produces corrected code wrapped in INTL
sentinels. The correction is:

  1. Written directly to the semantic index (bypasses the LoRA router).
  2. Saved as a Category C training pair to /workspace/data/<adapter>/corrections.jsonl.

The escalation layer never receives freeform context — structure is always preserved.
"""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-5"
DATA_DIR = Path(__file__).parent.parent / "data"


# ── Errors ────────────────────────────────────────────────────────────────────
class EscalationError(Exception):
    """Raised when escalation itself fails (Claude error, bad output, etc.)."""


# ── Claude client (lazy) ──────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EscalationError("ANTHROPIC_API_KEY environment variable not set.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ── Sentinel detection ────────────────────────────────────────────────────────
_BEGIN_PATTERNS = [
    r"#\s*═+\s*INTL:BEGIN",       # Python / JS / Go / Rust style
    r"--\s*═+\s*INTL:BEGIN",      # SQL style
    r"<!--\s*═+\s*INTL:BEGIN",    # HTML style
    r"/\*\s*═+\s*INTL:BEGIN",     # CSS style
]
_END_PATTERNS = [
    r"#\s*═+\s*INTL:END",
    r"--\s*═+\s*INTL:END",
    r"<!--\s*═+\s*INTL:END",
    r"/\*\s*═+\s*INTL:END",
]


def _has_sentinels(code: str) -> bool:
    has_begin = any(re.search(p, code) for p in _BEGIN_PATTERNS)
    has_end = any(re.search(p, code) for p in _END_PATTERNS)
    return has_begin and has_end


# ── System prompt builder ─────────────────────────────────────────────────────
def _build_system_prompt(language: str, profile: str, adapter: str) -> str:
    return textwrap.dedent(f"""\
        You are the INTL escalation compiler for {language} ({profile}).

        You receive a structured escalation package containing:
          - The original INTL block (the specification)
          - The failed compiled output (what the LoRA model produced)
          - The failing validation checks with error messages

        Your task: produce CORRECTED, PRODUCTION-READY {language} code that
        satisfies all failing checks and compiles correctly for the {profile} adapter.

        CRITICAL RULES
        --------------
        1. Output ONLY code — no explanation, no markdown fences, no commentary.
        2. Wrap your output in INTL sentinels appropriate for {language}:
           - Python/JS/TS/Go/Ruby/Rust/Swift/Kotlin: # ═══ INTL:BEGIN [id={{id}}] {{name}} ═══
           - SQL:  -- ═══ INTL:BEGIN [id={{id}}] {{name}} ═══
           - HTML: <!-- ═══ INTL:BEGIN [id={{id}}] {{name}} ═══ -->
           - CSS:  /* ═══ INTL:BEGIN [id={{id}}] {{name}} ═══ */
        3. The corrected code must address EVERY failing check listed.
        4. Do not modify anything not related to the failing checks.
        5. The INTL block is the authoritative specification — the failed output
           is only context. Base your correction on the INTL block, not the failed output.
        6. This correction will be saved as a training example — make it exemplary.
    """)


# ── escalate() ────────────────────────────────────────────────────────────────
def escalate(
    intl_block: str,
    failed_output: str,
    failing_checks: list[dict],
    language: str,
    adapter: str,
    retry_count: int,
    profile: Optional[str] = None,
    block_id: Optional[str] = None,
    block_name: Optional[str] = None,
    index=None,  # Optional SemanticIndex — if provided, corrected code is stored
) -> str:
    """Correct a failed LoRA compilation using the frontier Claude model.

    Args:
        intl_block:     Original INTL block (the specification). Never the compiled output.
        failed_output:  The code produced by the LoRA model that failed validation.
        failing_checks: List of dicts with keys: name, message, detail.
        language:       Human-readable target language name (e.g. "Python", "SQL").
        adapter:        Adapter name (e.g. "python_fastapi"). Used for training data path.
        retry_count:    How many LoRA retries were attempted before escalation.
        profile:        PROFILE string (defaults to adapter if not provided).
        block_id:       INTL block ID (e.g. "f001"). Used for index write + training pair.
        block_name:     INTL block name (e.g. "login"). Used for training pair.
        index:          Optional SemanticIndex instance. If provided, corrected code is
                        written directly (bypassing LoRA router).

    Returns:
        Corrected code string wrapped in INTL sentinels.

    Raises:
        EscalationError: If Claude fails or returns output without sentinels.
    """
    profile = profile or adapter
    block_id = block_id or "unknown"
    block_name = block_name or "unknown"

    logger.warning(
        "escalate: adapter=%s block=%s retry_count=%d failing=%s",
        adapter, block_id, retry_count,
        [c.get("name") for c in failing_checks],
    )

    # ── Build structured escalation package ───────────────────────────────────
    check_details = "\n".join(
        f"  [{c.get('name', '?')}] {c.get('message', '')} — {c.get('detail', '')}"
        for c in failing_checks
    )

    escalation_package = textwrap.dedent(f"""\
        === ESCALATION PACKAGE ===

        ADAPTER:      {adapter}
        LANGUAGE:     {language}
        BLOCK ID:     {block_id}
        RETRY COUNT:  {retry_count}

        --- ORIGINAL INTL BLOCK (authoritative specification) ---
        {intl_block.strip()}

        --- FAILED COMPILED OUTPUT ---
        {failed_output.strip()}

        --- FAILING VALIDATION CHECKS ---
        {check_details.strip() or "(none listed)"}
        ==========================
    """)

    # ── Call Claude ───────────────────────────────────────────────────────────
    system_prompt = _build_system_prompt(language, profile, adapter)

    client = _get_client()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": escalation_package}
            ],
        )
        corrected = response.content[0].text.strip()
    except anthropic.APIError as exc:
        raise EscalationError(f"escalate: Claude API error: {exc}") from exc

    # Strip any markdown fences Claude might add despite instructions
    corrected = re.sub(r"^```[a-z]*\s*", "", corrected, flags=re.MULTILINE)
    corrected = re.sub(r"```\s*$", "", corrected, flags=re.MULTILINE).strip()

    # Verify sentinels are present
    if not _has_sentinels(corrected):
        raise EscalationError(
            f"escalate: corrected output is missing INTL sentinels.\n\nRaw:\n{corrected}"
        )

    logger.info("escalate: correction produced (%d chars)", len(corrected))

    # ── Write to semantic index (bypasses LoRA router) ────────────────────────
    if index is not None:
        try:
            index.record_compiled(block_id, corrected)
            logger.info("escalate: wrote corrected code to index for block %s", block_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("escalate: failed to write to index: %s", exc)

    # ── Save as Category C training pair ─────────────────────────────────────
    _save_correction_pair(
        intl_block=intl_block,
        failed_output=failed_output,
        failing_checks=failing_checks,
        corrected=corrected,
        language=language,
        adapter=adapter,
        block_id=block_id,
        block_name=block_name,
    )

    return corrected


# ── Training pair persistence ─────────────────────────────────────────────────
def _build_system_message(language: str, adapter: str) -> str:
    """Build the system message used in training pairs (matches datagen.py format)."""
    return (
        f"You are the INTL compiler for {language} ({adapter}). "
        f"Given an INTL block, produce idiomatic, production-ready {language} code. "
        f"Wrap output in INTL:BEGIN / INTL:END sentinels with the correct block ID."
    )


def _save_correction_pair(
    intl_block: str,
    failed_output: str,
    failing_checks: list[dict],
    corrected: str,
    language: str,
    adapter: str,
    block_id: str,
    block_name: str,
) -> None:
    """Append a Category C training pair to /workspace/data/<adapter>/corrections.jsonl."""
    adapter_dir = DATA_DIR / adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = adapter_dir / "corrections.jsonl"

    check_summary = "; ".join(
        f"{c.get('name', '?')}: {c.get('message', '')}" for c in failing_checks
    )

    # Category C prompt format: INTL block + failed output + error summary
    prompt = (
        f"{intl_block.strip()}\n\n"
        f"# PREVIOUS ATTEMPT (failed):\n{failed_output.strip()}\n\n"
        f"# FAILING CHECKS: {check_summary}"
    )

    pair = {
        "system": _build_system_message(language, adapter),
        "prompt": prompt,
        "completion": corrected,
        "metadata": {
            "category": "C",
            "adapter": adapter,
            "block_id": block_id,
            "block_name": block_name,
            "failing_checks": [c.get("name") for c in failing_checks],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    with corrections_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        "escalate: saved Category C pair to %s (block=%s)", corrections_path, block_id
    )
