"""INTL Compiler Engine — Qwen2.5-Coder-3B + LoRA adapter (§1.2, §5.5).

Loads the base model with a LoRA adapter via PEFT, generates compiled output
at temperature 0.1, and wraps the output in INTL:BEGIN/END sentinels.

The compiler is adapter-aware: each profile loads a separate LoRA checkpoint
from HuggingFace (or a local path). Adapters are cached in memory so repeated
calls with the same profile don't reload the model.

Escalation:
  When the Validator returns failed checks, compile() raises EscalationRequired
  with the full escalation_package attached. The caller can then invoke the
  frontier model (Claude) with that package.
"""
from __future__ import annotations

import os
import re
import textwrap
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sentinel helpers ──────────────────────────────────────────────────────────
BEGIN_TPL = "# ═══ INTL:BEGIN [id={id}] {name} ═══"
END_TPL   = "# ═══ INTL:END   [id={id}] {name} ═══"

# SQL / HTML / CSS use -- or <!-- --> comment styles
_COMMENT_STYLES: dict[str, tuple[str, str]] = {
    "sql":        ("-- ═══ INTL:BEGIN [id={id}] {name} ═══",
                   "-- ═══ INTL:END   [id={id}] {name} ═══"),
    "html":       ("<!-- ═══ INTL:BEGIN [id={id}] {name} ═══ -->",
                   "<!-- ═══ INTL:END   [id={id}] {name} ═══ -->"),
    "css":        ("/* ═══ INTL:BEGIN [id={id}] {name} ═══ */",
                   "/* ═══ INTL:END   [id={id}] {name} ═══ */"),
}

def _sentinel_pair(block_id: str, name: str, profile: str) -> tuple[str, str]:
    for prefix, (bfmt, efmt) in _COMMENT_STYLES.items():
        if profile.startswith(prefix):
            return bfmt.format(id=block_id, name=name), efmt.format(id=block_id, name=name)
    return BEGIN_TPL.format(id=block_id, name=name), END_TPL.format(id=block_id, name=name)


def wrap_sentinels(code: str, block_id: str, name: str, profile: str) -> str:
    """Wrap *code* in INTL:BEGIN/END sentinels."""
    begin, end = _sentinel_pair(block_id, name, profile)
    return f"{begin}\n{code.rstrip()}\n{end}"


def strip_sentinels(code: str) -> str:
    """Remove INTL:BEGIN/END sentinel lines from *code*."""
    lines = [
        ln for ln in code.splitlines()
        if not re.search(r'INTL:(BEGIN|END)', ln, re.I)
    ]
    return "\n".join(lines)


# ── System prompt builder ─────────────────────────────────────────────────────
def _system_prompt(profile: str) -> str:
    lang_name = profile.replace("_", " ").title()
    return (
        f"You are the INTL compiler for {lang_name}. "
        f"Your task is to translate INTL (Intent Language) blocks into idiomatic, "
        f"production-ready {lang_name} code. "
        f"Output ONLY the compiled code — no explanations, no markdown fences. "
        f"Wrap the output in INTL:BEGIN [id=<id>] and INTL:END [id=<id>] sentinels."
    )


# ── Exceptions ────────────────────────────────────────────────────────────────
class CompilerError(Exception):
    """Raised when compilation fails unrecoverably."""


class EscalationRequired(Exception):
    """Raised when validator checks fail; carries the escalation package."""
    def __init__(self, message: str, escalation_package: dict):
        super().__init__(message)
        self.escalation_package = escalation_package


# ── Compiler result ───────────────────────────────────────────────────────────
@dataclass
class CompileResult:
    block_id: str
    name: str
    profile: str
    code: str                       # Final compiled code with sentinels
    raw_output: str                 # Raw model output
    validation_passed: bool
    escalated: bool = False
    escalation_code: str = ""       # Code produced by frontier after escalation


# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict[str, tuple] = {}   # profile → (model, tokenizer)


def _load_model(profile: str, adapter_id: str, base_model: str, device: str = "auto"):
    """Load base model + LoRA adapter with 4-bit quantisation (NF4 QLoRA)."""
    if profile in _model_cache:
        return _model_cache[profile]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
    except ImportError as e:
        raise CompilerError(
            "transformers / peft / bitsandbytes not installed. "
            "Run: pip install transformers peft bitsandbytes"
        ) from e

    logger.info("Loading base model %s for profile %s …", base_model, profile)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map=device,
        trust_remote_code=True,
    )
    # Load LoRA adapter if it looks like a real HF path (contains "/")
    if "/" in adapter_id and not adapter_id.startswith("./"):
        logger.info("Loading LoRA adapter %s …", adapter_id)
        try:
            model = PeftModel.from_pretrained(model, adapter_id)
        except Exception as exc:
            logger.warning("Could not load adapter %s: %s. Running base model.", adapter_id, exc)

    _model_cache[profile] = (model, tokenizer)
    return model, tokenizer


def _generate(
    model,
    tokenizer,
    system: str,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Frontier escalation ───────────────────────────────────────────────────────
def _escalate_to_frontier(escalation_package: dict) -> str:
    """Call Claude to fix the failed compilation. Returns corrected code."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise CompilerError("ANTHROPIC_API_KEY not set; cannot escalate to frontier")

    client = anthropic.Anthropic(api_key=api_key)
    block_id = escalation_package["block_id"]
    intl_block = escalation_package["intl_block"]
    failed_code = escalation_package["compiled_code"]
    failed_checks = escalation_package["failed_checks"]

    checks_txt = "\n".join(
        f"- {c['name']}: {c['message']}" + (f" ({c['detail']})" if c.get("detail") else "")
        for c in failed_checks
    )
    user_msg = textwrap.dedent(f"""
        The INTL compiler produced code that failed validation.

        ## INTL Block
        {intl_block}

        ## Failed Compiled Output
        ```
        {failed_code}
        ```

        ## Failed Checks
        {checks_txt}

        Please produce a corrected, complete, production-ready implementation.
        Output ONLY the corrected code. No explanations. No markdown fences.
    """).strip()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ── Main compiler class ───────────────────────────────────────────────────────
class Compiler:
    """INTL → target-language compiler.

    Typical usage::

        from intl.compiler import Compiler
        from intl.router import route

        compiler = Compiler()
        result = compiler.compile(intl_block, block_id="f001", name="login",
                                  profile="python_fastapi")
    """

    def __init__(self, device: str = "auto", max_new_tokens: int = 1024):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._validator = None   # lazy import

    @property
    def validator(self):
        if self._validator is None:
            from intl.validator import Validator
            self._validator = Validator()
        return self._validator

    def compile(
        self,
        intl_block: str,
        block_id: str,
        name: str,
        profile: str,
        *,
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        mutates: list[str] | None = None,
        emits: list[str] | None = None,
        observable: bool = False,
        return_type: str | None = None,
        escalate: bool = True,
    ) -> CompileResult:
        """Compile an INTL block to target-language code.

        Args:
            intl_block: Raw INTL source for the function/pipeline/type block.
            block_id: INTL block ID (e.g. ``"f001"``).
            name: Block name (e.g. ``"login"``).
            profile: INTL PROFILE (e.g. ``"python_fastapi"``).
            escalate: If True, failed validation auto-escalates to frontier model.

        Returns:
            :class:`CompileResult`

        Raises:
            :class:`EscalationRequired`: If escalate=False and validation fails.
            :class:`CompilerError`: On unrecoverable errors.
        """
        from intl.router import route, RouterError
        try:
            route_result = route(profile)
        except RouterError as e:
            raise CompilerError(str(e)) from e

        # Load model
        model, tokenizer = _load_model(
            profile, route_result.adapter_id, route_result.base_model, self.device
        )

        # Generate
        system = _system_prompt(profile)
        raw = _generate(model, tokenizer, system, intl_block,
                        max_new_tokens=self.max_new_tokens)

        # Ensure sentinels
        if "INTL:BEGIN" not in raw:
            raw = wrap_sentinels(raw, block_id, name, profile)

        # Validate
        vr = self.validator.validate(
            raw, block_id, profile,
            intl_block=intl_block,
            preconditions=preconditions,
            postconditions=postconditions,
            mutates=mutates,
            emits=emits,
            observable=observable,
            return_type=return_type,
        )

        if vr.passed:
            return CompileResult(
                block_id=block_id, name=name, profile=profile,
                code=raw, raw_output=raw, validation_passed=True,
            )

        # Validation failed
        logger.warning("Validation failed for [id=%s] %s — %d check(s) failed",
                       block_id, name, len(vr.failed_checks))

        if not escalate:
            raise EscalationRequired(
                f"Validation failed for block {block_id}", vr.escalation_package
            )

        # Escalate to frontier
        logger.info("Escalating block [id=%s] %s to frontier model …", block_id, name)
        try:
            fixed = _escalate_to_frontier(vr.escalation_package)
            if "INTL:BEGIN" not in fixed:
                fixed = wrap_sentinels(fixed, block_id, name, profile)
        except Exception as e:
            raise CompilerError(f"Frontier escalation failed: {e}") from e

        return CompileResult(
            block_id=block_id, name=name, profile=profile,
            code=fixed, raw_output=raw, validation_passed=False,
            escalated=True, escalation_code=fixed,
        )

    def compile_module(self, module, *, escalate: bool = True) -> list[CompileResult]:
        """Compile all dirty blocks in a :class:`~intl.parser.Module`.

        Returns list of :class:`CompileResult` (one per block).
        """
        results = []
        for block in module.blocks:
            bid = getattr(block, "id", None)
            bname = getattr(block, "name", "")
            if not bid:
                logger.warning("Block %s has no id — skipping", bname)
                continue
            intl_src = f"# block {bname} [{bid}]"   # simplified; real impl serialises block
            result = self.compile(
                intl_src, bid, bname, module.profile,
                preconditions=getattr(block, "preconditions", None),
                postconditions=getattr(block, "postconditions", None),
                mutates=getattr(block, "mutates", None),
                emits=getattr(block, "emits", None),
                observable=getattr(block, "observable", False),
                escalate=escalate,
            )
            results.append(result)
        return results
