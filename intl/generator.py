"""INTL Generator — Frontier Claude layer that converts freeform requirements
into INTL source files and produces PATCH blocks for incremental updates.

Three public functions:
  decompose(requirement)           → project manifest dict
  generate_module(manifest, spec)  → .intl source string
  patch(module_intl, change_req)   → PATCH block string

All Claude calls use claude-sonnet-4-6 at temperature 0 for determinism.
Prompt caching is applied on stable context sections to reduce token costs.

Generated .intl files are written to /workspace/intl/generated/ and validated
through parser.py before being returned to the caller.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import anthropic

from intl.parser import parse_safe, ParseError

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0
GENERATED_DIR = Path(__file__).parent / "generated"

# Compressed INTL syntax reference (~3K tokens) — injected as a cached block
# in generate_module() calls.
_SYNTAX_REF = textwrap.dedent("""\
    === INTL SYNTAX REFERENCE (v0.1) ===

    FILE STRUCTURE
    --------------
    MODULE <name> [id=<mXXX>]
    TARGET     "<path>"
    PROFILE    <adapter>          # python_fastapi | sql_postgres | typescript_express | …
    NAMESPACE  "<ns>"
    REQUIRES   <mXXX> <mXXX>     # space-separated module deps (optional)
    VERSION    "<semver>"         # optional

    FUNCTION <name> [id=<fXXX>]
    INTENT       "<one sentence>"
    PRECONDITION <expr>           # repeatable; compiles to guard clause
    POSTCONDITION <expr>          # repeatable; compiles to assertion / return constraint
    READS        <table>          # optional
    MUTATES      <table>          # optional
    EMITS        <event>          # optional
    OBSERVABLE                    # optional flag → logging wrapper
    CONFIDENCE   <0.0-1.0>        # optional; <0.8 triggers escalation review
    TIMEOUT      <dur>            # optional; e.g. 30s
    <body>
    END FUNCTION <name> [id=<fXXX>]

    PIPELINE <name> [id=<pXXX>]
    INTENT  "<one sentence>"
    STEP <n>: <StepName> → <func_ref>
    END PIPELINE <name> [id=<pXXX>]

    TYPE <name> [id=<tXXX>]
    FIELD <name>: <Type> [OPTIONAL] [DEFAULT <value>]
    END TYPE <name> [id=<tXXX>]

    PATCH [target=<id>] [position=before|after|replace]
    PRECONDITION <current_state>
    <new_intl_body>
    END PATCH

    BODY CONSTRUCTS
    ---------------
    QUERY <table> WHERE <expr> [LIMIT <n>]   → SELECT
    IF <expr> THEN <stmt> [ELSE <stmt>]
    IF <expr> THEN FAIL <ErrorType>("<msg>")
    FOREACH <var> IN <collection> DO <stmt>
    PARALLEL DO <stmt> AND <stmt> [AND …]
    FALLBACK <primary_expr> OR <fallback_expr>
    CACHE <key> TTL <dur>
    VALIDATE <expr> [FORMAT <fmt>] [ELSE FAIL <ErrorType>("<msg>")]
    TRANSACTION DO <stmt> [AND <stmt> …]
    PERSIST <table> INSERT (<k>: <v>, …)
    PERSIST <table> UPDATE WHERE <expr> SET (<k>: <v>, …)
    PERSIST <table> DELETE WHERE <expr>
    CALL <func_ref>(<args>)
    COMPUTE <var> = <expr>
    RETURN <expr>
    ASSERT <expr>
    EMIT <event>(<payload>)
    LOG <expr>
    AWAIT <expr>
    STREAM <source> WHERE <expr>
    PAGINATE <query> PAGE <page> SIZE <size>

    OPERATORS: ==, !=, >, <, >=, <=, AND, OR, NOT, IS NULL, IS NOT NULL
    LITERALS: "string", 42, 3.14, true/false, null, 30s, 24h, 7d
    NOW() — current timestamp
    <name>.length, <name>.format, etc. — attribute access
""")


# ── Errors ────────────────────────────────────────────────────────────────────
class GeneratorError(Exception):
    """Raised when decompose/generate/patch fails."""


# ── Claude client (lazy) ──────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise GeneratorError("ANTHROPIC_API_KEY environment variable not set.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _chat(messages: list[dict], system: list[dict] | str, max_tokens: int = 4096) -> str:
    """Send a message to Claude and return the text response."""
    client = _get_client()
    if isinstance(system, str):
        system_block = [{"type": "text", "text": system}]
    else:
        system_block = system

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        system=system_block,
        messages=messages,
    )
    return response.content[0].text.strip()


# ── decompose() ───────────────────────────────────────────────────────────────
_DECOMPOSE_SYSTEM = textwrap.dedent("""\
    You are an INTL architect. You decompose software requirements into a
    structured project manifest for the INTL compiler.

    INTL (Intent Language) is a structured intermediate representation. Frontier
    AI models write INTL; small LoRA adapters compile it into production code.

    Your job: given a freeform requirement, output a JSON project manifest.

    MANIFEST SCHEMA
    ---------------
    {
      "project": "<short_name>",
      "description": "<one sentence>",
      "modules": [
        {
          "id": "<mXXX>",            // globally unique, e.g. m001
          "name": "<snake_case>",
          "target": "<relative/path/to/output.ext>",
          "profile": "<adapter>",    // e.g. python_fastapi, sql_postgres
          "namespace": "<fully.qualified.ns>",
          "requires": ["<mXXX>"],    // IDs of modules this depends on
          "description": "<what this module does>",
          "functions": [
            {"id": "<fXXX>", "name": "<snake_case>", "description": "<intent>"}
          ],
          "pipelines": [
            {"id": "<pXXX>", "name": "<snake_case>", "description": "<intent>"}
          ],
          "types": [
            {"id": "<tXXX>", "name": "<PascalCase>", "description": "<what it models>"}
          ]
        }
      ],
      "shared_types": [],            // cross-module types (same schema as types above)
      "dependency_graph": {          // module_id → [dep_module_ids]
        "<mXXX>": ["<mXXX>"]
      }
    }

    RULES
    -----
    - IDs: modules m001+, functions f001+, pipelines p001+, types t001+ (globally unique)
    - Choose the most appropriate PROFILE for each module from the 24 adapters:
      python_fastapi, python_django, python_flask, typescript_express, typescript_nextjs,
      sql_postgres, sql_mysql, sql_tsql, sql_sqlite, php_laravel, php_vanilla,
      javascript_vanilla, html_jinja2, html_blade, css_tailwind, java_spring,
      csharp_dotnet, go_gin, ruby_rails, swift_ios, kotlin_android, dart_flutter,
      rust_axum, cpp_modern
    - Keep modules cohesive and single-responsibility
    - Output ONLY valid JSON — no markdown, no explanation
""")


def decompose(requirement: str) -> dict:
    """Decompose a freeform requirement into a project manifest.

    Args:
        requirement: Natural language description of what to build.

    Returns:
        Project manifest dict with modules, IDs, profiles, dependency graph.

    Raises:
        GeneratorError: If Claude returns invalid JSON or the manifest is malformed.
    """
    logger.info("decompose: calling Claude for requirement (%d chars)", len(requirement))

    system = [
        {
            "type": "text",
            "text": _DECOMPOSE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    messages = [
        {
            "role": "user",
            "content": f"REQUIREMENT:\n{requirement}",
        }
    ]

    raw = _chat(messages, system, max_tokens=8192)

    # Strip markdown fences if Claude added them
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeneratorError(f"decompose: Claude returned invalid JSON: {exc}\n\nRaw:\n{raw}") from exc

    # Basic validation
    if "modules" not in manifest or not isinstance(manifest["modules"], list):
        raise GeneratorError(f"decompose: manifest missing 'modules' list.\n\nRaw:\n{raw}")

    logger.info("decompose: manifest has %d modules", len(manifest["modules"]))
    return manifest


# ── generate_module() ─────────────────────────────────────────────────────────
_GENERATE_SYSTEM = textwrap.dedent("""\
    You are an INTL code generator. You write valid INTL source files (.intl)
    from a project manifest and a module specification.

    INTL is a structured intermediate representation. Every block you write will
    be compiled by a small LoRA model into production code. Accuracy of INTL
    syntax is critical — the parser is strict.

    Your output must be a complete, valid .intl file: MODULE block followed by
    FUNCTION, PIPELINE, and TYPE blocks as specified.

    RULES
    -----
    - Follow the INTL syntax reference exactly
    - Every FUNCTION must have INTENT, at least one PRECONDITION, and POSTCONDITION
    - Use only constructs from the syntax reference
    - IDs must match the manifest exactly (do not invent new IDs)
    - Output ONLY the .intl source — no markdown fences, no explanation
    - END tags must include both name and id: END FUNCTION <name> [id=<fXXX>]
""")


def generate_module(manifest: dict, module_spec: dict) -> str:
    """Generate a .intl source file for one module.

    Args:
        manifest: Full project manifest from decompose().
        module_spec: One entry from manifest["modules"].

    Returns:
        Valid .intl source string (also written to /workspace/intl/generated/).

    Raises:
        GeneratorError: If Claude returns invalid INTL or the parser rejects it.
    """
    module_name = module_spec.get("name", "unknown")
    logger.info("generate_module: generating %s", module_name)

    # Build context for modules that this one REQUIRES (for type awareness)
    required_ids = module_spec.get("requires", [])
    required_types: list[str] = []
    for mod in manifest.get("modules", []):
        if mod["id"] in required_ids:
            for t in mod.get("types", []):
                required_types.append(
                    f"TYPE {t['name']} [id={t['id']}]  # {t.get('description', '')}"
                )

    required_types_block = (
        "\n".join(required_types) if required_types else "(none)"
    )

    system = [
        # 1. Syntax reference — cached (stable across all generate calls)
        {
            "type": "text",
            "text": _SYNTAX_REF,
            "cache_control": {"type": "ephemeral"},
        },
        # 2. Project manifest — cached (stable across all modules in one project)
        {
            "type": "text",
            "text": f"=== PROJECT MANIFEST ===\n{json.dumps(manifest, indent=2)}",
            "cache_control": {"type": "ephemeral"},
        },
        # 3. Required module types — cached per dependency group
        {
            "type": "text",
            "text": f"=== TYPES FROM REQUIRED MODULES ===\n{required_types_block}",
            "cache_control": {"type": "ephemeral"},
        },
        # 4. Generation rules — not cached (small, varies per call)
        {
            "type": "text",
            "text": _GENERATE_SYSTEM,
        },
    ]

    messages = [
        {
            "role": "user",
            "content": (
                f"Generate the complete .intl source for this module:\n\n"
                f"{json.dumps(module_spec, indent=2)}"
            ),
        }
    ]

    raw = _chat(messages, system, max_tokens=8192)

    # Strip any markdown fences Claude might add
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    # Validate through parser
    _ast, _err = parse_safe(raw)
    if _err is not None:
        raise GeneratorError(
            f"generate_module: parser rejected INTL for module '{module_name}':\n{_err}\n\n"
            f"Generated source:\n{raw}"
        )

    # Write to disk
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"{module_name}.intl"
    out_path.write_text(raw, encoding="utf-8")
    logger.info("generate_module: wrote %s", out_path)

    return raw


async def _generate_module_async(
    manifest: dict, module_spec: dict, semaphore: asyncio.Semaphore
) -> tuple[str, str]:
    """Async wrapper so we can run generate_module() concurrently."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        intl_source = await loop.run_in_executor(
            None, generate_module, manifest, module_spec
        )
        return module_spec["name"], intl_source


async def generate_all_modules(
    manifest: dict, max_concurrent: int = 4
) -> dict[str, str]:
    """Generate all modules in the manifest concurrently.

    Args:
        manifest: Full project manifest from decompose().
        max_concurrent: Max parallel Claude API calls (default 4).

    Returns:
        Dict of {module_name: intl_source}.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _generate_module_async(manifest, mod, semaphore)
        for mod in manifest.get("modules", [])
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


# ── patch() ───────────────────────────────────────────────────────────────────
_PATCH_SYSTEM = textwrap.dedent("""\
    You are an INTL patch generator. You produce PATCH blocks that describe
    surgical modifications to existing INTL source files.

    A PATCH block targets a specific FUNCTION or PIPELINE by ID and describes
    the minimal change needed. It is NOT a rewrite — only the changed lines.

    PATCH SYNTAX
    ------------
    PATCH [target=<id>] [position=before|after|replace]
    PRECONDITION <current_state_assertion>
    <new_or_modified_intl_body_lines>
    END PATCH

    RULES
    -----
    - target= must be an existing ID from the INTL source
    - position=replace means replace the entire body of the target block
    - position=before/after inserts lines before/after the target
    - PRECONDITION inside PATCH describes what must be true of the *current* code
    - Output ONLY the PATCH block — no explanation, no markdown fences
    - Keep the patch minimal: only include lines that change
""")


def patch(module_intl: str, change_request: str) -> str:
    """Generate a PATCH block for an incremental change to an existing module.

    Args:
        module_intl:    Existing .intl source (NOT compiled output).
        change_request: Natural language description of the change to make.

    Returns:
        PATCH block string (validated to start with 'PATCH [').

    Raises:
        GeneratorError: If Claude returns something that doesn't look like a PATCH block.
    """
    logger.info("patch: generating patch for change_request (%d chars)", len(change_request))

    system = [
        {
            "type": "text",
            "text": _PATCH_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    messages = [
        {
            "role": "user",
            "content": (
                f"EXISTING INTL SOURCE:\n{module_intl}\n\n"
                f"CHANGE REQUEST:\n{change_request}"
            ),
        }
    ]

    raw = _chat(messages, system, max_tokens=2048)

    # Strip markdown fences
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    # Minimal sanity check — must start with PATCH
    if not raw.upper().startswith("PATCH"):
        raise GeneratorError(
            f"patch: Claude did not return a PATCH block.\n\nRaw:\n{raw}"
        )

    logger.info("patch: generated PATCH block (%d chars)", len(raw))
    return raw
