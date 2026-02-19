# INTL Compiler

**INTL (Intent Language)** is a structured intermediate representation designed to be written by frontier AI models and compiled into idiomatic, production-ready code in 24 target languages and frameworks.

Small fine-tuned **LoRA adapters on Qwen2.5-Coder-3B** do the heavy lifting — each adapter specialises in one target language/framework combination, trained on thousands of INTL → code pairs.

---

## Architecture

```
User Requirement (freeform)
        │
        ▼
  generator.py  ──── decompose() ────────────► Project Manifest (JSON)
        │
        ▼
  generator.py  ──── generate_module() ──────► .intl files  (parallel, one per module)
        │
        ▼
  parser.py     ──── parse() ────────────────► Typed AST
        │
        ▼
  index.py      ──── index_module() ─────────► SQLite Semantic Index
        │
        ▼
  router.py     ──── route() ────────────────► Adapter name  (O(1) lookup)
        │
        ▼
  compiler.py   ──── compile() ──────────────► Target code  (Qwen2.5-Coder-3B + LoRA)
        │
        ▼
  validator.py  ──── validate() ─────────────► T1–T7 checks
        │ fail
        ▼
  escalation.py ──── escalate() ─────────────► Corrected code  (Claude frontier model)
        │
        ▼
  generator.py  ──── patch() ────────────────► PATCH blocks  (incremental updates)
```

---

## How It Works

1. **Generator** (`generator.py`) — calls Claude to decompose a freeform requirement into a project manifest, then generates `.intl` source files for each module in parallel. Incremental updates produce PATCH blocks.
2. **Parser** (`grammar.lark` + `parser.py`) — validates INTL source against a Lark EBNF grammar and produces a typed AST (dataclasses). 24 constructs, precise line/column errors.
3. **Semantic Index** (`index.py`) — SQLite database tracking every compiled block, dirty state, dependency graph, and compilation history.
4. **LoRA Router** (`router.py`) — O(1) lookup table mapping language profile strings to adapter names. No inference required.
5. **Compiler** (`compiler.py`) — loads Qwen2.5-Coder-3B with the appropriate LoRA adapter via PEFT. Temperature 0.1. Wraps every output in `INTL:BEGIN / INTL:END` sentinels.
6. **Validator** (`validator.py`) — seven deterministic checks (T1–T7) on every compiled output. Returns a `ValidationResult` with pass/fail details and an escalation package on failure.
7. **Escalation** (`escalation.py`) — Claude frontier layer invoked when the LoRA retry budget is exhausted. Corrections are saved as Category C training pairs, continuously improving adapters.
8. **Training Data Generator** (`datagen.py`) — generates JSONL training pairs (A/B/C split) for any adapter using Claude.
9. **CLI** (`cli.py`) — `intl compile`, `intl build`, `intl status`, `intl adapters`, `intl validate`.

---

## 24 Target Adapters

| Phase | Adapters | Status |
|-------|----------|--------|
| **0** | `python_fastapi`, `sql_postgres` | 🔲 Training pending |
| **1** | `python_django`, `python_flask`, `typescript_express`, `sql_mysql` | 🔲 Training pending |
| **2** | `typescript_nextjs`, `php_laravel`, `php_vanilla`, `javascript_vanilla`, `html_jinja2`, `html_blade`, `css_tailwind` | 🔲 Training pending |
| **3** | `java_spring`, `csharp_dotnet`, `go_gin`, `ruby_rails`, `sql_tsql`, `sql_sqlite` | 🔲 Training pending |
| **4** | `swift_ios`, `kotlin_android`, `dart_flutter` | 🔲 Training pending |
| **5** | `rust_axum`, `cpp_modern` | 🔲 Training pending |

---

## Component Status

| Component | File | Tests | Status |
|-----------|------|-------|--------|
| Lark Grammar | `intl/grammar.lark` | — | ✅ Done |
| Parser | `intl/parser.py` | `tests/test_parser.py` | ✅ Done |
| Semantic Index | `intl/index.py` | `tests/test_index.py` | ✅ Done |
| LoRA Router | `intl/router.py` | `tests/test_router.py` | ✅ Done |
| Validator T1–T7 | `intl/validator.py` | `tests/test_validator.py` | ✅ Done |
| Training Data Gen | `intl/datagen.py` | — | ✅ Done |
| Compiler Engine | `intl/compiler.py` | — | ✅ Done |
| CLI | `intl/cli.py` | — | ✅ Done |
| Generator | `intl/generator.py` | `tests/test_generator.py` | ✅ Done |
| Escalation | `intl/escalation.py` | `tests/test_escalation.py` | ✅ Done |
| **Total tests** | | **193 / 193 passing** | ✅ |

---

## Quick Start

```bash
pip install -e .

# Compile a single .intl file
intl compile mymodule.intl --profile python_fastapi

# Build an entire project (auto-discovers .intl files)
intl build project/

# Check compilation status
intl status

# List available adapters
intl adapters

# Validate a compiled output
intl validate output.py --profile python_fastapi
```

---

## Project Structure

```
intl-compiler/
├── README.md
├── SPEC.md                        ← mirror of INTL_Specification.md
├── intl/
│   ├── grammar.lark               ← Lark EBNF — 24 constructs
│   ├── parser.py                  ← typed AST dataclasses
│   ├── index.py                   ← SQLite semantic index
│   ├── router.py                  ← O(1) adapter lookup
│   ├── compiler.py                ← Qwen2.5-Coder-3B + LoRA via PEFT
│   ├── validator.py               ← T1–T7 validation pipeline
│   ├── datagen.py                 ← JSONL training pair generator (Claude)
│   ├── generator.py               ← INTL source generator (Claude)
│   ├── escalation.py              ← Frontier correction layer (Claude)
│   ├── cli.py                     ← CLI entry point
│   ├── generated/                 ← .intl files produced by generator.py
│   └── tests/
│       ├── test_parser.py
│       ├── test_index.py
│       ├── test_router.py
│       ├── test_validator.py
│       ├── test_generator.py
│       └── test_escalation.py
├── configs/
│   └── adapters.json              ← adapter registry (24 entries)
├── data/
│   └── <adapter>/
│       ├── train.jsonl            ← A+B pairs (~60/30 split)
│       ├── validation.jsonl       ← 200-pair held-out set
│       └── corrections.jsonl      ← Category C (escalation corrections)
├── docs/
│   └── INTL_Specification.md      ← full language spec
└── scripts/
    └── train_adapter.sh           ← Vast.ai training script
```

---

## Training Data Format

Each JSONL line is a `{system, prompt, completion}` triple:

```jsonl
{
  "system": "You are the INTL compiler for Python FastAPI...",
  "prompt": "FUNCTION login [id=f001]\n  INTENT ...",
  "completion": "# ═══ INTL:BEGIN [id=f001] login ═══\nasync def login(...): ..."
}
```

| Category | Split | Description |
|----------|-------|-------------|
| A | ~60% | Fresh INTL → target code |
| B | ~30% | PATCH blocks |
| C | ~10% | Error correction (also from escalation) |

200 validation pairs per adapter, held out from training.

---

## Validation Checks (T1–T7)

| Check | Name | Description |
|-------|------|-------------|
| T1 | Syntax | Output is syntactically valid for the target language |
| T2 | Sentinels | `INTL:BEGIN` / `INTL:END` present with correct block ID |
| T3 | Preconditions | Every `PRECONDITION` token appears in compiled code |
| T4 | Postconditions | Every `POSTCONDITION` token appears in compiled code |
| T5 | Side Effects | `MUTATES`/`OBSERVABLE` declarations honoured |
| T6 | Types | Return type matches `RETURN` declaration |
| T7 | No Placeholders | No `TODO`, `FIXME`, `unimplemented!()` etc. in output |

Failed checks trigger re-compilation (up to retry budget), then escalation to Claude.

---

## Training Cost

Total estimated training cost: **~$2.85** across all 24 adapters on Vast.ai RTX 4090 instances (~$0.12/adapter).

---

## Models

| Role | Model |
|------|-------|
| Base compiler | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| LoRA adapters | `$HF_USERNAME/intl-adapters` (HuggingFace) |
| Generator / Escalation | `claude-sonnet-4-6` |
| Training data gen | `claude-sonnet-4-6` |

---

## License

Proprietary — Confidential.
