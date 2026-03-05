# INTL — Intermediate Language Compiler

**AI code generation through an intermediate language.** Claude generates structured INTL source from natural language requirements; fine-tuned LoRA adapters on Qwen2.5-Coder-3B compile it to production code across 24 target languages and frameworks.

## How It Works

```
Natural Language Requirements
    |
Generator (Claude) -> Structured INTL Source
    |
Parser -> AST (Lark grammar)
    |
Semantic Index (SQLite)
    |
LoRA Router -> Select target adapter
    |
Compiler (Qwen2.5-Coder-3B + LoRA) -> Target language code
    |
Validator (T1-T7 checks) -> Production code
    |
[On failure] Escalation -> Claude corrects errors
```

## Why an Intermediate Language?

LLMs are good at understanding intent but inconsistent at producing correct code across different languages. INTL separates the two concerns:

1. **Understanding** -- A frontier model (Claude) converts requirements into a structured, unambiguous intermediate representation
2. **Compilation** -- Small, fine-tuned models (3B parameters) handle the mechanical translation to each target language

This gives you frontier-quality understanding with fast, cheap, deterministic compilation.

## Supported Targets (24)

Python (FastAPI, Django), TypeScript (Express, Next.js), C# (.NET), Java (Spring), Go (Gin), Rust (Actix), Ruby (Rails), PHP (Laravel), Swift, Kotlin, and more.

## Pipeline Components

| Component | File | Tests | Status |
|-----------|------|-------|--------|
| Lark Grammar | `intl/grammar.lark` | -- | Done |
| Parser | `intl/parser.py` | `tests/test_parser.py` | Done |
| Semantic Index | `intl/index.py` | `tests/test_index.py` | Done |
| LoRA Router | `intl/router.py` | `tests/test_router.py` | Done |
| Validator T1-T7 | `intl/validator.py` | `tests/test_validator.py` | Done |
| Training Data Gen | `intl/datagen.py` | -- | Done |
| Compiler Engine | `intl/compiler.py` | -- | Done |
| CLI | `intl/cli.py` | -- | Done |
| Generator | `intl/generator.py` | `tests/test_generator.py` | Done |
| Escalation | `intl/escalation.py` | `tests/test_escalation.py` | Done |

## Validation Pipeline

| Level | Check |
|-------|-------|
| T1 | Syntax validity |
| T2 | Type consistency |
| T3 | Import resolution |
| T4 | API contract compliance |
| T5 | Security patterns |
| T6 | Integration tests |
| T7 | Frontier model review |

**193/193 tests passing.**

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

## Tech Stack

Python, Claude API, Qwen2.5-Coder-3B, LoRA, Lark (parser), SQLite, HuggingFace

## License

MIT
