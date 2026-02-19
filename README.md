# INTL Compiler

**INTL (Intent Language)** is a structured intermediate representation designed to be written by frontier AI models and compiled into idiomatic, production-ready code in 24 target languages.

## Architecture

```
INTL source → Parser (Lark) → Semantic Index (SQLite) → LoRA Router → Compiler Engine → Validator
                                                                                        ↓ fail
                                                                         Frontier escalation (Claude)
```

## How It Works

- **Frontier AI models** (Claude, GPT-4, etc.) write INTL — a precise, target-agnostic specification of *what* code should do.
- **Small fine-tuned LoRA adapters** (Qwen2.5-Coder-3B) compile INTL into production code for each target language/framework.
- **Validation pipeline** (T1–T7) checks every output. Failures escalate to frontier models automatically.

## 24 Target Adapters

| Phase | Adapters |
|-------|----------|
| 0 | `python_fastapi`, `sql_postgres` |
| 1 | `python_django`, `python_flask`, `typescript_express`, `sql_mysql` |
| 2 | `typescript_nextjs`, `php_laravel`, `php_vanilla`, `javascript_vanilla`, `html_jinja2`, `html_blade`, `css_tailwind` |
| 3 | `java_spring`, `csharp_dotnet`, `go_gin`, `ruby_rails`, `sql_tsql`, `sql_sqlite` |
| 4 | `swift_ios`, `kotlin_android`, `dart_flutter` |
| 5 | `rust_axum`, `cpp_modern` |

## Quick Start

```bash
pip install -e .
intl compile mymodule.intl --profile python_fastapi
intl build project/
intl status
intl adapters
```

## Project Structure

```
intl-compiler/
├── intl/
│   ├── grammar.lark       # Lark EBNF grammar
│   ├── parser.py          # AST parser
│   ├── index.py           # Semantic index (SQLite)
│   ├── router.py          # LoRA adapter router
│   ├── compiler.py        # Compiler engine (Qwen2.5-Coder-3B + LoRA)
│   ├── validator.py       # T1–T7 validation pipeline
│   ├── datagen.py         # Training data generator
│   ├── cli.py             # CLI interface
│   └── tests/             # Test suite
├── configs/
│   └── adapters.json      # Adapter registry
├── docs/
│   └── INTL_Specification.md
├── scripts/
│   └── train_adapter.sh   # Vast.ai training script
└── data/                  # Training data (per adapter)
```

## Training Cost

Total estimated training cost: **~$2.85** on Vast.ai RTX 4090 instances.

## License

Proprietary — Confidential.
