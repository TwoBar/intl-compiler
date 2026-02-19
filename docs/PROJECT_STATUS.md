# INTL Compiler — Project Status

_Last updated: 2026-02-19_

---

## Overall Progress

```
Phase          Components  Tests   Training Data   Adapters Trained
─────────────────────────────────────────────────────────────────────
Core pipeline  10 / 10     193 ✅   0 / 24          0 / 24
Phase 0        —           —        0 / 2           0 / 2
Phase 1        —           —        0 / 4           0 / 4
Phase 2        —           —        0 / 7           0 / 7
Phase 3        —           —        0 / 6           0 / 6
Phase 4        —           —        0 / 3           0 / 3
Phase 5        —           —        0 / 2           0 / 2
```

---

## Component Status

| # | Component | File | Tests | Closed Issue | Status |
|---|-----------|------|-------|--------------|--------|
| 1 | Lark Grammar | `intl/grammar.lark` | — | #1 | ✅ Done |
| 2 | Parser | `intl/parser.py` | `test_parser.py` | #2 | ✅ Done |
| 3 | Semantic Index | `intl/index.py` | `test_index.py` | #3 | ✅ Done |
| 4 | LoRA Router | `intl/router.py` | `test_router.py` | #4 | ✅ Done |
| 5 | Validator T1–T7 | `intl/validator.py` | `test_validator.py` | #5 | ✅ Done |
| 6 | Training Data Gen | `intl/datagen.py` | — | #6 | ✅ Done |
| 7 | Compiler Engine | `intl/compiler.py` | — | #7 | ✅ Done |
| 8 | CLI | `intl/cli.py` | — | #8 | ✅ Done |
| 9 | Parser Tests | `tests/test_parser.py` | — | #9 | ✅ Done |
| 10 | Index Tests | `tests/test_index.py` | — | #10 | ✅ Done |
| 11 | Router Tests | `tests/test_router.py` | — | #11 | ✅ Done |
| 12 | Validator Tests | `tests/test_validator.py` | — | #12 | ✅ Done |
| — | Generator | `intl/generator.py` | `test_generator.py` | f96f830 | ✅ Done |
| — | Escalation | `intl/escalation.py` | `test_escalation.py` | f96f830 | ✅ Done |

**Test suite: 193 / 193 passing ✅**

---

## Training Data Status

| Adapter | Phase | Data Issue | Pairs Generated | Training Issue | Trained |
|---------|-------|-----------|-----------------|---------------|---------|
| python_fastapi | 0 | #13 | 0 (corrections.jsonl: 16 rows) | #37 | ❌ |
| sql_postgres | 0 | #14 | 0 | #38 | ❌ |
| python_django | 1 | #15 | 0 | #39 | ❌ |
| python_flask | 1 | #16 | 0 | #40 | ❌ |
| typescript_express | 1 | #17 | 0 | #41 | ❌ |
| sql_mysql | 1 | #18 | 0 | #42 | ❌ |
| typescript_nextjs | 2 | #19 | 0 | #43 | ❌ |
| php_laravel | 2 | #20 | 0 | #44 | ❌ |
| php_vanilla | 2 | #21 | 0 | #45 | ❌ |
| javascript_vanilla | 2 | #22 | 0 | #46 | ❌ |
| html_jinja2 | 2 | #23 | 0 | #47 | ❌ |
| html_blade | 2 | #24 | 0 | #48 | ❌ |
| css_tailwind | 2 | #25 | 0 | #49 | ❌ |
| java_spring | 3 | #26 | 0 | #50 | ❌ |
| csharp_dotnet | 3 | #27 | 0 | #51 | ❌ |
| go_gin | 3 | #28 | 0 | #52 | ❌ |
| ruby_rails | 3 | #29 | 0 | #53 | ❌ |
| sql_tsql | 3 | #30 | 0 | #54 | ❌ |
| sql_sqlite | 3 | #31 | 0 | #55 | ❌ |
| swift_ios | 4 | #32 | 0 | #56 | ❌ |
| kotlin_android | 4 | #33 | 0 | #57 | ❌ |
| dart_flutter | 4 | #34 | 0 | #58 | ❌ |
| rust_axum | 5 | #35 | 0 | #59 | ❌ |
| cpp_modern | 5 | #36 | 0 | #60 | ❌ |

---

## Next Actions

1. **Generate training pairs** — start with Phase 0: `python_fastapi`, `sql_postgres`
   ```bash
   python3 -m intl.datagen --adapter python_fastapi --count 3000 --val 200
   python3 -m intl.datagen --adapter sql_postgres --count 3000 --val 200
   ```

2. **Train Phase 0 adapters** — on Vast.ai RTX 4090 (~$0.12 each)
   ```bash
   bash scripts/train_adapter.sh python_fastapi
   bash scripts/train_adapter.sh sql_postgres
   ```

3. **Push adapters to HuggingFace** — `$HF_USERNAME/intl-adapters`

4. **Repeat for Phase 1–5**

---

## Git Log

```
f96f830  feat: build generator.py and escalation.py — 193 tests pass
4762d81  fix: grammar TYPE fields + PATCH precondition/position syntax — 165 tests pass
8e27406  feat: build Lark grammar and parser with typed AST (issues #1, #2)
33d8abb  init: INTL spec, adapter configs, training script, repo structure
```
