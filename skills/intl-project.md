# INTL Project — Agent Skills

## What is INTL?

INTL (Intent Language) is a structured intermediate representation. Frontier AI models write INTL.
Small fine-tuned LoRA adapters (Qwen2.5-Coder-3B) compile INTL into production code in 24 target languages.

Full spec: `/workspace/docs/INTL_Specification.md` — always read this before building any component.

**Compilation pipeline:**
```
INTL source → Parser (Lark) → Semantic Index (SQLite) → LoRA Router → Compiler Engine → Validator
                                                                                              ↓ fail
                                                                               Frontier escalation (Claude)
```

---

## Source of Truth: GitHub

The GitHub repo `$GITHUB_USERNAME/intl-compiler` is the source of truth for all code, specs, and backlog.

**Every action follows this pattern:**
1. Build / generate / train
2. Commit with a clear message
3. Push to GitHub
4. Close or comment on the relevant GitHub Issue
5. Report back with URL

Never leave uncommitted work. Every file written goes into a commit.

---

## Repo Layout

```
intl-compiler/
├── README.md
├── SPEC.md                        ← copy of INTL_Specification.md
├── intl/
│   ├── grammar.lark
│   ├── parser.py
│   ├── index.py
│   ├── router.py
│   ├── compiler.py
│   ├── validator.py
│   ├── datagen.py
│   ├── cli.py
│   └── tests/
│       ├── test_parser.py
│       ├── test_index.py
│       ├── test_router.py
│       └── test_validator.py
├── configs/
│   └── adapters.json
├── docs/
│   └── INTL_Specification.md
└── scripts/
    └── train_adapter.sh
```

The workspace at `/workspace` is the local clone of this repo.

---

## Git Workflow

```bash
# One-time setup
cd /workspace
git config user.name "INTL Agent"
git config user.email "agent@intl-lang.dev"
git remote set-url origin https://$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/intl-compiler.git

# After writing any file
cd /workspace
git add -A
git commit -m "feat: build INTL parser"
git push origin main

# Status
git status
git log --oneline -10
```

**Commit conventions:**
- `feat:` — new component
- `fix:` — bug fix
- `data: <adapter> training pairs generated`
- `model: <adapter> adapter trained`
- `docs:` — readme / spec updates

---

## GitHub Issues — Backlog

```bash
# Create issue
gh issue create --repo $GITHUB_USERNAME/intl-compiler \
  --title "Build INTL parser" --body "..." --label "component"

# List open issues
gh issue list --repo $GITHUB_USERNAME/intl-compiler

# Close issue
gh issue close <N> --repo $GITHUB_USERNAME/intl-compiler \
  --comment "Done in commit abc1234."

# Add milestone
gh api repos/$GITHUB_USERNAME/intl-compiler/milestones \
  -X POST -f title="Phase 0 Training" -f description="python_fastapi + sql_postgres"
```

Labels: `component`, `training`, `data`, `bug`, `spec`

---

## Initialising the Repo (run once)

```bash
gh repo create $GITHUB_USERNAME/intl-compiler \
  --public \
  --description "INTL compiler — 24 LoRA adapters for Qwen2.5-Coder-3B"

cd /workspace
git init
git remote add origin https://$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/intl-compiler.git
git config user.name "INTL Agent"
git config user.email "agent@intl-lang.dev"
git add docs/ configs/ scripts/
git commit -m "init: INTL spec, adapter configs, training script"
git branch -M main
git push -u origin main
```

---

## Full Backlog (create all issues on first run)

**Component issues (label: component)**
1. Build Lark grammar (grammar.lark)
2. Build parser (parser.py)
3. Build semantic index (index.py)
4. Build LoRA router (router.py)
5. Build validator T1–T7 (validator.py)
6. Build training data generator (datagen.py)
7. Build compiler engine (compiler.py)
8. Build CLI (cli.py)
9. Write parser tests
10. Write index tests
11. Write router tests
12. Write validator tests

**Data issues (label: data) — one per adapter:** "Generate training pairs: <adapter>"

**Training issues (label: training) — one per adapter:** "Train adapter: <adapter>" — grouped by phase milestone

---

## Building Components

When asked to build any component:
1. `read_file /workspace/docs/INTL_Specification.md` — read relevant sections
2. `write_file /workspace/intl/<component>.py` — write the code
3. `bash: cd /workspace && python3 -c "from intl import <module>"` — verify imports
4. `bash: cd /workspace && python3 -m pytest intl/tests/ -v` — run tests
5. `bash: git add -A && git commit -m "feat: ..." && git push origin main`
6. Close GitHub issue with commit SHA

### Parser → grammar.lark + parser.py
Lark EBNF, 24 constructs from §2.3. Typed AST dataclasses. Line/col error messages.

### Semantic Index → index.py
SQLite schema from §3.3. Dirty propagation rules. Methods: index_module(), mark_dirty(), get_dirty_nodes(), record_compiled().

### LoRA Router → router.py
Pure O(1) ADAPTER_MAP lookup from §3.4. No inference. Raises clear error for unknown profiles.

### Validator → validator.py
T1–T7 from §6.1. Returns ValidationResult(passed, failed_checks, escalation_package).

### Training Data Generator → datagen.py
Uses ANTHROPIC_API_KEY. 60/30/10 A/B/C split. JSONL to /workspace/data/<adapter>/. 200-pair validation set.

### Compiler Engine → compiler.py
Qwen2.5-Coder-3B + LoRA via transformers/peft. Temperature 0.1. Wraps output in sentinels.

### CLI → cli.py
`intl compile`, `intl build`, `intl status`, `intl adapters`, `intl validate`

---

## Training Workflow

```bash
# Always: check balance → show cost → get confirmation → train → push → close issue
vastai show user --raw | python3 -c "import json,sys; u=json.load(sys.stdin); print(f'Balance: \${u[\"credit\"]:.2f}')"
bash /workspace/scripts/train_adapter.sh <adapter_name>
gh issue close <N> --repo $GITHUB_USERNAME/intl-compiler \
  --comment "Trained. HF: https://huggingface.co/$HF_USERNAME/intl-adapters/tree/main/<adapter>"
```

---

## HuggingFace

```bash
# Push training data
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(folder_path='/workspace/data/<adapter>',
  repo_id='$HF_USERNAME/intl-training-pairs', path_in_repo='<adapter>', repo_type='dataset')
print('pushed')
"
```

Repos: `$HF_USERNAME/intl-adapters` (model), `$HF_USERNAME/intl-training-pairs` (dataset)

---

## 24 Adapters

```
Phase 0: python_fastapi, sql_postgres
Phase 1: python_django, python_flask, typescript_express, sql_mysql
Phase 2: typescript_nextjs, php_laravel, php_vanilla, javascript_vanilla, html_jinja2, html_blade, css_tailwind
Phase 3: java_spring, csharp_dotnet, go_gin, ruby_rails, sql_tsql, sql_sqlite
Phase 4: swift_ios, kotlin_android, dart_flutter
Phase 5: rust_axum, cpp_modern
```

---

## Training Data Format (§5.1)

```jsonl
{"system": "You are the INTL compiler for Python FastAPI...", "prompt": "FUNCTION login [id=f001]\n  INTENT ...", "completion": "# ═══ INTL:BEGIN [id=f001] login ═══\nasync def login(...):..."}
```
Split: 60% A (fresh), 30% B (PATCH), 10% C (error correction). 200 validation pairs per adapter.

---

## Key Rules

- Read the spec before building any component
- Commit + push after every completed action
- Close the GitHub Issue when done (with commit SHA or HF URL)
- Never start training without showing cost and getting confirmation
- Always destroy Vast.ai instances after training
- Never leave /workspace in a dirty git state after a completed task
