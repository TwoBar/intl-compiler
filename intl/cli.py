"""INTL Command-Line Interface.

Commands:
  intl compile   — Compile an .intl file to target language
  intl build     — Parse + index + compile all dirty blocks
  intl status    — Show index stats and dirty blocks
  intl adapters  — List available adapters and phases
  intl validate  — Validate compiled output against T1-T7 checks
  intl datagen   — Generate training data for an adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ok(msg: str):
    print(f"  ✓ {msg}")

def _err(msg: str):
    print(f"  ✗ {msg}", file=sys.stderr)

def _info(msg: str):
    print(f"  · {msg}")


# ── compile ───────────────────────────────────────────────────────────────────
def cmd_compile(args):
    """Parse an .intl file, route to adapter, compile, validate, and write output."""
    from intl.parser import parse_safe
    from intl.router import route, RouterError
    from intl.validator import validate

    src = Path(args.file)
    if not src.exists():
        _err(f"File not found: {src}")
        sys.exit(1)

    source = src.read_text()
    module, err = parse_safe(source)
    if err:
        _err(f"Parse error: {err}")
        sys.exit(1)
    _ok(f"Parsed {len(module.blocks)} block(s) from {src.name}")

    try:
        r = route(module.profile)
    except RouterError as e:
        _err(str(e))
        sys.exit(1)
    _ok(f"Routed to adapter: {r.profile} (phase {r.phase})")

    if args.dry_run:
        _info("Dry-run mode — skipping model inference")
        return

    try:
        from intl.compiler import Compiler
        compiler = Compiler()
        results = compiler.compile_module(module, escalate=not args.no_escalate)
    except Exception as e:
        _err(f"Compilation error: {e}")
        sys.exit(1)

    for res in results:
        status = "escalated" if res.escalated else ("ok" if res.validation_passed else "failed")
        _ok(f"[{res.block_id}] {res.name} — {status}")
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a") as f:
                f.write(res.code + "\n\n")
    if args.output:
        _ok(f"Output written → {args.output}")


# ── build ─────────────────────────────────────────────────────────────────────
def cmd_build(args):
    """Parse + index + compile all dirty blocks in an .intl file."""
    from intl.parser import parse_safe
    from intl.index import SemanticIndex

    src = Path(args.file)
    if not src.exists():
        _err(f"File not found: {src}")
        sys.exit(1)

    source = src.read_text()
    module, err = parse_safe(source)
    if err:
        _err(f"Parse error: {err}")
        sys.exit(1)
    _ok(f"Parsed: {module.name} [{module.id}]")

    db_path = Path(args.db) if args.db else Path(".intl-index.db")
    idx = SemanticIndex(db_path)
    block_ids = idx.index_module(module)
    _ok(f"Indexed {len(block_ids)} block(s) → {db_path}")

    dirty = idx.get_dirty_nodes()
    _info(f"{len(dirty)} dirty block(s) queued for compilation")

    if args.dry_run or not dirty:
        idx.close()
        return

    try:
        from intl.compiler import Compiler
        compiler = Compiler()
        for node in dirty:
            _info(f"Compiling [{node.block_id}] {node.name} ({node.profile}) …")
            try:
                block = next((b for b in module.blocks if getattr(b, "id") == node.block_id), None)
                result = compiler.compile(
                    intl_block=f"# block {node.name} [{node.block_id}]",
                    block_id=node.block_id,
                    name=node.name,
                    profile=node.profile,
                    preconditions=getattr(block, "preconditions", None) if block else None,
                    postconditions=getattr(block, "postconditions", None) if block else None,
                    mutates=getattr(block, "mutates", None) if block else None,
                    emits=getattr(block, "emits", None) if block else None,
                    observable=getattr(block, "observable", False) if block else False,
                )
                idx.record_compiled(node.block_id, result.code)
                status = "escalated" if result.escalated else "ok"
                _ok(f"[{node.block_id}] {node.name} — {status}")
            except Exception as e:
                _err(f"[{node.block_id}] {node.name} — {e}")
    finally:
        idx.close()


# ── status ────────────────────────────────────────────────────────────────────
def cmd_status(args):
    """Show semantic index statistics and dirty blocks."""
    from intl.index import SemanticIndex

    db_path = Path(args.db) if args.db else Path(".intl-index.db")
    if not db_path.exists():
        _err(f"No index found at {db_path}. Run `intl build` first.")
        sys.exit(1)

    idx = SemanticIndex(db_path)
    stats = idx.stats()
    print(f"\nINTL Semantic Index — {db_path}")
    print(f"  Modules  : {stats['modules']}")
    print(f"  Blocks   : {stats['blocks']}")
    print(f"  Dirty    : {stats['dirty_blocks']}")

    modules = idx.get_all_modules()
    if modules:
        print("\n  Modules:")
        for m in modules:
            dirty_icon = "⚡" if m["dirty"] else "✓"
            print(f"    {dirty_icon} {m['id']} — {m['name']} [{m['profile']}] → {m['target']}")

    dirty = idx.get_dirty_nodes()
    if dirty:
        print(f"\n  Dirty blocks ({len(dirty)}):")
        for d in dirty:
            print(f"    · [{d.block_id}] {d.name} ({d.kind}) in {d.module_id}")

    idx.close()


# ── adapters ──────────────────────────────────────────────────────────────────
def cmd_adapters(args):
    """List all adapters grouped by phase."""
    from intl.router import list_profiles, ADAPTER_MAP

    phases: dict[int, list[str]] = {}
    for profile in list_profiles():
        ph = ADAPTER_MAP[profile]["phase"]
        phases.setdefault(ph, []).append(profile)

    print("\nINTL Adapters")
    for phase in sorted(phases.keys()):
        print(f"\n  Phase {phase}:")
        for profile in sorted(phases[phase]):
            meta = ADAPTER_MAP[profile]
            threshold_pct = int(meta["threshold"] * 100)
            print(f"    · {profile:<30} {meta['pairs']:>5} pairs  threshold={threshold_pct}%")
    print()


# ── validate ──────────────────────────────────────────────────────────────────
def cmd_validate(args):
    """Validate compiled code against T1–T7 checks."""
    from intl.validator import Validator

    code_path = Path(args.file)
    if not code_path.exists():
        _err(f"File not found: {code_path}")
        sys.exit(1)

    code = code_path.read_text()
    v = Validator()
    result = v.validate(
        compiled_code=code,
        block_id=args.block_id,
        profile=args.profile,
        intl_block=args.intl_block or "",
    )
    print()
    print(v.summary(result))
    print()
    if not result.passed and result.escalation_package:
        _info("Escalation package prepared (use --json to inspect)")
        if args.json:
            print(json.dumps(result.escalation_package, indent=2))
    sys.exit(0 if result.passed else 1)


# ── datagen ───────────────────────────────────────────────────────────────────
def cmd_datagen(args):
    """Generate training data for a specified adapter."""
    from intl.datagen import generate, push_to_hub

    _info(f"Generating {args.count} train + {args.val} validation pairs for '{args.adapter}' …")
    train_path, val_path = generate(
        adapter=args.adapter,
        count=args.count,
        val_count=args.val,
        dry_run=args.dry_run,
    )
    _ok(f"Train → {train_path}")
    _ok(f"Val   → {val_path}")

    if args.push and not args.dry_run:
        url = push_to_hub(args.adapter)
        _ok(f"HF    → {url}")


# ── Main parser ───────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="intl",
        description="INTL Compiler — Intent Language → production code",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compile
    p_compile = sub.add_parser("compile", help="Compile an .intl file")
    p_compile.add_argument("file", help="Path to .intl source file")
    p_compile.add_argument("-o", "--output", help="Output file path")
    p_compile.add_argument("--dry-run", action="store_true", help="Parse+route only, skip model")
    p_compile.add_argument("--no-escalate", action="store_true",
                           help="Raise error instead of escalating to frontier")
    p_compile.set_defaults(func=cmd_compile)

    # build
    p_build = sub.add_parser("build", help="Parse + index + compile dirty blocks")
    p_build.add_argument("file", help="Path to .intl source file")
    p_build.add_argument("--db", help="Index DB path (default: .intl-index.db)")
    p_build.add_argument("--dry-run", action="store_true", help="Index only, skip compilation")
    p_build.set_defaults(func=cmd_build)

    # status
    p_status = sub.add_parser("status", help="Show index statistics")
    p_status.add_argument("--db", help="Index DB path")
    p_status.set_defaults(func=cmd_status)

    # adapters
    p_adapters = sub.add_parser("adapters", help="List available adapters")
    p_adapters.set_defaults(func=cmd_adapters)

    # validate
    p_val = sub.add_parser("validate", help="Run T1–T7 validation checks")
    p_val.add_argument("file", help="Path to compiled output file")
    p_val.add_argument("--block-id", required=True, help="Block ID (e.g. f001)")
    p_val.add_argument("--profile", required=True, help="INTL profile (e.g. python_fastapi)")
    p_val.add_argument("--intl-block", default="", help="Original INTL source (for escalation)")
    p_val.add_argument("--json", action="store_true", help="Print escalation package as JSON")
    p_val.set_defaults(func=cmd_validate)

    # datagen
    p_dg = sub.add_parser("datagen", help="Generate training pairs for an adapter")
    p_dg.add_argument("--adapter", required=True)
    p_dg.add_argument("--count", type=int, default=1000)
    p_dg.add_argument("--val", type=int, default=200)
    p_dg.add_argument("--push", action="store_true", help="Push to HuggingFace after generation")
    p_dg.add_argument("--dry-run", action="store_true")
    p_dg.set_defaults(func=cmd_datagen)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
