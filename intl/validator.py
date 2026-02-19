"""INTL Validator — T1–T7 test pipeline (§6.1).

Runs after every compilation. Returns ValidationResult with pass/fail details
and an escalation package when checks fail.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str          # e.g. "T1-Syntax"
    passed: bool
    message: str = ""
    detail: str = ""


@dataclass
class ValidationResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    failed_checks: list[CheckResult] = field(default_factory=list)
    escalation_package: Optional[dict] = None

    @classmethod
    def _build(
        cls,
        checks: list[CheckResult],
        intl_block: str,
        compiled_code: str,
        block_id: str,
    ) -> "ValidationResult":
        failed = [c for c in checks if not c.passed]
        passed_all = len(failed) == 0
        esc = None
        if not passed_all:
            esc = {
                "block_id": block_id,
                "intl_block": intl_block,
                "compiled_code": compiled_code,
                "failed_checks": [
                    {"name": c.name, "message": c.message, "detail": c.detail}
                    for c in failed
                ],
            }
        return cls(passed=passed_all, checks=checks, failed_checks=failed,
                   escalation_package=esc)


# ── Individual checks ─────────────────────────────────────────────────────────

# T1 — Syntax: attempt a tree-sitter or ast.parse for Python; regex for others
def _check_t1_syntax(code: str, profile: str) -> CheckResult:
    """T1: Zero parse errors in compiled output."""
    name = "T1-Syntax"
    try:
        if profile.startswith("python"):
            import ast as py_ast
            py_ast.parse(code)
            return CheckResult(name, True, "Python AST parse OK")
        elif profile.startswith("typescript") or profile.startswith("javascript"):
            # Lightweight heuristic: balanced braces/brackets
            if _balanced(code):
                return CheckResult(name, True, "Brace balance OK")
            return CheckResult(name, False, "Unbalanced braces/brackets",
                               detail="Brace balance check failed")
        elif profile.startswith("sql"):
            # Very lightweight: must not be empty, must contain SELECT/INSERT/UPDATE/DELETE/CREATE
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|BEGIN|COMMIT)\b',
                         code, re.I):
                return CheckResult(name, True, "SQL keyword present")
            return CheckResult(name, False, "No SQL keyword found in output")
        elif profile.startswith("html"):
            # Must have at least one HTML tag
            if re.search(r'<[a-zA-Z][^>]*>', code):
                return CheckResult(name, True, "HTML tag present")
            return CheckResult(name, False, "No HTML tag found in output")
        elif profile.startswith("css"):
            if re.search(r'[\.\#]?[\w-]+\s*\{', code):
                return CheckResult(name, True, "CSS rule present")
            return CheckResult(name, False, "No CSS rule found in output")
        else:
            # Fallback: balanced braces
            if _balanced(code):
                return CheckResult(name, True, "Brace balance OK (fallback)")
            return CheckResult(name, False, "Unbalanced braces/brackets")
    except SyntaxError as e:
        return CheckResult(name, False, f"Syntax error: {e}", detail=str(e))
    except Exception as e:
        return CheckResult(name, False, f"Parse check error: {e}")


def _balanced(code: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in code:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


# T2 — Sentinel: INTL:BEGIN and INTL:END with correct ID
def _check_t2_sentinel(code: str, block_id: str) -> CheckResult:
    name = "T2-Sentinel"
    begin_pat = re.compile(
        r'INTL:BEGIN\s+\[id=' + re.escape(block_id) + r'\]', re.IGNORECASE
    )
    end_pat = re.compile(
        r'INTL:END\s+\[id=' + re.escape(block_id) + r'\]', re.IGNORECASE
    )
    has_begin = bool(begin_pat.search(code))
    has_end = bool(end_pat.search(code))
    if has_begin and has_end:
        return CheckResult(name, True, "BEGIN/END sentinels present")
    missing = []
    if not has_begin:
        missing.append(f"INTL:BEGIN [id={block_id}]")
    if not has_end:
        missing.append(f"INTL:END [id={block_id}]")
    return CheckResult(name, False,
                       f"Missing sentinels: {', '.join(missing)}",
                       detail=f"Expected sentinels for id={block_id}")


# T3 — Preconditions: each PRECONDITION must map to a guard in output
def _check_t3_preconditions(
    code: str, preconditions: list[str], profile: str
) -> CheckResult:
    name = "T3-Preconditions"
    if not preconditions:
        return CheckResult(name, True, "No preconditions to check")
    missing = []
    for pre in preconditions:
        # Extract key identifiers from precondition expression
        tokens = set(re.findall(r'[a-zA-Z_]\w*', pre))
        # At least one token from the precondition should appear in a guard context
        found = False
        for tok in tokens:
            if tok in ('IS', 'NOT', 'NULL', 'AND', 'OR', 'THEN', 'IF', 'True', 'False'):
                continue
            if re.search(
                r'\b' + re.escape(tok) + r'\b',
                code
            ):
                found = True
                break
        if not found:
            missing.append(pre)
    if not missing:
        return CheckResult(name, True, f"All {len(preconditions)} precondition(s) mapped")
    return CheckResult(
        name, False,
        f"{len(missing)} precondition(s) not mapped in output",
        detail=f"Missing guards for: {'; '.join(missing)}"
    )


# T4 — Postconditions: return/assertion constructs present
def _check_t4_postconditions(
    code: str, postconditions: list[str], profile: str
) -> CheckResult:
    name = "T4-Postconditions"
    if not postconditions:
        return CheckResult(name, True, "No postconditions to check")
    # For postconditions we check that the output has some return/assertion indicator
    has_return = bool(re.search(r'\breturn\b|\bRETURN\b|\bResult\b|\bresponse\b', code))
    if not has_return:
        return CheckResult(name, False, "No return/result expression found in output",
                           detail=f"Postconditions: {'; '.join(postconditions)}")
    # Check key identifiers from each postcondition appear
    missing = []
    for post in postconditions:
        tokens = [t for t in re.findall(r'[a-zA-Z_]\w*', post)
                  if t not in ('IS', 'NOT', 'NULL', 'NOW', 'AND', 'OR', 'True', 'False')]
        found = any(re.search(r'\b' + re.escape(t) + r'\b', code) for t in tokens)
        if not found and tokens:
            missing.append(post)
    if not missing:
        return CheckResult(name, True, f"All {len(postconditions)} postcondition(s) mapped")
    return CheckResult(
        name, False,
        f"{len(missing)} postcondition(s) not verified in output",
        detail=f"Unverified: {'; '.join(missing)}"
    )


# T5 — Side effects: MUTATES → write op present; OBSERVABLE → log call present
def _check_t5_side_effects(
    code: str,
    mutates: list[str],
    observable: bool,
    emits: list[str],
    profile: str,
) -> CheckResult:
    name = "T5-SideEffects"
    issues = []

    if mutates:
        write_patterns = [
            r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b',
            r'\.save\b', r'\.create\b', r'\.update\b', r'\.delete\b',
            r'\.add\b', r'\.put\b', r'\.post\b',
            r'session\.commit\b', r'db\.commit\b', r'\.execute\b',
        ]
        has_write = any(re.search(p, code, re.I) for p in write_patterns)
        if not has_write:
            issues.append(f"MUTATES {mutates} declared but no write operation found")

    if observable:
        log_patterns = [
            r'\blog\b', r'\blogger\b', r'logging\.', r'\.info\b', r'\.debug\b',
            r'\.warning\b', r'console\.log', r'Log\.', r'@observable', r'@log',
        ]
        has_log = any(re.search(p, code, re.I) for p in log_patterns)
        if not has_log:
            issues.append("OBSERVABLE declared but no logging call found")

    if emits:
        emit_patterns = [r'\bemit\b', r'\bpublish\b', r'\bdispatch\b', r'\bsend\b']
        has_emit = any(re.search(p, code, re.I) for p in emit_patterns)
        if not has_emit:
            issues.append(f"EMITS {emits} declared but no event emit found")

    if not issues:
        return CheckResult(name, True, "All side effects accounted for")
    return CheckResult(name, False, "; ".join(issues))


# T6 — Types: return type matches TYPE declaration
def _check_t6_types(
    code: str, return_type: str | None, profile: str
) -> CheckResult:
    name = "T6-Types"
    if not return_type:
        return CheckResult(name, True, "No TYPE declaration to check")
    if re.search(r'\b' + re.escape(return_type) + r'\b', code):
        return CheckResult(name, True, f"Return type {return_type!r} found in output")
    # Softer check for common aliases
    return CheckResult(
        name, False,
        f"Return type {return_type!r} not found in compiled output",
        detail=f"Expected to find {return_type!r} in output"
    )


# T7 — No placeholders: TODO, FIXME, pass (Python), unimplemented!()
def _check_t7_no_placeholders(code: str) -> CheckResult:
    name = "T7-NoPlaceholders"
    patterns = [
        (r'\bTODO\b', 'TODO'),
        (r'\bFIXME\b', 'FIXME'),
        (r'^\s*pass\s*$', 'bare pass'),
        (r'\bunimplemented!\(\)', 'unimplemented!()'),
        (r'\bNotImplementedError\b', 'NotImplementedError'),
        (r'\bthrow new Error\("not implemented"\)', 'throw not implemented'),
        (r'\bplaceholder\b', 'placeholder'),
    ]
    found = []
    for pat, label in patterns:
        if re.search(pat, code, re.I | re.M):
            found.append(label)
    if not found:
        return CheckResult(name, True, "No placeholders found")
    return CheckResult(
        name, False,
        f"Placeholder(s) found: {', '.join(found)}",
        detail="Output contains incomplete code markers"
    )


# ── Validator public API ──────────────────────────────────────────────────────
class Validator:
    """Run T1–T7 checks on compiled output."""

    def validate(
        self,
        compiled_code: str,
        block_id: str,
        profile: str,
        intl_block: str = "",
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        mutates: list[str] | None = None,
        emits: list[str] | None = None,
        observable: bool = False,
        return_type: str | None = None,
    ) -> ValidationResult:
        """Run all T1–T7 checks.

        Args:
            compiled_code: The generated target-language code.
            block_id: INTL block ID (e.g. ``"f001"``).
            profile: INTL PROFILE string (e.g. ``"python_fastapi"``).
            intl_block: Original INTL source (for escalation package).
            preconditions: List of PRECONDITION expression strings.
            postconditions: List of POSTCONDITION expression strings.
            mutates: List of MUTATES table/collection names.
            emits: List of EMITS event names.
            observable: Whether OBSERVABLE flag is set.
            return_type: Optional TYPE name for T6 check.

        Returns:
            :class:`ValidationResult`
        """
        pre = preconditions or []
        post = postconditions or []
        mut = mutates or []
        emi = emits or []

        checks = [
            _check_t1_syntax(compiled_code, profile),
            _check_t2_sentinel(compiled_code, block_id),
            _check_t3_preconditions(compiled_code, pre, profile),
            _check_t4_postconditions(compiled_code, post, profile),
            _check_t5_side_effects(compiled_code, mut, observable, emi, profile),
            _check_t6_types(compiled_code, return_type, profile),
            _check_t7_no_placeholders(compiled_code),
        ]

        return ValidationResult._build(checks, intl_block, compiled_code, block_id)

    def summary(self, result: ValidationResult) -> str:
        """Human-readable summary string."""
        lines = []
        for c in result.checks:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  {icon} {c.name}: {c.message}")
            if not c.passed and c.detail:
                lines.append(f"      → {c.detail}")
        status = "PASSED" if result.passed else "FAILED"
        lines.insert(0, f"Validation {status} ({sum(c.passed for c in result.checks)}/{len(result.checks)} checks)")
        return "\n".join(lines)


# Module-level convenience
_default_validator = Validator()

def validate(
    compiled_code: str,
    block_id: str,
    profile: str,
    **kwargs,
) -> ValidationResult:
    """Convenience wrapper around :class:`Validator`.validate()."""
    return _default_validator.validate(compiled_code, block_id, profile, **kwargs)
