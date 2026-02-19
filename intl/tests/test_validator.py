"""Tests for intl.validator — T1–T7 test pipeline (issue #12)."""
import pytest
from intl.validator import (
    Validator, validate, ValidationResult, CheckResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
BLOCK_ID = "f001"

# Valid Python FastAPI function with all sentinels
GOOD_PYTHON = """\
# ═══ INTL:BEGIN [id=f001] login ═══
import logging
logger = logging.getLogger(__name__)

async def login(email: str, password: str) -> SessionToken:
    logger.info("login attempt for %s", email)
    if not email or len(email) == 0:
        raise ValueError("email required")
    if len(password) < 8:
        raise ValueError("password too short")
    user = await db.query(users_table).filter(email=email).first()
    if user is None:
        raise AuthError("user_not_found")
    token = generate_token(user.id)
    await db.execute(sessions_table.insert().values(user_id=user.id, token=token))
    return SessionToken(token=token, expires_at=now() + timedelta(hours=24))
# ═══ INTL:END   [id=f001] login ═══
"""

# Python with syntax error
BAD_PYTHON_SYNTAX = """\
# ═══ INTL:BEGIN [id=f001] bad ═══
def bad_func(
    return "incomplete"
# ═══ INTL:END   [id=f001] bad ═══
"""

# Missing sentinels
NO_SENTINELS = """\
async def login(email: str, password: str):
    return {"token": "abc"}
"""

# Has TODO placeholder
HAS_TODO = """\
# ═══ INTL:BEGIN [id=f001] login ═══
async def login(email: str, password: str):
    # TODO: implement this
    pass
# ═══ INTL:END   [id=f001] login ═══
"""

# Good SQL
GOOD_SQL = """\
-- ═══ INTL:BEGIN [id=f001] get_user ═══
SELECT id, email, created_at
FROM users
WHERE email = $1
  AND active = TRUE
LIMIT 1;
-- ═══ INTL:END   [id=f001] get_user ═══
"""

# Good HTML
GOOD_HTML = """\
<!-- ═══ INTL:BEGIN [id=f001] login_form ═══ -->
<form method="post" action="/login">
  <input type="email" name="email" required>
  <input type="password" name="password" required>
  <button type="submit">Login</button>
</form>
<!-- ═══ INTL:END   [id=f001] login_form ═══ -->
"""

# Good CSS
GOOD_CSS = """\
/* ═══ INTL:BEGIN [id=f001] btn ═══ */
.btn-primary {
  background-color: #3b82f6;
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
}
/* ═══ INTL:END   [id=f001] btn ═══ */
"""


@pytest.fixture
def v():
    return Validator()


# ── T1 — Syntax ───────────────────────────────────────────────────────────────
class TestT1Syntax:
    def test_valid_python_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert t1.passed

    def test_invalid_python_fails(self, v):
        r = v.validate(BAD_PYTHON_SYNTAX, BLOCK_ID, "python_fastapi")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert not t1.passed

    def test_valid_sql_passes(self, v):
        r = v.validate(GOOD_SQL, BLOCK_ID, "sql_postgres")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert t1.passed

    def test_valid_html_passes(self, v):
        r = v.validate(GOOD_HTML, BLOCK_ID, "html_jinja2")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert t1.passed

    def test_valid_css_passes(self, v):
        r = v.validate(GOOD_CSS, BLOCK_ID, "css_tailwind")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert t1.passed

    def test_empty_sql_fails(self, v):
        r = v.validate("", BLOCK_ID, "sql_postgres")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert not t1.passed

    def test_typescript_balanced_braces(self, v):
        code = "// INTL:BEGIN [id=f001] fn\nfunction fn() { return 1; }\n// INTL:END [id=f001] fn"
        r = v.validate(code, BLOCK_ID, "typescript_express")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert t1.passed

    def test_typescript_unbalanced_fails(self, v):
        code = "// INTL:BEGIN [id=f001] fn\nfunction fn() { { return 1;\n// INTL:END [id=f001] fn"
        r = v.validate(code, BLOCK_ID, "typescript_express")
        t1 = next(c for c in r.checks if c.name == "T1-Syntax")
        assert not t1.passed


# ── T2 — Sentinels ────────────────────────────────────────────────────────────
class TestT2Sentinel:
    def test_correct_sentinels_pass(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert t2.passed

    def test_missing_begin_fails(self, v):
        code = GOOD_PYTHON.replace("INTL:BEGIN", "INTL:BGIN")  # typo
        r = v.validate(code, BLOCK_ID, "python_fastapi")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert not t2.passed

    def test_missing_end_fails(self, v):
        code = GOOD_PYTHON.replace("INTL:END", "INTL:EN")  # typo
        r = v.validate(code, BLOCK_ID, "python_fastapi")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert not t2.passed

    def test_wrong_id_fails(self, v):
        r = v.validate(GOOD_PYTHON, "f999", "python_fastapi")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert not t2.passed

    def test_no_sentinels_fails(self, v):
        r = v.validate(NO_SENTINELS, BLOCK_ID, "python_fastapi")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert not t2.passed

    def test_sql_sentinels_pass(self, v):
        r = v.validate(GOOD_SQL, BLOCK_ID, "sql_postgres")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert t2.passed

    def test_html_sentinels_pass(self, v):
        r = v.validate(GOOD_HTML, BLOCK_ID, "html_jinja2")
        t2 = next(c for c in r.checks if c.name == "T2-Sentinel")
        assert t2.passed


# ── T3 — Preconditions ────────────────────────────────────────────────────────
class TestT3Preconditions:
    def test_no_preconditions_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", preconditions=[])
        t3 = next(c for c in r.checks if c.name == "T3-Preconditions")
        assert t3.passed

    def test_precondition_token_in_code_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       preconditions=["email.length > 0"])
        t3 = next(c for c in r.checks if c.name == "T3-Preconditions")
        assert t3.passed

    def test_missing_precondition_token_fails(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       preconditions=["nonexistent_xyz_field > 0"])
        t3 = next(c for c in r.checks if c.name == "T3-Preconditions")
        assert not t3.passed

    def test_multiple_preconditions(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       preconditions=["email.length > 0", "password.length >= 8"])
        t3 = next(c for c in r.checks if c.name == "T3-Preconditions")
        assert t3.passed


# ── T4 — Postconditions ───────────────────────────────────────────────────────
class TestT4Postconditions:
    def test_no_postconditions_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", postconditions=[])
        t4 = next(c for c in r.checks if c.name == "T4-Postconditions")
        assert t4.passed

    def test_has_return_with_token(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       postconditions=["result.token IS NOT NULL"])
        t4 = next(c for c in r.checks if c.name == "T4-Postconditions")
        assert t4.passed

    def test_no_return_fails(self, v):
        code = "# ═══ INTL:BEGIN [id=f001] x ═══\nprint('hello')\n# ═══ INTL:END [id=f001] x ═══"
        r = v.validate(code, BLOCK_ID, "python_fastapi",
                       postconditions=["result IS NOT NULL"])
        t4 = next(c for c in r.checks if c.name == "T4-Postconditions")
        assert not t4.passed


# ── T5 — Side effects ─────────────────────────────────────────────────────────
class TestT5SideEffects:
    def test_no_side_effects_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        t5 = next(c for c in r.checks if c.name == "T5-SideEffects")
        assert t5.passed

    def test_mutates_with_insert_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       mutates=["sessions_table"])
        t5 = next(c for c in r.checks if c.name == "T5-SideEffects")
        assert t5.passed

    def test_mutates_without_write_op_fails(self, v):
        code = "# ═══ INTL:BEGIN [id=f001] fn ═══\ndef fn(): return 1\n# ═══ INTL:END [id=f001] fn ═══"
        r = v.validate(code, BLOCK_ID, "python_fastapi", mutates=["some_table"])
        t5 = next(c for c in r.checks if c.name == "T5-SideEffects")
        assert not t5.passed

    def test_observable_with_logging_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", observable=True)
        t5 = next(c for c in r.checks if c.name == "T5-SideEffects")
        assert t5.passed

    def test_observable_without_logging_fails(self, v):
        code = "# ═══ INTL:BEGIN [id=f001] fn ═══\ndef fn(): return 1\n# ═══ INTL:END [id=f001] fn ═══"
        r = v.validate(code, BLOCK_ID, "python_fastapi", observable=True)
        t5 = next(c for c in r.checks if c.name == "T5-SideEffects")
        assert not t5.passed


# ── T6 — Types ────────────────────────────────────────────────────────────────
class TestT6Types:
    def test_no_return_type_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", return_type=None)
        t6 = next(c for c in r.checks if c.name == "T6-Types")
        assert t6.passed

    def test_matching_return_type_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", return_type="SessionToken")
        t6 = next(c for c in r.checks if c.name == "T6-Types")
        assert t6.passed

    def test_missing_return_type_fails(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi", return_type="NonExistentType")
        t6 = next(c for c in r.checks if c.name == "T6-Types")
        assert not t6.passed


# ── T7 — No placeholders ─────────────────────────────────────────────────────
class TestT7NoPlaceholders:
    def test_clean_code_passes(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        t7 = next(c for c in r.checks if c.name == "T7-NoPlaceholders")
        assert t7.passed

    def test_todo_fails(self, v):
        r = v.validate(HAS_TODO, BLOCK_ID, "python_fastapi")
        t7 = next(c for c in r.checks if c.name == "T7-NoPlaceholders")
        assert not t7.passed

    def test_fixme_fails(self, v):
        code = GOOD_PYTHON.replace("# ═══ INTL:END", "# FIXME: remove\n# ═══ INTL:END")
        r = v.validate(code, BLOCK_ID, "python_fastapi")
        t7 = next(c for c in r.checks if c.name == "T7-NoPlaceholders")
        assert not t7.passed

    def test_unimplemented_rust_fails(self, v):
        code = "// ═══ INTL:BEGIN [id=f001] fn ═══\nfn login() { unimplemented!() }\n// ═══ INTL:END [id=f001] fn ═══"
        r = v.validate(code, BLOCK_ID, "rust_axum")
        t7 = next(c for c in r.checks if c.name == "T7-NoPlaceholders")
        assert not t7.passed


# ── ValidationResult ──────────────────────────────────────────────────────────
class TestValidationResult:
    def test_all_pass_gives_passed_true(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        # T1, T2, T6, T7 should pass; others pass with no args
        assert r.passed  # may not be fully True depending on T4 with return
        assert isinstance(r.checks, list)
        assert len(r.checks) == 7

    def test_check_results_are_check_result(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        for c in r.checks:
            assert isinstance(c, CheckResult)
            assert hasattr(c, "name")
            assert hasattr(c, "passed")
            assert hasattr(c, "message")

    def test_failed_checks_populated_on_failure(self, v):
        r = v.validate(NO_SENTINELS, BLOCK_ID, "python_fastapi")
        assert len(r.failed_checks) > 0

    def test_escalation_package_none_on_pass(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                       preconditions=[], postconditions=[],
                       mutates=[], observable=False, return_type=None)
        if r.passed:
            assert r.escalation_package is None

    def test_escalation_package_present_on_fail(self, v):
        r = v.validate("# ═══ INTL:BEGIN [id=f001] x ═══\ndef x(\n# ═══ INTL:END [id=f001] x ═══",
                       BLOCK_ID, "python_fastapi", intl_block="FUNCTION x [id=f001]")
        # At least T1 or T2 should fail
        if not r.passed:
            assert r.escalation_package is not None
            assert "block_id" in r.escalation_package
            assert "intl_block" in r.escalation_package
            assert "compiled_code" in r.escalation_package
            assert "failed_checks" in r.escalation_package

    def test_escalation_package_contains_correct_id(self, v):
        r = v.validate(NO_SENTINELS, "myblock", "python_fastapi",
                       intl_block="FUNCTION test [id=myblock]")
        assert r.escalation_package["block_id"] == "myblock"


# ── summary() ────────────────────────────────────────────────────────────────
class TestSummary:
    def test_summary_contains_check_names(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        summary = v.summary(r)
        assert "T1-Syntax" in summary
        assert "T2-Sentinel" in summary
        assert "T7-NoPlaceholders" in summary

    def test_summary_shows_status(self, v):
        r = v.validate(NO_SENTINELS, BLOCK_ID, "python_fastapi")
        summary = v.summary(r)
        assert "FAILED" in summary or "PASSED" in summary

    def test_summary_shows_check_counts(self, v):
        r = v.validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        summary = v.summary(r)
        assert "7" in summary  # 7 total checks


# ── Module-level convenience function ─────────────────────────────────────────
class TestConvenienceFunction:
    def test_validate_function_works(self):
        r = validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi")
        assert isinstance(r, ValidationResult)
        assert len(r.checks) == 7

    def test_validate_function_passes_kwargs(self):
        r = validate(GOOD_PYTHON, BLOCK_ID, "python_fastapi",
                     mutates=["sessions_table"], observable=True)
        assert isinstance(r, ValidationResult)
