"""Tests for intl/escalation.py.

Tests cover:
  - escalate() returns corrected code with sentinels
  - Missing sentinels → EscalationError
  - Category C training pair is written to corrections.jsonl
  - index.record_compiled() is called when index is provided
  - EscalationError raised on API failure

Claude API calls are mocked with unittest.mock.
"""
from __future__ import annotations

import json
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch as mock_patch

# ── Helpers ───────────────────────────────────────────────────────────────────
INTL_BLOCK = textwrap.dedent("""\
    FUNCTION login [id=f001]
    INTENT       "validate credentials and return session token"
    PRECONDITION email.length > 0
    PRECONDITION password.length >= 8
    POSTCONDITION result.token IS NOT NULL
    READS        users_table
    MUTATES      sessions_table
    OBSERVABLE
    RETURN SessionToken(token: token)
    END FUNCTION login [id=f001]
""")

FAILED_OUTPUT = textwrap.dedent("""\
    async def login(email, password):
        # TODO: implement
        pass
""")

FAILING_CHECKS = [
    {"name": "T7-NoPlaceholders", "message": "TODO found", "detail": "line 2: # TODO: implement"},
    {"name": "T4-Postconditions", "message": "result.token not asserted", "detail": ""},
]

CORRECTED_CODE = textwrap.dedent("""\
    # ═══ INTL:BEGIN [id=f001] login ═══
    import logging
    from datetime import datetime, timedelta

    logger = logging.getLogger(__name__)

    async def login(email: str, password: str) -> SessionToken:
        assert len(email) > 0, "email must not be empty"
        assert len(password) >= 8, "password must be at least 8 chars"
        user = await db.query(users_table).filter(email=email).first()
        if not user or not verify_hash(password, user.password_hash):
            raise AuthError("invalid_credentials")
        token = generate_token(user.id)
        await db.insert(sessions_table, user_id=user.id, token=token)
        logger.info("login: user=%s", user.id)
        result = SessionToken(token=token)
        assert result.token is not None
        return result
    # ═══ INTL:END   [id=f001] login ═══
""")


def _fake_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestEscalateSuccess(unittest.TestCase):
    """escalate() happy path — Claude returns corrected code with sentinels."""

    def _call(self, corrected: str = CORRECTED_CODE, index=None) -> str:
        from intl.escalation import escalate
        with mock_patch("intl.escalation._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(corrected)
            return escalate(
                intl_block=INTL_BLOCK,
                failed_output=FAILED_OUTPUT,
                failing_checks=FAILING_CHECKS,
                language="Python",
                adapter="python_fastapi",
                retry_count=2,
                block_id="f001",
                block_name="login",
                index=index,
            )

    def test_returns_corrected_code(self):
        result = self._call()
        self.assertIn("INTL:BEGIN", result)
        self.assertIn("INTL:END", result)

    def test_markdown_fences_stripped(self):
        fenced = f"```python\n{CORRECTED_CODE.strip()}\n```"
        result = self._call(fenced)
        self.assertIn("INTL:BEGIN", result)

    def test_index_record_compiled_called_when_provided(self):
        mock_index = MagicMock()
        self._call(index=mock_index)
        mock_index.record_compiled.assert_called_once_with("f001", unittest.mock.ANY)

    def test_index_not_required(self):
        # Should not raise even without an index
        result = self._call(index=None)
        self.assertIn("INTL:BEGIN", result)


class TestEscalateFailure(unittest.TestCase):
    """escalate() should raise EscalationError on bad output."""

    def test_missing_sentinels_raises(self):
        from intl.escalation import EscalationError, escalate
        bad = "def login(): return None  # no sentinels"
        with mock_patch("intl.escalation._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(bad)
            with self.assertRaises(EscalationError):
                escalate(
                    intl_block=INTL_BLOCK,
                    failed_output=FAILED_OUTPUT,
                    failing_checks=FAILING_CHECKS,
                    language="Python",
                    adapter="python_fastapi",
                    retry_count=1,
                )

    def test_api_error_raises_escalation_error(self):
        import anthropic
        from intl.escalation import EscalationError, escalate
        with mock_patch("intl.escalation._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = anthropic.APIError(
                message="rate limit", request=MagicMock(), body=None
            )
            with self.assertRaises(EscalationError):
                escalate(
                    intl_block=INTL_BLOCK,
                    failed_output=FAILED_OUTPUT,
                    failing_checks=FAILING_CHECKS,
                    language="Python",
                    adapter="python_fastapi",
                    retry_count=1,
                )

    def test_no_api_key_raises_escalation_error(self):
        import os
        from intl.escalation import EscalationError, escalate
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        # Reset cached client
        import intl.escalation as esc_mod
        esc_mod._client = None
        try:
            with self.assertRaises(EscalationError):
                escalate(
                    intl_block=INTL_BLOCK,
                    failed_output=FAILED_OUTPUT,
                    failing_checks=FAILING_CHECKS,
                    language="Python",
                    adapter="python_fastapi",
                    retry_count=1,
                )
        finally:
            if orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig
            esc_mod._client = None


class TestCorrectionPairSaved(unittest.TestCase):
    """Category C training pair must be written to corrections.jsonl."""

    def test_corrections_jsonl_written(self):
        import tempfile
        from unittest.mock import patch as up

        from intl.escalation import escalate

        with tempfile.TemporaryDirectory() as tmpdir:
            with up("intl.escalation.DATA_DIR", new=Path(tmpdir)):
                with up("intl.escalation._get_client") as mock_client:
                    mock_client.return_value.messages.create.return_value = _fake_response(CORRECTED_CODE)
                    escalate(
                        intl_block=INTL_BLOCK,
                        failed_output=FAILED_OUTPUT,
                        failing_checks=FAILING_CHECKS,
                        language="Python",
                        adapter="python_fastapi",
                        retry_count=1,
                        block_id="f001",
                        block_name="login",
                    )

                corrections = Path(tmpdir) / "python_fastapi" / "corrections.jsonl"
                self.assertTrue(corrections.exists())
                lines = corrections.read_text().strip().splitlines()
                self.assertEqual(len(lines), 1)

                pair = json.loads(lines[0])
                self.assertEqual(pair["metadata"]["category"], "C")
                self.assertEqual(pair["metadata"]["block_id"], "f001")
                self.assertIn("login", pair["metadata"]["block_name"])
                self.assertIn("INTL:BEGIN", pair["completion"])

    def test_multiple_calls_append(self):
        import tempfile
        from unittest.mock import patch as up

        from intl.escalation import escalate

        with tempfile.TemporaryDirectory() as tmpdir:
            with up("intl.escalation.DATA_DIR", new=Path(tmpdir)):
                with up("intl.escalation._get_client") as mock_client:
                    mock_client.return_value.messages.create.return_value = _fake_response(CORRECTED_CODE)
                    for _ in range(3):
                        escalate(
                            intl_block=INTL_BLOCK,
                            failed_output=FAILED_OUTPUT,
                            failing_checks=FAILING_CHECKS,
                            language="Python",
                            adapter="python_fastapi",
                            retry_count=1,
                        )

                corrections = Path(tmpdir) / "python_fastapi" / "corrections.jsonl"
                lines = corrections.read_text().strip().splitlines()
                self.assertEqual(len(lines), 3)


class TestSentinelDetection(unittest.TestCase):
    """_has_sentinels() should detect all comment-style sentinels."""

    def test_python_style(self):
        from intl.escalation import _has_sentinels
        code = "# ═══ INTL:BEGIN [id=f001] test ═══\npass\n# ═══ INTL:END   [id=f001] test ═══"
        self.assertTrue(_has_sentinels(code))

    def test_sql_style(self):
        from intl.escalation import _has_sentinels
        code = "-- ═══ INTL:BEGIN [id=q001] query ═══\nSELECT 1;\n-- ═══ INTL:END   [id=q001] query ═══"
        self.assertTrue(_has_sentinels(code))

    def test_html_style(self):
        from intl.escalation import _has_sentinels
        code = "<!-- ═══ INTL:BEGIN [id=h001] tmpl ═══ -->\n<div/>\n<!-- ═══ INTL:END [id=h001] tmpl ═══ -->"
        self.assertTrue(_has_sentinels(code))

    def test_css_style(self):
        from intl.escalation import _has_sentinels
        code = "/* ═══ INTL:BEGIN [id=c001] styles ═══ */\n.x{}\n/* ═══ INTL:END [id=c001] styles ═══ */"
        self.assertTrue(_has_sentinels(code))

    def test_no_sentinels_returns_false(self):
        from intl.escalation import _has_sentinels
        self.assertFalse(_has_sentinels("def foo(): pass"))

    def test_only_begin_returns_false(self):
        from intl.escalation import _has_sentinels
        self.assertFalse(_has_sentinels("# ═══ INTL:BEGIN [id=f001] x ═══\npass"))


class TestEscalationImport(unittest.TestCase):
    def test_all_public_symbols_importable(self):
        from intl.escalation import (  # noqa: F401
            EscalationError,
            _has_sentinels,
            escalate,
        )


if __name__ == "__main__":
    unittest.main()
