"""Tests for intl/generator.py.

These tests cover the pure / non-Claude logic:
  - _has_sentinels() equivalent logic in patch validation
  - GeneratorError is raised on bad JSON from decompose()
  - generate_module() validates output through the parser (patched Claude)
  - patch() validates that returned text starts with PATCH

Claude API calls are mocked with unittest.mock so no real API key is needed.
"""
from __future__ import annotations

import json
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch as mock_patch

# ── Helpers to build fake Anthropic response ──────────────────────────────────
def _fake_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestDecomposeJSON(unittest.TestCase):
    """decompose() should parse Claude's JSON response into a dict."""

    def _call(self, claude_text: str) -> dict:
        from intl.generator import decompose
        with mock_patch("intl.generator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(claude_text)
            return decompose("Build a todo app")

    def test_valid_manifest_returned_as_dict(self):
        manifest = {
            "project": "todo",
            "description": "A simple todo app",
            "modules": [
                {
                    "id": "m001",
                    "name": "tasks",
                    "target": "src/tasks.py",
                    "profile": "python_fastapi",
                    "namespace": "app.tasks",
                    "requires": [],
                    "description": "Task management",
                    "functions": [],
                    "pipelines": [],
                    "types": [],
                }
            ],
            "shared_types": [],
            "dependency_graph": {"m001": []},
        }
        result = self._call(json.dumps(manifest))
        self.assertEqual(result["project"], "todo")
        self.assertEqual(len(result["modules"]), 1)
        self.assertEqual(result["modules"][0]["id"], "m001")

    def test_markdown_fences_stripped(self):
        manifest = {"project": "x", "modules": []}
        raw = f"```json\n{json.dumps(manifest)}\n```"
        result = self._call(raw)
        self.assertEqual(result["project"], "x")

    def test_invalid_json_raises_generator_error(self):
        from intl.generator import GeneratorError
        with self.assertRaises(GeneratorError):
            self._call("this is not json")

    def test_missing_modules_key_raises_generator_error(self):
        from intl.generator import GeneratorError
        with self.assertRaises(GeneratorError):
            self._call('{"project": "x"}')

    def test_modules_not_list_raises_generator_error(self):
        from intl.generator import GeneratorError
        with self.assertRaises(GeneratorError):
            self._call('{"project": "x", "modules": "bad"}')


class TestGenerateModule(unittest.TestCase):
    """generate_module() should write a .intl file and validate it through the parser."""

    _VALID_INTL = textwrap.dedent("""\
        MODULE tasks [id=m001]
        TARGET    "src/tasks.py"
        PROFILE   python_fastapi
        NAMESPACE "app.tasks"

        FUNCTION create_task [id=f001]
        INTENT       "create a new task and persist it"
        PRECONDITION title.length > 0
        POSTCONDITION result IS NOT NULL
        MUTATES      tasks_table
        PERSIST tasks_table INSERT (title: title)
        RETURN Task(title: title)
        END FUNCTION create_task [id=f001]
    """)

    _MANIFEST = {
        "project": "todo",
        "modules": [
            {
                "id": "m001",
                "name": "tasks",
                "target": "src/tasks.py",
                "profile": "python_fastapi",
                "namespace": "app.tasks",
                "requires": [],
                "description": "Task management",
                "functions": [{"id": "f001", "name": "create_task", "description": "create a task"}],
                "pipelines": [],
                "types": [],
            }
        ],
    }

    def test_valid_intl_is_written_and_returned(self):
        from intl.generator import generate_module
        with mock_patch("intl.generator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(self._VALID_INTL)
            result = generate_module(self._MANIFEST, self._MANIFEST["modules"][0])

        self.assertIn("MODULE tasks", result)
        self.assertIn("FUNCTION create_task", result)

        # Check file was written
        out = Path(__file__).parent.parent / "generated" / "tasks.intl"
        self.assertTrue(out.exists())
        self.assertIn("MODULE tasks", out.read_text())

    def test_parser_rejection_raises_generator_error(self):
        from intl.generator import GeneratorError, generate_module
        bad_intl = "this is not valid intl at all ##!!"
        with mock_patch("intl.generator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(bad_intl)
            with self.assertRaises(GeneratorError):
                generate_module(self._MANIFEST, self._MANIFEST["modules"][0])

    def test_markdown_fences_stripped_before_parse(self):
        from intl.generator import generate_module
        fenced = f"```intl\n{self._VALID_INTL.strip()}\n```"
        with mock_patch("intl.generator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(fenced)
            result = generate_module(self._MANIFEST, self._MANIFEST["modules"][0])
        self.assertIn("MODULE tasks", result)


class TestPatch(unittest.TestCase):
    """patch() should return a string that starts with PATCH."""

    _EXISTING_INTL = textwrap.dedent("""\
        MODULE tasks [id=m001]
        TARGET    "src/tasks.py"
        PROFILE   python_fastapi
        NAMESPACE "app.tasks"

        FUNCTION create_task [id=f001]
        INTENT       "create a new task"
        PRECONDITION title.length > 0
        POSTCONDITION result IS NOT NULL
        RETURN Task(title: title)
        END FUNCTION create_task [id=f001]
    """)

    def _call(self, claude_text: str) -> str:
        from intl.generator import patch
        with mock_patch("intl.generator._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = _fake_response(claude_text)
            return patch(self._EXISTING_INTL, "Add a description field to create_task")

    def test_valid_patch_block_returned(self):
        patch_block = textwrap.dedent("""\
            PATCH [target=f001] [position=replace]
            PRECONDITION title.length > 0
            PRECONDITION description.length >= 0
            POSTCONDITION result IS NOT NULL
            RETURN Task(title: title, description: description)
            END PATCH
        """)
        result = self._call(patch_block)
        self.assertTrue(result.upper().startswith("PATCH"))

    def test_non_patch_response_raises_generator_error(self):
        from intl.generator import GeneratorError
        with self.assertRaises(GeneratorError):
            self._call("Sure! Here is my suggestion: add description.")

    def test_markdown_fences_stripped(self):
        patch_block = "PATCH [target=f001] [position=replace]\nRETURN Task()\nEND PATCH"
        fenced = f"```intl\n{patch_block}\n```"
        result = self._call(fenced)
        self.assertTrue(result.upper().startswith("PATCH"))


class TestGeneratorImport(unittest.TestCase):
    def test_all_public_symbols_importable(self):
        from intl.generator import (  # noqa: F401
            GeneratorError,
            decompose,
            generate_all_modules,
            generate_module,
            patch,
        )


if __name__ == "__main__":
    unittest.main()
