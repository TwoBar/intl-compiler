"""Tests for intl.parser — Lark grammar + typed AST (issue #9)."""
import pytest
from intl.parser import (
    parse, parse_safe,
    Module, FunctionBlock, TypeBlock, PatchBlock,
    Query, Persist, Assign, If, Fail, Return, Loop, Emit, Block,
    Literal, Ref, BinOp, ParseError,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
MINIMAL_MODULE = """
MODULE auth [id=m001]
TARGET    "src/services/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.services.auth"
"""

FULL_FUNCTION = """
MODULE auth [id=m001]
TARGET    "src/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.auth"

FUNCTION login [id=f001]
  INTENT       "validate credentials and return a session token"
  PRECONDITION email.length > 0
  PRECONDITION password.length >= 8
  POSTCONDITION result.token IS NOT NULL
  READS        users_table
  MUTATES      sessions_table
  OBSERVABLE
  CONFIDENCE   0.95

  user = QUERY users_table WHERE email == email LIMIT 1
  IF user IS NULL THEN FAIL AuthError("user_not_found")
  token = generate_token(user.id)
  PERSIST sessions_table INSERT (user_id: user.id, token: token)
  RETURN SessionToken(token: token)
END FUNCTION login [id=f001]
"""

PIPELINE_SOURCE = """
MODULE shop [id=m002]
TARGET    "src/checkout.py"
PROFILE   python_fastapi
NAMESPACE "app.checkout"

PIPELINE checkout [id=p001]
  INTENT "process a customer checkout"
  PRECONDITION cart.items.length > 0
  POSTCONDITION result.order_id IS NOT NULL
  MUTATES orders_table

  SEQUENCE
    order = QUERY orders_table WHERE id == cart.id LIMIT 1
    PERSIST orders_table INSERT (cart_id: cart.id, total: cart.total)
  END SEQUENCE
  RETURN OrderConfirmation(order_id: order.id)
END PIPELINE checkout [id=p001]
"""

TYPE_SOURCE = """
MODULE types [id=m003]
TARGET    "src/types.py"
PROFILE   python_fastapi
NAMESPACE "app.types"

TYPE SessionToken [id=t001]
  token     : String NOT NULL
  expires_at: DateTime NOT NULL
  user_id   : Integer
END TYPE SessionToken [id=t001]
"""

PATCH_SOURCE = """
MODULE auth [id=m001]
TARGET    "src/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.auth"

PATCH FUNCTION login [id=f001]
  INTENT "add rate limiting to login"
  INSERT PRECONDITION rate_limit_not_exceeded
  INSERT BEFORE "user = QUERY"
    rate_check = QUERY rate_limits WHERE user_id == email LIMIT 1
    IF rate_check.attempts > 5 THEN FAIL RateLimitError("too_many_attempts")
  END INSERT
END PATCH FUNCTION login [id=f001]
"""

REQUIRES_SOURCE = """
MODULE auth [id=m001]
TARGET    "src/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.auth"
REQUIRES  m002 m003
VERSION   "1.2.0"

FUNCTION ping [id=f001]
  INTENT "health check"
  POSTCONDITION result IS NOT NULL
  RETURN "pong"
END FUNCTION ping [id=f001]
"""


# ── MODULE tests ──────────────────────────────────────────────────────────────
class TestModuleParsing:
    def test_minimal_module(self):
        mod = parse(MINIMAL_MODULE)
        assert isinstance(mod, Module)
        assert mod.name == "auth"
        assert mod.id == "m001"
        assert mod.target == "src/services/auth.py"
        assert mod.profile == "python_fastapi"
        assert mod.namespace == "app.services.auth"
        assert mod.blocks == []

    def test_requires_and_version(self):
        mod = parse(REQUIRES_SOURCE)
        assert mod.requires == ["m002", "m003"]
        assert mod.version == "1.2.0"

    def test_module_with_blocks(self):
        mod = parse(FULL_FUNCTION)
        assert len(mod.blocks) == 1


# ── FUNCTION tests ────────────────────────────────────────────────────────────
class TestFunctionParsing:
    def setup_method(self):
        self.mod = parse(FULL_FUNCTION)
        self.fn = self.mod.blocks[0]

    def test_function_type(self):
        assert isinstance(self.fn, FunctionBlock)

    def test_function_metadata(self):
        assert self.fn.name == "login"
        assert self.fn.id == "f001"
        assert "credentials" in self.fn.intent

    def test_preconditions(self):
        assert len(self.fn.preconditions) == 2

    def test_postconditions(self):
        assert len(self.fn.postconditions) == 1

    def test_reads_mutates(self):
        assert "users_table" in self.fn.reads
        assert "sessions_table" in self.fn.mutates

    def test_observable_flag(self):
        assert self.fn.observable is True

    def test_confidence(self):
        assert self.fn.confidence == pytest.approx(0.95)

    def test_body_not_empty(self):
        assert len(self.fn.body) > 0

    def test_query_in_body(self):
        queries = [s for s in self.fn.body if isinstance(s, Query)]
        assert len(queries) >= 1

    def test_assign_in_body(self):
        assigns = [s for s in self.fn.body if isinstance(s, Assign)]
        assert len(assigns) >= 1

    def test_persist_in_body(self):
        persists = [s for s in self.fn.body if isinstance(s, Persist)]
        assert len(persists) >= 1

    def test_return_in_body(self):
        returns = [s for s in self.fn.body if isinstance(s, Return)]
        assert len(returns) >= 1


# ── PIPELINE tests ────────────────────────────────────────────────────────────
class TestPipelineParsing:
    def setup_method(self):
        self.mod = parse(PIPELINE_SOURCE)
        self.pl = self.mod.blocks[0]

    def test_pipeline_kind(self):
        assert isinstance(self.pl, FunctionBlock)
        assert self.pl.kind == "pipeline"

    def test_pipeline_id(self):
        assert self.pl.id == "p001"

    def test_sequence_block(self):
        seqs = [s for s in self.pl.body if isinstance(s, Block) and s.kind == "sequence"]
        assert len(seqs) >= 1


# ── TYPE tests ────────────────────────────────────────────────────────────────
class TestTypeParsing:
    def setup_method(self):
        self.mod = parse(TYPE_SOURCE)
        self.ty = self.mod.blocks[0]

    def test_type_instance(self):
        assert isinstance(self.ty, TypeBlock)

    def test_type_name_id(self):
        assert self.ty.name == "SessionToken"
        assert self.ty.id == "t001"

    def test_fields(self):
        assert len(self.ty.fields) >= 2
        field_names = [f.name for f in self.ty.fields]
        assert "token" in field_names
        assert "expires_at" in field_names

    def test_not_null(self):
        token_field = next(f for f in self.ty.fields if f.name == "token")
        assert token_field.not_null is True


# ── PATCH tests ───────────────────────────────────────────────────────────────
class TestPatchParsing:
    def setup_method(self):
        self.mod = parse(PATCH_SOURCE)
        self.patch = self.mod.blocks[0]

    def test_patch_instance(self):
        assert isinstance(self.patch, PatchBlock)

    def test_patch_target(self):
        assert self.patch.name == "login"
        assert self.patch.id == "f001"

    def test_insert_precondition(self):
        assert len(self.patch.insert_preconditions) >= 1


# ── Error handling ────────────────────────────────────────────────────────────
class TestParseErrors:
    def test_parse_safe_returns_error_on_bad_input(self):
        mod, err = parse_safe("this is not valid INTL source !!!!")
        assert mod is None
        assert err is not None
        assert isinstance(err, ParseError)

    def test_parse_safe_returns_module_on_valid_input(self):
        mod, err = parse_safe(MINIMAL_MODULE)
        assert mod is not None
        assert err is None

    def test_error_has_line_info(self):
        _, err = parse_safe("BAD BAD BAD\nMORE BAD")
        if err:
            assert hasattr(err, "line")

    def test_empty_string_fails(self):
        mod, err = parse_safe("")
        assert err is not None

    def test_missing_end_fails(self):
        src = """
MODULE auth [id=m001]
TARGET    "src/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.auth"
FUNCTION login [id=f001]
  INTENT "login"
  RETURN result
"""
        mod, err = parse_safe(src)
        assert err is not None


# ── AST node structure ────────────────────────────────────────────────────────
class TestASTNodes:
    def test_module_has_position(self):
        mod = parse(MINIMAL_MODULE)
        assert hasattr(mod, "line")
        assert hasattr(mod, "col")

    def test_function_body_nodes_have_position(self):
        mod = parse(FULL_FUNCTION)
        fn = mod.blocks[0]
        for stmt in fn.body:
            assert hasattr(stmt, "line"), f"{type(stmt)} missing line"

    def test_function_returns_dataclass(self):
        mod = parse(FULL_FUNCTION)
        fn = mod.blocks[0]
        from dataclasses import fields
        assert len(fields(fn)) > 5

    def test_multiple_blocks(self):
        multi = MINIMAL_MODULE + "\n" + "\n".join(
            FULL_FUNCTION.split("\n")[7:]  # strip MODULE header
        )
        mod, err = parse_safe(MINIMAL_MODULE)
        assert err is None
