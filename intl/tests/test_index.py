"""Tests for intl.index — SemanticIndex SQLite store (issue #10)."""
import time
import pytest
from intl.index import SemanticIndex, DirtyNode
from intl.parser import parse


INTL_M001 = """
MODULE auth [id=m001]
TARGET    "src/auth.py"
PROFILE   python_fastapi
NAMESPACE "app.auth"
REQUIRES  m002

FUNCTION login [id=f001]
  INTENT "validate credentials"
  PRECONDITION email.length > 0
  POSTCONDITION result.token IS NOT NULL
  READS   users_table
  MUTATES sessions_table
  RETURN SessionToken(token: token)
END FUNCTION login [id=f001]

FUNCTION logout [id=f002]
  INTENT "invalidate a session"
  PRECONDITION session_id IS NOT NULL
  MUTATES sessions_table
  RETURN "ok"
END FUNCTION logout [id=f002]
"""

INTL_M002 = """
MODULE users [id=m002]
TARGET    "src/users.py"
PROFILE   python_fastapi
NAMESPACE "app.users"

FUNCTION get_user [id=f003]
  INTENT "fetch user by id"
  PRECONDITION user_id > 0
  READS users_table
  RETURN User(id: user_id)
END FUNCTION get_user [id=f003]
"""

INTL_M003 = """
MODULE orders [id=m003]
TARGET    "src/orders.py"
PROFILE   python_fastapi
NAMESPACE "app.orders"
REQUIRES  m001

FUNCTION create_order [id=f004]
  INTENT "create a new order"
  PRECONDITION items.length > 0
  MUTATES orders_table
  RETURN Order(id: order_id)
END FUNCTION create_order [id=f004]
"""


@pytest.fixture
def idx():
    """In-memory SemanticIndex instance."""
    index = SemanticIndex(":memory:")
    yield index
    index.close()


@pytest.fixture
def m001(idx):
    mod = parse(INTL_M001)
    idx.index_module(mod)
    return mod


@pytest.fixture
def m002(idx):
    mod = parse(INTL_M002)
    idx.index_module(mod)
    return mod


# ── index_module ──────────────────────────────────────────────────────────────
class TestIndexModule:
    def test_returns_block_ids(self, idx):
        mod = parse(INTL_M001)
        ids = idx.index_module(mod)
        assert set(ids) == {"f001", "f002"}

    def test_module_stored(self, idx, m001):
        m = idx.get_module("m001")
        assert m is not None
        assert m["name"] == "auth"
        assert m["profile"] == "python_fastapi"
        assert m["target"] == "src/auth.py"

    def test_blocks_stored(self, idx, m001):
        blocks = idx.get_blocks_for_module("m001")
        assert len(blocks) == 2
        bids = {b["id"] for b in blocks}
        assert bids == {"f001", "f002"}

    def test_new_blocks_are_dirty(self, idx, m001):
        blocks = idx.get_blocks_for_module("m001")
        assert all(b["dirty"] == 1 for b in blocks)

    def test_dependency_stored(self, idx, m001, m002):
        # m001 REQUIRES m002 — dependency row should exist
        dep = idx.db.execute(
            "SELECT * FROM dependencies WHERE from_module='m001' AND to_module='m002'"
        ).fetchone()
        assert dep is not None

    def test_idempotent_reindex(self, idx):
        mod = parse(INTL_M001)
        ids1 = idx.index_module(mod)
        ids2 = idx.index_module(mod)
        assert set(ids1) == set(ids2)
        assert idx.stats()["modules"] == 1

    def test_multiple_modules(self, idx, m001, m002):
        stats = idx.stats()
        assert stats["modules"] == 2
        assert stats["blocks"] >= 3


# ── get_dirty_nodes ───────────────────────────────────────────────────────────
class TestDirtyNodes:
    def test_all_new_blocks_dirty(self, idx, m001):
        dirty = idx.get_dirty_nodes()
        assert len(dirty) == 2
        assert all(isinstance(d, DirtyNode) for d in dirty)

    def test_dirty_node_fields(self, idx, m001):
        dirty = idx.get_dirty_nodes()
        node = next(d for d in dirty if d.block_id == "f001")
        assert node.module_id == "m001"
        assert node.name == "login"
        assert node.profile == "python_fastapi"
        assert node.kind in ("function", "pipeline", "type", "patch", "functionblock")

    def test_after_compile_dirty_reduces(self, idx, m001):
        idx.record_compiled("f001", "def login(): pass")
        dirty = idx.get_dirty_nodes()
        dirty_ids = {d.block_id for d in dirty}
        assert "f001" not in dirty_ids
        assert "f002" in dirty_ids


# ── record_compiled ───────────────────────────────────────────────────────────
class TestRecordCompiled:
    def test_marks_block_clean(self, idx, m001):
        idx.record_compiled("f001", "async def login(): ...")
        block = idx.get_block("f001")
        assert block["dirty"] == 0
        assert block["output_code"] == "async def login(): ..."

    def test_compiled_at_set(self, idx, m001):
        before = time.time()
        idx.record_compiled("f001", "async def login(): ...")
        after = time.time()
        block = idx.get_block("f001")
        assert before <= block["compiled_at"] <= after

    def test_module_goes_clean_when_all_blocks_compiled(self, idx, m001):
        idx.record_compiled("f001", "def login(): ...")
        idx.record_compiled("f002", "def logout(): ...")
        mod = idx.get_module("m001")
        assert mod["dirty"] == 0

    def test_module_stays_dirty_while_one_block_dirty(self, idx, m001):
        idx.record_compiled("f001", "def login(): ...")
        mod = idx.get_module("m001")
        assert mod["dirty"] == 1


# ── mark_dirty ───────────────────────────────────────────────────────────────
class TestMarkDirty:
    def test_mark_clean_module_dirty(self, idx, m001):
        idx.record_compiled("f001", "...")
        idx.record_compiled("f002", "...")
        assert idx.get_module("m001")["dirty"] == 0
        idx.mark_dirty("m001")
        assert idx.get_module("m001")["dirty"] == 1

    def test_mark_dirty_marks_all_blocks(self, idx, m001):
        idx.record_compiled("f001", "...")
        idx.record_compiled("f002", "...")
        idx.mark_dirty("m001")
        blocks = idx.get_blocks_for_module("m001")
        assert all(b["dirty"] == 1 for b in blocks)


# ── Dirty propagation ─────────────────────────────────────────────────────────
class TestDirtyPropagation:
    def test_change_to_dependency_marks_dependent_dirty(self, idx):
        """m003 REQUIRES m001. Dirtying m001 should cascade to m003."""
        m1 = parse(INTL_M001)
        m3 = parse(INTL_M003)
        idx.index_module(m1)
        idx.index_module(m3)

        # Compile everything clean
        for bid in ("f001", "f002"):
            idx.record_compiled(bid, "...")
        idx.record_compiled("f004", "...")

        # Both modules should be clean
        assert idx.get_module("m001")["dirty"] == 0
        assert idx.get_module("m003")["dirty"] == 0

        # Now dirty m001
        idx.mark_dirty("m001")

        # m003 depends on m001 → should be dirty too
        assert idx.get_module("m001")["dirty"] == 1
        assert idx.get_module("m003")["dirty"] == 1

    def test_independent_module_not_affected(self, idx, m001, m002):
        """m002 does not require m001, so dirtying m001 should not affect m002."""
        # Compile m002 clean
        idx.record_compiled("f003", "...")
        assert idx.get_module("m002")["dirty"] == 0

        idx.mark_dirty("m001")
        # m002 should remain clean
        assert idx.get_module("m002")["dirty"] == 0


# ── stats ─────────────────────────────────────────────────────────────────────
class TestStats:
    def test_empty_index(self, idx):
        s = idx.stats()
        assert s == {"modules": 0, "blocks": 0, "dirty_blocks": 0}

    def test_after_indexing(self, idx, m001, m002):
        s = idx.stats()
        assert s["modules"] == 2
        assert s["blocks"] == 3
        assert s["dirty_blocks"] == 3

    def test_after_compilation(self, idx, m001):
        idx.record_compiled("f001", "...")
        idx.record_compiled("f002", "...")
        s = idx.stats()
        assert s["dirty_blocks"] == 0


# ── get helpers ───────────────────────────────────────────────────────────────
class TestGetHelpers:
    def test_get_block_returns_none_for_unknown(self, idx):
        assert idx.get_block("nonexistent") is None

    def test_get_module_returns_none_for_unknown(self, idx):
        assert idx.get_module("nonexistent") is None

    def test_get_all_modules_ordered(self, idx, m001, m002):
        mods = idx.get_all_modules()
        ids = [m["id"] for m in mods]
        assert ids == sorted(ids)

    def test_get_blocks_for_module(self, idx, m001):
        blocks = idx.get_blocks_for_module("m001")
        assert len(blocks) == 2
        for b in blocks:
            assert b["module_id"] == "m001"

    def test_get_blocks_empty_for_unknown_module(self, idx):
        blocks = idx.get_blocks_for_module("unknown")
        assert blocks == []
