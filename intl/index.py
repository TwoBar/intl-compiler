"""INTL Semantic Index — SQLite-backed AST storage with dirty propagation."""
from __future__ import annotations
import json, sqlite3, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    target      TEXT NOT NULL,
    profile     TEXT NOT NULL,
    namespace   TEXT NOT NULL,
    version     TEXT DEFAULT '',
    ast_json    TEXT NOT NULL,
    dirty       INTEGER DEFAULT 1,
    compiled_at REAL DEFAULT 0,
    indexed_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS blocks (
    id          TEXT PRIMARY KEY,
    module_id   TEXT NOT NULL REFERENCES modules(id),
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,  -- function/pipeline/type/patch
    ast_json    TEXT NOT NULL,
    dirty       INTEGER DEFAULT 1,
    compiled_at REAL DEFAULT 0,
    output_code TEXT DEFAULT '',
    indexed_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dependencies (
    from_module TEXT NOT NULL REFERENCES modules(id),
    to_module   TEXT NOT NULL REFERENCES modules(id),
    PRIMARY KEY (from_module, to_module)
);

CREATE INDEX IF NOT EXISTS idx_blocks_module ON blocks(module_id);
CREATE INDEX IF NOT EXISTS idx_blocks_dirty ON blocks(dirty);
CREATE INDEX IF NOT EXISTS idx_deps_to ON dependencies(to_module);
"""


@dataclass
class DirtyNode:
    block_id: str
    module_id: str
    name: str
    kind: str
    profile: str


class SemanticIndex:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ── Index a parsed module ──
    def index_module(self, module) -> list[str]:
        """Store/update a Module AST. Returns list of block IDs indexed."""
        now = time.time()
        ast_blob = json.dumps({"name": module.name, "id": module.id,
                               "target": module.target, "profile": module.profile,
                               "namespace": module.namespace, "version": module.version,
                               "requires": module.requires})
        self.db.execute(
            "INSERT OR REPLACE INTO modules (id,name,target,profile,namespace,version,ast_json,dirty,indexed_at) "
            "VALUES (?,?,?,?,?,?,?,1,?)",
            (module.id, module.name, module.target, module.profile,
             module.namespace, module.version, ast_blob, now))

        # Update dependencies
        self.db.execute("DELETE FROM dependencies WHERE from_module=?", (module.id,))
        for dep in module.requires:
            self.db.execute("INSERT OR IGNORE INTO dependencies (from_module,to_module) VALUES (?,?)",
                            (module.id, dep))

        # Index blocks
        block_ids = []
        for block in module.blocks:
            bid = getattr(block, "id", None)
            bname = getattr(block, "name", "")
            bkind = getattr(block, "kind", type(block).__name__.lower().replace("block", ""))
            bast = json.dumps({"name": bname, "id": bid, "kind": bkind})
            self.db.execute(
                "INSERT OR REPLACE INTO blocks (id,module_id,name,kind,ast_json,dirty,indexed_at) "
                "VALUES (?,?,?,?,?,1,?)",
                (bid, module.id, bname, bkind, bast, now))
            block_ids.append(bid)

        self.db.commit()
        self._propagate_dirty(module.id)
        return block_ids

    # ── Dirty propagation ──
    def mark_dirty(self, module_id: str):
        """Mark a module and all its blocks dirty, then propagate to dependents."""
        self.db.execute("UPDATE modules SET dirty=1 WHERE id=?", (module_id,))
        self.db.execute("UPDATE blocks SET dirty=1 WHERE module_id=?", (module_id,))
        self.db.commit()
        self._propagate_dirty(module_id)

    def _propagate_dirty(self, module_id: str):
        """Cascade dirty flag to modules that REQUIRE this one."""
        rows = self.db.execute(
            "SELECT from_module FROM dependencies WHERE to_module=?", (module_id,)).fetchall()
        for row in rows:
            dep_id = row["from_module"]
            cur = self.db.execute("SELECT dirty FROM modules WHERE id=?", (dep_id,)).fetchone()
            if cur and not cur["dirty"]:
                self.db.execute("UPDATE modules SET dirty=1 WHERE id=?", (dep_id,))
                self.db.execute("UPDATE blocks SET dirty=1 WHERE module_id=?", (dep_id,))
                self.db.commit()
                self._propagate_dirty(dep_id)  # recursive

    # ── Query dirty nodes ──
    def get_dirty_nodes(self) -> list[DirtyNode]:
        """Return all blocks needing recompilation."""
        rows = self.db.execute(
            "SELECT b.id, b.module_id, b.name, b.kind, m.profile "
            "FROM blocks b JOIN modules m ON b.module_id = m.id "
            "WHERE b.dirty = 1 ORDER BY m.id, b.id").fetchall()
        return [DirtyNode(block_id=r["id"], module_id=r["module_id"],
                          name=r["name"], kind=r["kind"], profile=r["profile"]) for r in rows]

    # ── Record compilation result ──
    def record_compiled(self, block_id: str, output_code: str):
        """Mark a block as compiled with output code."""
        now = time.time()
        self.db.execute(
            "UPDATE blocks SET dirty=0, compiled_at=?, output_code=? WHERE id=?",
            (now, output_code, block_id))
        # If all blocks in module are clean, mark module clean
        row = self.db.execute(
            "SELECT module_id FROM blocks WHERE id=?", (block_id,)).fetchone()
        if row:
            mid = row["module_id"]
            still_dirty = self.db.execute(
                "SELECT COUNT(*) as c FROM blocks WHERE module_id=? AND dirty=1",
                (mid,)).fetchone()["c"]
            if still_dirty == 0:
                self.db.execute("UPDATE modules SET dirty=0, compiled_at=? WHERE id=?", (now, mid))
        self.db.commit()

    # ── Lookup helpers ──
    def get_block(self, block_id: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM blocks WHERE id=?", (block_id,)).fetchone()
        return dict(row) if row else None

    def get_module(self, module_id: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM modules WHERE id=?", (module_id,)).fetchone()
        return dict(row) if row else None

    def get_all_modules(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM modules ORDER BY id").fetchall()]

    def get_blocks_for_module(self, module_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM blocks WHERE module_id=? ORDER BY id", (module_id,)).fetchall()]

    def stats(self) -> dict:
        mods = self.db.execute("SELECT COUNT(*) as c FROM modules").fetchone()["c"]
        blks = self.db.execute("SELECT COUNT(*) as c FROM blocks").fetchone()["c"]
        dirty = self.db.execute("SELECT COUNT(*) as c FROM blocks WHERE dirty=1").fetchone()["c"]
        return {"modules": mods, "blocks": blks, "dirty_blocks": dirty}
