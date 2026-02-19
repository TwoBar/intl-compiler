"""INTL Parser — Lark grammar → typed AST dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from lark import Lark, Transformer, v_args, Token

GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"

# ── AST Node base ──
@dataclass
class Node:
    line: int = 0
    col: int = 0

# ── Expressions ──
@dataclass
class Ref(Node):
    parts: list[str] = field(default_factory=list)
@dataclass
class Literal(Node):
    value: object = None
    kind: str = ""  # string/number/duration/bool/null
@dataclass
class NowCall(Node): pass
@dataclass
class FuncCall(Node):
    name: Ref = None
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
@dataclass
class BinOp(Node):
    op: str = ""
    left: object = None
    right: object = None
@dataclass
class UnaryOp(Node):
    op: str = ""
    operand: object = None

# ── Module ──
@dataclass
class Module(Node):
    name: str = ""
    id: str = ""
    target: str = ""
    profile: str = ""
    namespace: str = ""
    requires: list[str] = field(default_factory=list)
    version: str = ""
    blocks: list = field(default_factory=list)

# ── Function / Pipeline fields ──
@dataclass
class FunctionBlock(Node):
    name: str = ""
    id: str = ""
    intent: str = ""
    preconditions: list = field(default_factory=list)
    postconditions: list = field(default_factory=list)
    reads: list[str] = field(default_factory=list)
    mutates: list[str] = field(default_factory=list)
    emits: list[str] = field(default_factory=list)
    observable: bool = False
    confidence: float = 1.0
    timeout: str = ""
    body: list = field(default_factory=list)
    kind: str = "function"  # or "pipeline"

# ── Type ──
@dataclass
class TypeBlock(Node):
    name: str = ""
    id: str = ""
    fields: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
@dataclass
class FieldDecl(Node):
    name: str = ""
    type_expr: str = ""
    not_null: bool = False
    default: object = None

# ── Patch ──
@dataclass
class PatchBlock(Node):
    target_kind: str = ""  # function/pipeline
    name: str = ""
    id: str = ""
    intent: str = ""
    insert_preconditions: list[str] = field(default_factory=list)
    remove_preconditions: list[str] = field(default_factory=list)
    ops: list = field(default_factory=list)
@dataclass
class PatchOp(Node):
    action: str = ""  # insert/replace/remove
    position: str = ""
    match_text: str = ""
    body: list = field(default_factory=list)

# ── Body statements ──
@dataclass
class Query(Node):
    target: str = ""; table: str = ""; where: object = None; limit: object = None
@dataclass
class Persist(Node):
    table: str = ""; action: str = ""; fields: dict = field(default_factory=dict); where: object = None
@dataclass
class Assign(Node):
    target: str = ""; value: object = None
@dataclass
class Call(Node):
    target: str = ""; name: Ref = None; args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict); is_await: bool = False
@dataclass
class If(Node):
    condition: object = None; body: list = field(default_factory=list)
    else_body: list = field(default_factory=list)
@dataclass
class Fail(Node):
    error: str = ""; message: str = ""
@dataclass
class Return(Node):
    value: object = None
@dataclass
class Loop(Node):
    collection: Ref = None; var: str = ""; body: list = field(default_factory=list)
@dataclass
class Emit(Node):
    event: str = ""; fields: dict = field(default_factory=dict)
@dataclass
class Block(Node):
    kind: str = ""; body: list = field(default_factory=list)  # sequence/parallel/transaction/lock/subscribe
    name: str = ""
@dataclass
class Fallback(Node):
    primary: list = field(default_factory=list); fallback: list = field(default_factory=list)
@dataclass
class CacheGet(Node):
    key: object = None; alias: str = ""
@dataclass
class CacheSet(Node):
    key: object = None; value: object = None; ttl: str = ""
@dataclass
class Validate(Node):
    names: list[str] = field(default_factory=list); schema: str = ""
@dataclass
class Paginate(Node):
    target: str = ""; table: str = ""; page: object = None; size: object = None
@dataclass
class Aggregate(Node):
    target: str = ""; table: str = ""; by: str = ""; compute: str = ""
@dataclass
class Transform(Node):
    target: str = ""; expr: object = None; using: str = ""


def _pos(meta_or_token):
    """Extract line/col from Lark meta or token."""
    if hasattr(meta_or_token, 'line'):
        return meta_or_token.line, meta_or_token.column
    return 0, 0


def _strip_quotes(s):
    if isinstance(s, Token):
        s = str(s)
    return s.strip('"')


# ── Transformer: Lark tree → AST ──
@v_args(meta=True)
class ASTBuilder(Transformer):
    # --- Expressions ---
    def expression(self, meta, items): return items[0]
    def or_expr(self, meta, items):
        r = items[0]
        for i in range(1, len(items)):
            r = BinOp(op="OR", left=r, right=items[i], *_pos(meta))
        return r
    def and_expr(self, meta, items):
        r = items[0]
        for i in range(1, len(items)):
            r = BinOp(op="AND", left=r, right=items[i], *_pos(meta))
        return r
    def not_op(self, meta, items): return UnaryOp(op="NOT", operand=items[0], *_pos(meta))
    def comparison(self, meta, items):
        if len(items) == 1: return items[0]
        return BinOp(op=items[1], left=items[0], right=items[2], *_pos(meta))
    def addition(self, meta, items):
        r = items[0]
        for i in range(1, len(items), 2):
            r = BinOp(op=str(items[i]), left=r, right=items[i+1], *_pos(meta))
        return r
    def multiplication(self, meta, items):
        r = items[0]
        for i in range(1, len(items), 2):
            r = BinOp(op=str(items[i]), left=r, right=items[i+1], *_pos(meta))
        return r

    # comp_op
    def is_not(self, meta, items): return "IS NOT"
    def is_op(self, meta, items): return "IS"
    def eq(self, meta, items): return "=="
    def neq(self, meta, items): return "!="
    def gte(self, meta, items): return ">="
    def lte(self, meta, items): return "<="
    def gt(self, meta, items): return ">"
    def lt(self, meta, items): return "<"

    # atoms
    def string_literal(self, meta, items): return Literal(value=_strip_quotes(items[0]), kind="string", *_pos(meta))
    def number_literal(self, meta, items): return Literal(value=float(str(items[0])), kind="number", *_pos(meta))
    def duration_literal(self, meta, items): return Literal(value=str(items[0]), kind="duration", *_pos(meta))
    def true_literal(self, meta, items): return Literal(value=True, kind="bool", *_pos(meta))
    def false_literal(self, meta, items): return Literal(value=False, kind="bool", *_pos(meta))
    def null_literal(self, meta, items): return Literal(value=None, kind="null", *_pos(meta))
    def now_call(self, meta, items): return NowCall(*_pos(meta))
    def ref(self, meta, items): return items[0]
    def paren_expr(self, meta, items): return items[0]
    def func_call(self, meta, items):
        name = items[0]
        args, kwargs = self._unpack_call_args(items[1] if len(items) > 1 else ([], {}))
        return FuncCall(name=name, args=args, kwargs=kwargs, *_pos(meta))

    def dotted_name(self, meta, items):
        return Ref(parts=[str(t) for t in items], *_pos(meta))
    def id_tag(self, meta, items): return str(items[0])
    def name_list(self, meta, items): return [str(t) for t in items]
    def field_pair(self, meta, items): return (str(items[0]), items[1])
    def field_list(self, meta, items): return dict(items)
    def arg_list(self, meta, items): return list(items)

    # call_args
    def call_kwargs(self, meta, items): return ([], dict(items))
    def call_positional(self, meta, items): return (items[0] if isinstance(items[0], list) else items, {})
    def call_empty(self, meta, items): return ([], {})

    def _unpack_call_args(self, ca):
        if isinstance(ca, tuple): return ca
        return ([], {})

    # --- Module ---
    def module_decl(self, meta, items):
        name, mid = str(items[0]), items[1]
        m = Module(name=name, id=mid, *_pos(meta))
        for f in items[2:]:
            if isinstance(f, tuple):
                setattr(m, f[0], f[1])
        return m
    def module_field(self, meta, items): return items[0]
    def target_field(self, meta, items): return ("target", _strip_quotes(items[0]))
    def profile_field(self, meta, items): return ("profile", str(items[0]))
    def namespace_field(self, meta, items): return ("namespace", _strip_quotes(items[0]))
    def requires_field(self, meta, items): return ("requires", items[0])
    def version_field(self, meta, items): return ("version", _strip_quotes(items[0]))

    # --- Function / Pipeline ---
    def function_block(self, meta, items):
        return self._build_func(meta, items, "function")
    def pipeline_block(self, meta, items):
        return self._build_func(meta, items, "pipeline")

    def _build_func(self, meta, items, kind):
        name, fid = str(items[0]), items[1]
        # items ends with closing name + id_tag
        fb = FunctionBlock(name=name, id=fid, kind=kind, *_pos(meta))
        for it in items[2:]:
            if isinstance(it, tuple):
                k, v = it
                if k == "intent": fb.intent = v
                elif k == "precondition": fb.preconditions.append(v)
                elif k == "postcondition": fb.postconditions.append(v)
                elif k == "reads": fb.reads.append(v)
                elif k == "mutates": fb.mutates.append(v)
                elif k == "emits": fb.emits.append(v)
                elif k == "observable": fb.observable = True
                elif k == "confidence": fb.confidence = v
                elif k == "timeout": fb.timeout = v
            elif isinstance(it, Node):
                fb.body.append(it)
        return fb

    def function_field(self, meta, items): return items[0]
    def intent_field(self, meta, items): return ("intent", _strip_quotes(items[0]))
    def precondition_field(self, meta, items): return ("precondition", items[0])
    def postcondition_field(self, meta, items): return ("postcondition", items[0])
    def reads_field(self, meta, items): return ("reads", str(items[0]))
    def mutates_field(self, meta, items): return ("mutates", str(items[0]))
    def emits_field(self, meta, items): return ("emits", str(items[0]))
    def observable_field(self, meta, items): return ("observable", True)
    def confidence_field(self, meta, items): return ("confidence", float(str(items[0])))
    def timeout_field(self, meta, items): return ("timeout", str(items[0]))

    # --- Type ---
    def type_block(self, meta, items):
        name, tid = str(items[0]), items[1]
        tb = TypeBlock(name=name, id=tid, *_pos(meta))
        for it in items[2:]:
            if isinstance(it, FieldDecl): tb.fields.append(it)
            elif isinstance(it, Node): tb.constraints.append(it)
            elif isinstance(it, str): pass  # closing name/id
        return tb
    def type_field(self, meta, items): return items[0]
    def field_decl(self, meta, items):
        fd = FieldDecl(name=str(items[0]), type_expr=items[1], *_pos(meta))
        for mod in items[2:]:
            if mod == "NOT_NULL": fd.not_null = True
            else: fd.default = mod
        return fd
    def constraint_decl(self, meta, items): return items[0]
    def not_null(self, meta, items): return "NOT_NULL"
    def default_val(self, meta, items): return items[0]
    def simple_type(self, meta, items): return str(items[0])
    def parameterized_type(self, meta, items): return f"{items[0]}({items[1]})"
    def list_type(self, meta, items): return f"{items[0]}[]"
    def optional_type(self, meta, items): return f"{items[0]}?"
    def map_type(self, meta, items): return f"Map<{items[0]},{items[1]}>"

    # --- Patch ---
    def patch_block(self, meta, items):
        kind = items[0]
        name, pid = str(items[1]), items[2]
        pb = PatchBlock(target_kind=kind, name=name, id=pid, *_pos(meta))
        for it in items[3:]:
            if isinstance(it, tuple):
                k, v = it
                if k == "intent": pb.intent = v
                elif k == "insert_pre": pb.insert_preconditions.append(v)
                elif k == "remove_pre": pb.remove_preconditions.append(v)
            elif isinstance(it, PatchOp): pb.ops.append(it)
        return pb
    def patch_function(self, meta, items): return "function"
    def patch_pipeline(self, meta, items): return "pipeline"
    def patch_field(self, meta, items): return items[0]
    def insert_precondition(self, meta, items): return ("insert_pre", _strip_quotes(items[0]))
    def remove_precondition(self, meta, items): return ("remove_pre", _strip_quotes(items[0]))
    def patch_insert(self, meta, items): return PatchOp(action="insert", position=items[0][0], match_text=items[0][1], body=list(items[1:]), *_pos(meta))
    def patch_replace(self, meta, items): return PatchOp(action="replace", position=items[0][0], match_text=items[0][1], body=list(items[1:]), *_pos(meta))
    def patch_remove(self, meta, items): return PatchOp(action="remove", position=items[0][0], match_text=items[0][1], *_pos(meta))
    def after_line(self, meta, items): return ("after", _strip_quotes(items[0]))
    def before_line(self, meta, items): return ("before", _strip_quotes(items[0]))
    def at_line(self, meta, items): return ("at", _strip_quotes(items[0]))

    # --- Body stmts ---
    def body_stmt(self, meta, items): return items[0]
    def query_stmt(self, meta, items):
        lim = items[3] if len(items) > 3 else None
        return Query(target=str(items[0]), table=str(items[1]), where=items[2], limit=lim, *_pos(meta))
    def persist_insert_stmt(self, meta, items):
        return Persist(table=str(items[0]), action="INSERT", fields=items[1], *_pos(meta))
    def persist_update_stmt(self, meta, items):
        return Persist(table=str(items[0]), action="UPDATE", fields=items[1], where=items[2], *_pos(meta))
    def persist_delete_stmt(self, meta, items):
        return Persist(table=str(items[0]), action="DELETE", where=items[1], *_pos(meta))
    def assign_stmt(self, meta, items):
        return Assign(target=str(items[0]), value=items[1], *_pos(meta))

    def call_assign_stmt(self, meta, items):
        args, kwargs = self._unpack_call_args(items[2] if len(items) > 2 else ([], {}))
        return Call(target=str(items[0]), name=items[1], args=args, kwargs=kwargs, *_pos(meta))
    def call_bare_stmt(self, meta, items):
        args, kwargs = self._unpack_call_args(items[1] if len(items) > 1 else ([], {}))
        return Call(name=items[0], args=args, kwargs=kwargs, *_pos(meta))
    def await_assign_stmt(self, meta, items):
        args, kwargs = self._unpack_call_args(items[2] if len(items) > 2 else ([], {}))
        return Call(target=str(items[0]), name=items[1], args=args, kwargs=kwargs, is_await=True, *_pos(meta))
    def await_bare_stmt(self, meta, items):
        args, kwargs = self._unpack_call_args(items[1] if len(items) > 1 else ([], {}))
        return Call(name=items[0], args=args, kwargs=kwargs, is_await=True, *_pos(meta))

    def if_inline(self, meta, items):
        return If(condition=items[0], body=[items[1]], *_pos(meta))
    def if_simple(self, meta, items):
        return If(condition=items[0], body=list(items[1:]), *_pos(meta))
    def if_else(self, meta, items):
        cond = items[0]
        # find split — body items are Nodes, need to split at midpoint
        rest = items[1:]
        mid = len(rest) // 2  # heuristic; we'll track properly
        body, els = [], []
        in_else = False
        for it in rest:
            if not in_else:
                body.append(it)
            else:
                els.append(it)
            # if_else rule: body_stmt+ ELSE body_stmt+, items are interleaved
        # Actually Lark gives us all body stmts in order; the rule ensures split
        # We need a marker. Let's split by counting: items[0]=cond, then body+, then else+
        # Lark flattens it. We'll use a different approach below.
        return If(condition=cond, body=body, else_body=els, *_pos(meta))

    def inline_fail(self, meta, items): return Fail(error=str(items[0]), message=_strip_quotes(items[1]), *_pos(meta))
    def inline_return(self, meta, items): return Return(value=items[0], *_pos(meta))
    def inline_call(self, meta, items):
        args, kwargs = self._unpack_call_args(items[1] if len(items) > 1 else ([], {}))
        return Call(name=items[0], args=args, kwargs=kwargs, *_pos(meta))

    def fail_stmt(self, meta, items): return Fail(error=str(items[0]), message=_strip_quotes(items[1]), *_pos(meta))
    def return_stmt(self, meta, items): return Return(value=items[0], *_pos(meta))

    def loop_stmt(self, meta, items):
        return Loop(collection=items[0], var=str(items[1]), body=list(items[2:]), *_pos(meta))
    def emit_stmt(self, meta, items):
        return Emit(event=str(items[0]), fields=items[1], *_pos(meta))

    def sequence_block(self, meta, items): return Block(kind="sequence", body=list(items), *_pos(meta))
    def parallel_block(self, meta, items): return Block(kind="parallel", body=list(items), *_pos(meta))
    def transaction_block(self, meta, items): return Block(kind="transaction", body=list(items), *_pos(meta))
    def lock_block(self, meta, items): return Block(kind="lock", name=str(items[0]), body=list(items[1:]), *_pos(meta))
    def subscribe_block(self, meta, items): return Block(kind="subscribe", name=str(items[0]), body=list(items[1:]), *_pos(meta))
    def fallback_block(self, meta, items):
        # items = primary stmts + fallback stmts; split by midpoint
        # Lark rule: body_stmt+ FALLBACK : body_stmt+
        # All items are body stmts. We mark split via sentinel.
        return Fallback(primary=[], fallback=[], *_pos(meta))  # TODO: proper split

    def cache_get_stmt(self, meta, items): return CacheGet(key=items[0], alias=str(items[1]), *_pos(meta))
    def cache_set_stmt(self, meta, items): return CacheSet(key=items[0], value=items[1], ttl=str(items[2]), *_pos(meta))
    def validate_stmt(self, meta, items): return Validate(names=items[0], schema=str(items[1]), *_pos(meta))
    def paginate_stmt(self, meta, items):
        return Paginate(target=str(items[0]), table=str(items[1]), page=items[2], size=items[3], *_pos(meta))
    def aggregate_stmt(self, meta, items):
        return Aggregate(target=str(items[0]), table=str(items[1]), by=str(items[2]), compute=str(items[3]), *_pos(meta))
    def transform_stmt(self, meta, items):
        return Transform(target=str(items[0]), expr=items[1], using=str(items[2]), *_pos(meta))

    # --- Top level ---
    def block(self, meta, items): return items[0]
    def start(self, meta, items):
        mod = items[0]
        mod.blocks = list(items[1:])
        return mod


# ── Public API ──
_parser: Lark | None = None

def get_parser() -> Lark:
    global _parser
    if _parser is None:
        _parser = Lark(
            GRAMMAR_PATH.read_text(),
            parser="earley",
            propagate_positions=True,
        )
    return _parser


def parse(source: str) -> Module:
    """Parse INTL source → Module AST. Raises lark.exceptions on error."""
    tree = get_parser().parse(source)
    return ASTBuilder().transform(tree)


class ParseError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line, self.col = line, col
        super().__init__(f"[line {line}, col {col}] {message}")


def parse_safe(source: str) -> tuple[Module | None, ParseError | None]:
    """Parse with error wrapping. Returns (module, None) or (None, error)."""
    try:
        return parse(source), None
    except Exception as e:
        line = getattr(e, 'line', 0) or 0
        col = getattr(e, 'column', 0) or getattr(e, 'col', 0) or 0
        return None, ParseError(str(e), line, col)
