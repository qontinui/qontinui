"""Guard: no debug-log writes into the OS temp dir or the user's home dir.

Core used to scatter ad-hoc ``open(<tempdir>/qontinui_*_debug.log, "a")``
writes across the action executors, HAL, wrappers and multistate glue. They
duplicated (or bypassed) ``qontinui.logging`` and littered the temp dir on
every automation run. This AST guard keeps them from coming back.

Two arms, both scanning ``src/qontinui/**``:

(a) a write-capable file open -- ``open(path, "a"|"w"|"x"|"+"...)`` or a
    ``.open(...)`` / ``.write_text(...)`` / ``.write_bytes(...)`` method call
    -- whose path references a *temp/home source*. ``Name`` path arguments are
    resolved through assignments in the enclosing function(s), then the
    module, so ``p = os.path.join(tempfile.gettempdir(), "x"); open(p, "a")``
    is caught.
(b) any assignment whose value references a temp/home source AND contains a
    ``".log"`` string literal. This catches module-level constants such as
    ``_DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / "x.log"`` even where the
    open() sits behind an attribute or a helper this guard cannot resolve.

A *temp/home source* is: a ``gettempdir()`` call, a ``TEMP``/``TMP``/``TMPDIR``
environment lookup, a string literal rooted at ``/tmp``, or
``expanduser("~...")`` / ``Path.home()``.

Legitimate temp-dir uses (artifact/cache directories, not debug logs) go in
``ALLOWLIST`` as ``"<repo-relative file>:<function>"``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "qontinui"

# "<path relative to src/qontinui>:<enclosing function or <module>>"
# Entries are only for deliberate, non-debug-log uses. Each one needs a reason.
ALLOWLIST: dict[str, str] = {
    # Deliberately empty. The legitimate temp/home users -- checkpointing/
    # checkpoint_service.py (artifact dir), embeddings/cache.py (cache dir),
    # extraction/runtime/playwright/extractor.py (screenshot staging dir) and
    # navigation/path_reliability.py (~/.qontinui persisted statistics) -- only
    # build a directory path or write through a parameter: they do not open a
    # temp/home path for writing in place nor name a ".log" file, so neither arm
    # flags them and they need no entry.
}

_WRITE_MODE_CHARS = set("wax+")
_WRITE_METHODS = {"write_text", "write_bytes"}
# Modules whose open() takes the path first, like the builtin.
_OPEN_MODULES = {"io", "codecs", "gzip", "bz2", "lzma"}
# Logging file sinks: any tainted argument is a hit.
_FILE_SINKS = {
    "FileHandler",
    "WatchedFileHandler",
    "RotatingFileHandler",
    "TimedRotatingFileHandler",
    "basicConfig",
}
_TEMP_ENV_KEYS = {"TEMP", "TMP", "TMPDIR"}
_SCOPE_BARRIERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _str_consts(node: ast.AST) -> list[str]:
    return [
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _walk_scope(nodes: list[ast.stmt]):
    """Yield every node of ``nodes`` without descending into nested scopes."""
    stack = list(reversed(nodes))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, _SCOPE_BARRIERS):
            continue  # a nested scope's assignments are not this scope's
        stack.extend(reversed(list(ast.iter_child_nodes(n))))


def _record(out: dict[str, list[ast.AST]], target: ast.AST, value: ast.AST) -> None:
    if isinstance(target, (ast.Name, ast.Attribute)):
        out.setdefault(ast.unparse(target), []).append(value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(
            target.elts
        ):
            for t, v in zip(target.elts, value.elts, strict=True):
                _record(out, t, v)
        else:
            for t in target.elts:
                _record(out, t, value)


def _collect_assigns(body: list[ast.stmt]) -> dict[str, list[ast.AST]]:
    """Every value assigned to each name (and ``a.b`` target) in one scope."""
    out: dict[str, list[ast.AST]] = {}
    for n in _walk_scope(body):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                _record(out, t, n.value)
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            _record(out, n.target, n.value)
    return out


def _class_self_attrs(node: ast.ClassDef) -> dict[str, list[ast.AST]]:
    """``self.x = ...`` assignments from ANY method of the class."""
    out: dict[str, list[ast.AST]] = {}
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for key, values in _collect_assigns(item.body).items():
                if key.startswith("self."):
                    out.setdefault(key, []).extend(values)
    return out


def _param_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef):
    args = node.args
    positional = args.posonlyargs + args.args
    out: dict[str, list[ast.AST]] = {}
    for arg, default in zip(
        reversed(positional), reversed(args.defaults), strict=False
    ):
        out.setdefault(arg.arg, []).append(default)
    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if kw_default is not None:
            out.setdefault(arg.arg, []).append(kw_default)
    return out


class _Scan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scopes: list[tuple[str, dict[str, list[ast.AST]]]] = [("<module>", {})]
        self.findings: list[tuple[str, int, str]] = []  # (scope, line, why)
        self._resolving: set[str] = set()

    # -- scope bookkeeping -------------------------------------------------
    def _resolve(self, key: str) -> list[ast.AST]:
        for _, assigns in reversed(self.scopes):
            if key in assigns:
                return assigns[key]
        return []

    def _tainted(self, node: ast.AST) -> bool:
        """True when ``node`` (transitively, through resolved names) names a temp/home source."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                name = _call_name(n)
                if name == "gettempdir":
                    return True
                if name == "expanduser" and any(
                    s.startswith("~") for s in _str_consts(n)
                ):
                    return True
                if (
                    name == "home"
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "Path"
                ):
                    return True
                if name in {"getenv", "get"} and any(
                    s in _TEMP_ENV_KEYS for s in _str_consts(n)
                ):
                    return True
            elif isinstance(n, ast.Subscript):
                if any(s in _TEMP_ENV_KEYS for s in _str_consts(n.slice)):
                    return True
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                if n.value == "/tmp" or n.value.startswith("/tmp/"):
                    return True
            if (
                isinstance(n, ast.Attribute)
                and n.attr == "tempdir"
                and isinstance(n.value, ast.Name)
                and n.value.id == "tempfile"
            ):
                return True
            if isinstance(n, (ast.Name, ast.Attribute)):
                key = ast.unparse(n)
                if key in self._resolving:
                    continue
                values = self._resolve(key)
                if values:
                    # Latch guards `p = p + "x"` style cycles during the recursive walk.
                    self._resolving.add(key)
                    try:
                        if any(self._tainted(v) for v in values):
                            return True
                    finally:
                        self._resolving.discard(key)
        return False

    @property
    def _scope_name(self) -> str:
        return self.scopes[-1][0]

    def visit_Module(self, node: ast.Module) -> None:
        self.scopes[0] = ("<module>", _collect_assigns(node.body))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        scope = _collect_assigns(node.body)
        for key, values in _class_self_attrs(node).items():
            scope.setdefault(key, []).extend(values)
        self.scopes.append((node.name, scope))
        self.generic_visit(node)
        self.scopes.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        scope = _param_defaults(node)
        for key, values in _collect_assigns(node.body).items():
            scope.setdefault(key, []).extend(values)
        self.scopes.append((node.name, scope))
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    # -- arm (b): assignments naming a temp/home source plus a ".log" literal
    def _check_assign(self, node: ast.AST, value: ast.AST | None) -> None:
        if value is None:
            return
        if any(".log" in s for s in _str_consts(value)) and self._tainted(value):
            self.findings.append((self._scope_name, node.lineno, "assignment"))

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_assign(node, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assign(node, node.value)
        self.generic_visit(node)

    # -- arm (a): write-capable opens and logging file sinks ---------------
    def visit_Call(self, node: ast.Call) -> None:
        self._check_open(node)
        self.generic_visit(node)

    def _hit(self, node: ast.Call, why: str) -> None:
        self.findings.append((self._scope_name, node.lineno, why))

    def _check_open(self, node: ast.Call) -> None:
        func = node.func
        is_builtin_open = isinstance(func, ast.Name) and func.id == "open"
        is_module_open = (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and isinstance(func.value, ast.Name)
            and func.value.id in _OPEN_MODULES
        )
        if is_builtin_open or is_module_open:
            path = node.args[0] if node.args else _kw(node, "file")
            mode_node = node.args[1] if len(node.args) > 1 else _kw(node, "mode")
            if path is not None and _is_write_mode(mode_node) and self._tainted(path):
                self._hit(node, "open")
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            # Path(...).open(mode)
            mode_node = node.args[0] if node.args else _kw(node, "mode")
            if _is_write_mode(mode_node) and self._tainted(func.value):
                self._hit(node, ".open")
        elif isinstance(func, ast.Attribute) and func.attr in _WRITE_METHODS:
            if self._tainted(func.value):
                self._hit(node, f".{func.attr}")
        elif _call_name(node) in _FILE_SINKS:
            args = [*node.args, *(kw.value for kw in node.keywords)]
            if any(self._tainted(a) for a in args):
                self._hit(node, _call_name(node))


def _kw(node: ast.Call, name: str) -> ast.AST | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_write_mode(mode: ast.AST | None) -> bool:
    if mode is None:
        return False
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return bool(_WRITE_MODE_CHARS & set(mode.value))
    return True  # non-literal mode: be conservative


def scan_source(source: str) -> list[tuple[str, int, str]]:
    scan = _Scan()
    scan.visit(ast.parse(source))
    return scan.findings


def _violations() -> dict[str, list[tuple[str, int, str]]]:
    out: dict[str, list[tuple[str, int, str]]] = {}
    for py in sorted(SRC_ROOT.rglob("*.py")):
        rel = py.relative_to(SRC_ROOT).as_posix()
        hits = [
            (scope, line, why)
            for scope, line, why in scan_source(py.read_text(encoding="utf-8-sig"))
            if f"{rel}:{scope}" not in ALLOWLIST
        ]
        if hits:
            out[rel] = hits
    return out


def test_no_temp_or_home_debug_log_writes() -> None:
    violations = _violations()
    report = "\n".join(
        f"  {rel}: {len(hits)} site(s) at lines {sorted({h[1] for h in hits})}"
        for rel, hits in violations.items()
    )
    total = sum(len(h) for h in violations.values())
    assert not violations, (
        f"{total} temp-dir/home debug-log write site(s) in {len(violations)} file(s). "
        "Use qontinui.logging.get_logger(__name__).debug instead:\n" + report
    )


@pytest.mark.parametrize(
    "source",
    [
        # (a) direct
        "import tempfile, os\ndef f():\n    open(os.path.join(tempfile.gettempdir(), 'a.log'), 'a')\n",
        # (a) via a Name resolved in the enclosing function
        "import tempfile, os\ndef f():\n    p = os.path.join(tempfile.gettempdir(), 'a.txt')\n    with open(p, 'w') as h:\n        h.write('x')\n",
        # (a) Name resolved through a module-level assignment, from a nested closure
        "import tempfile\nBASE = tempfile.gettempdir()\ndef f():\n    def g():\n        open(BASE + '/x', 'a')\n    g()\n",
        # (a) /tmp literal and expanduser
        "def f():\n    open('/tmp/x.txt', 'w')\n",
        "import os\ndef f():\n    open(os.path.expanduser('~/x.txt'), mode='a')\n",
        # (a) TEMP environment lookup
        "import os\ndef f():\n    open(os.environ['TEMP'] + '/x', 'w')\n",
        # (a) pathlib write
        "import tempfile\nfrom pathlib import Path\ndef f():\n    (Path(tempfile.gettempdir()) / 'x').write_text('y')\n",
        # (a) io.open / logging file sinks / self attribute / tuple unpack / default arg
        "import io, tempfile\ndef f():\n    io.open(tempfile.gettempdir() + '/x', 'a')\n",
        "import logging, tempfile\ndef f():\n    logging.FileHandler(tempfile.gettempdir() + '/x')\n",
        "import logging, tempfile\ndef f():\n    logging.basicConfig(filename=tempfile.gettempdir() + '/x')\n",
        "import tempfile\nclass C:\n    def __init__(self):\n        self.p = tempfile.gettempdir() + '/x'\n    def w(self):\n        open(self.p, 'a')\n",
        "import tempfile\ndef f():\n    d, n = tempfile.gettempdir(), 'x'\n    open(d + n, 'w')\n",
        "import tempfile\ndef f(p=tempfile.gettempdir()):\n    open(p, 'w')\n",
        "import tempfile\ndef f():\n    open(tempfile.tempdir + '/x', 'w')\n",
        # (a) reassignment / branch: ANY tainted value counts
        "import tempfile\ndef f(c):\n    if c:\n        p = tempfile.gettempdir() + '/x'\n    else:\n        p = 'y'\n    open(p, 'a')\n",
        # (b) module constant, open() behind an unresolvable attribute
        "import tempfile\nfrom pathlib import Path\n_DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / 'x.log'\n",
    ],
)
def test_guard_detects_known_bad_shapes(source: str) -> None:
    assert scan_source(source), source


@pytest.mark.parametrize(
    "source",
    [
        # reads are fine
        "import tempfile, os\ndef f():\n    open(os.path.join(tempfile.gettempdir(), 'a.log'))\n",
        "import tempfile, os\ndef f():\n    open(os.path.join(tempfile.gettempdir(), 'a.log'), 'r')\n",
        # a non-temp write is fine
        "def f(p):\n    open(p, 'w')\n",
        # a temp dir with no .log literal and no write is fine
        "import tempfile\ndef f():\n    return tempfile.gettempdir()\n",
        # a parameter that merely shares a name with another function's tainted local
        "import tempfile\ndef a():\n    p = tempfile.gettempdir() + '/x.txt'\n    return p\ndef b(p):\n    open(p, 'w')\n",
        # self-referential assignment must not loop forever
        "def f():\n    p = p + 'x'\n    open(p, 'w')\n",
    ],
)
def test_guard_ignores_known_good_shapes(source: str) -> None:
    assert not scan_source(source), source


def test_allowlist_entries_are_live() -> None:
    """A stale allowlist entry would silently widen the guard."""
    live: set[str] = set()
    for py in SRC_ROOT.rglob("*.py"):
        rel = py.relative_to(SRC_ROOT).as_posix()
        for scope, _line, _why in scan_source(py.read_text(encoding="utf-8-sig")):
            live.add(f"{rel}:{scope}")
    stale = sorted(set(ALLOWLIST) - live)
    assert not stale, f"allowlist entries match no flagged site: {stale}"
