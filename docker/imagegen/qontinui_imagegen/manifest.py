"""Licence manifest, pinned hashes and recipes for the local image provider.

``docker/imagegen/manifest.yaml`` is decision record D2
(``decision_record/image-generation-weights-are-permissively-licensed``, a5ec7670 **v2**)
in data: every component the image provider loads -- weights, inference libraries,
vendored model code -- carries its SPDX licence beside its pinned sha256s, so admission is
checked by a test instead of remembered. This module mirrors the v2 record: its admitted
licence ids, its single admitted ``WITH`` exception, its platform class and its rules for
reading a licence (dist-info licence files AND bundled native libraries; statically linked
dependencies from the upstream lockfile; unknown-licence code removed through a derived
file). ``docker/imagegen/recipes/*.yaml`` are the versioned generation recipes, and
:func:`recipe_hash` is the provenance hash every generated asset records.

The checker is strict on purpose (plan ``2026-10-13-image-generation-and-visual-judgment``,
Phase 2):

* every ``spdx`` expression -- a component's, a bundled native library's, and every
  statically linked dependency's -- must evaluate inside the admitted ids
  (:data:`ADMITTED_SPDX`, :data:`ADMITTED_EXCEPTIONS`), which live in this reviewed code,
  never in the manifest they judge; a NEW id turns the test red until someone reads it and
  amends the record;
* the components D2 excludes by name are refused even when their licence is mislabelled,
  and so is any vendored or derived file whose upstream is one of them;
* every file is pinned by sha256 and size (one sha256, one size), every revision and
  commit by a full commit id, and every licence's evidence by the pin it was read at;
* a wheel records the licence files and native libraries it ships (``bundled``), and
  every licence file it ships must be one whose bytes were read; a platform runtime a
  wheel bundles is pinned under ``platform``, outside the admitted families;
* vendored model code on disk must match its pinned bytes, and nothing unlisted may sit
  beside it;
* no recipe may enable a network fetch: the runtime contract is ``HF_HUB_OFFLINE=1`` with
  ``local_files_only=True``, no ``trust_remote_code``, no ``custom_pipeline``, and
  ``torch.load`` restricted to ``weights_only=True``.

THE RECIPE HASH. :func:`recipe_hash` is the sha256 of the UTF-8 bytes of
``json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)`` (:func:`canonical_json`), where ``document`` is
``{"recipe": <the recipe>, "manifest": {"runners": ..., "components": ...}}`` holding the
runners and components the recipe references. It identifies what generates the pixels,
so it leaves out the recipe's ``description`` and every licence-only component field
(:data:`HASH_EXCLUDED_COMPONENT_FIELDS`: ``spdx``, ``licence_url``, ``licence_evidence``,
``licence_read_at``, ``notes``, ``static_dependencies``, ``bundled``) -- a licence re-read or
relabel changes no pixels, and licence provenance travels separately in the asset's
sidecar (Phase 11). This is NOT RFC 8785 (JCS): keys sort by Python ``str`` order (code
points; JCS sorts UTF-16 code units, which differs only outside the BMP), numbers use
Python's ``repr`` (``4.0`` stays ``4.0``; JCS writes ``4``), and string escaping follows
Python's ``json`` module. A verifier in another language reproduces these rules, not JCS.

Run ``python -m qontinui_imagegen.manifest`` from ``docker/imagegen``: it prints every
violation and exits non-zero when there is one. ``--hashes`` prints each recipe's hash and
``--footprint`` the install size counting each sha256 once (see :func:`install_footprint`).

The module is plain Python 3.10 (the llama-swap image's interpreter) plus PyYAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeGuard

import yaml

IMAGEGEN_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = IMAGEGEN_ROOT / "manifest.yaml"
RECIPES_DIR = IMAGEGEN_ROOT / "recipes"

DECISION_RECORD_NAME = "image-generation-weights-are-permissively-licensed"
DECISION_RECORD_ID = "a5ec7670-843b-4556-ab52-7f5f7662ee09"
#: The version of the decision record this module mirrors.
DECISION_RECORD_VERSION = 2

#: The licence ids decision record a5ec7670 v2 admits: v1's Apache-2.0, MIT and
#: BSD-family ids (plan Phase 2 step 4) and v2's amendment for licences bundled inside
#: admitted packages (BSL-1.0, Zlib, libpng-2.0 -- not the older ``Libpng`` -- and IJG).
#: Widening this set is an amendment to the record, never an edit to the manifest.
ADMITTED_SPDX: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "BSL-1.0",
        "Zlib",
        "libpng-2.0",
        "IJG",
    }
)

#: The only ``WITH`` exceptions a5ec7670 v2 admits, keyed by the licence they may follow.
#: Any other exception id (``Commons-Clause``, a ``LicenseRef-*`` rider) is a blocker until
#: it is read and admitted there: an exception can withdraw rights the base licence grants.
ADMITTED_EXCEPTIONS: dict[str, frozenset[str]] = {"Apache-2.0": frozenset({"LLVM-exception"})}

#: Components D2 excludes by name, refused even when an entry claims an admitted licence.
#: Each pattern is matched against a NORMALISED name (lowercase, every run of other
#: characters turned into one space -- see :func:`normalise_name`), so ``FLUX.1-dev``,
#: ``flux_1_dev`` and ``flux-dev`` all read ``flux 1 dev`` or ``flux dev``. The Stability
#: Community Licence has no name pattern on purpose: Stability also publishes MIT-licensed
#: artifacts, so the licence check is the only honest test there.
EXCLUDED_COMPONENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bflux(?: [12])?(?: [a-z0-9]+)*? (?:dev|kontext|redux)\b"),
        "the FLUX [dev] family (FLUX.1 [dev], FLUX.2 [dev], Kontext, Redux) is under the "
        "FLUX [dev] non-commercial licence",
    ),
    (
        re.compile(r"\bqwen image 2 1\b"),
        "Qwen-Image-2.1 is under the Qwen Research License (non-commercial)",
    ),
    (re.compile(r"\brmbg 2\b"), "RMBG-2.0 is under CC BY-NC 4.0"),
    (
        re.compile(r"\bomni ?svg\b"),
        "OmniSVG is fine-tuned from a non-commercial base (Qwen2.5-VL-3B)",
    ),
    (
        re.compile(r"hunyuan"),
        "the Tencent Hunyuan community licence excludes the EU, the UK and South Korea",
    ),
    (re.compile(r"\bkrea 2\b"), "Krea 2's community licence is seat-capped"),
    (
        re.compile(r"^(?:comfy ?ui|comfy org comfy ?ui|comfyanonymous comfy ?ui)$"),
        "ComfyUI is GPL-3.0 (plan section 'Applying D2 to D1.1')",
    ),
)

#: The runtime contract the facade applies to every load (plan Phase 2 step 4, Phase 10).
REQUIRED_RUNTIME: dict[str, dict[str, Any]] = {
    "env": {"HF_HUB_OFFLINE": "1"},
    "from_pretrained": {"local_files_only": True, "trust_remote_code": False},
    "torch_load": {"weights_only": True},
}

#: Component fields left out of :func:`recipe_hash`: licence provenance, not what is
#: loaded. A licence re-read or relabel changes no pixels; the asset sidecar carries it.
HASH_EXCLUDED_COMPONENT_FIELDS: frozenset[str] = frozenset(
    {
        "spdx",
        "licence_url",
        "licence_evidence",
        "licence_read_at",
        "notes",
        "static_dependencies",
        "bundled",
    }
)
#: Recipe fields left out of :func:`recipe_hash` (prose only).
HASH_EXCLUDED_RECIPE_FIELDS: frozenset[str] = frozenset({"description"})

POST_OPS: frozenset[str] = frozenset({"upscale", "matte", "vectorize"})
SEED_BASES: frozenset[str] = frozenset({"request_or_random"})
SEED_PER_CANDIDATE: frozenset[str] = frozenset({"add_index"})

#: Keys a recipe may never carry: each is a Hugging Face Hub loader argument that fetches,
#: or (diffusers' ``custom_pipeline`` / ``custom_revision``) fetches and EXECUTES, code or
#: weights; a recipe names manifest ids only.
HUB_LOADER_KEYS: frozenset[str] = frozenset(
    {
        "repo_id",
        "revision",
        "pretrained_model_name_or_path",
        "token",
        "use_auth_token",
        "cache_dir",
        "force_download",
        "resume_download",
        "proxies",
        "mirror",
        "endpoint",
        "custom_pipeline",
        "custom_revision",
    }
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "decision_record",
        "target",
        "runtime",
        "platform",
        "scope",
        "vendor_dir",
        "components",
        "runners",
    }
)
_COMMON_COMPONENT_KEYS = frozenset(
    {"id", "kind", "spdx", "licence_url", "licence_evidence", "licence_read_at", "files", "notes"}
)
_KIND_KEYS: dict[str, frozenset[str]] = {
    "model": frozenset(
        {"source", "role", "runner", "scheduler_class", "size_multiple", "architecture"}
    ),
    "library": frozenset({"source", "bundled", "static_dependencies"}),
    "vendored_code": frozenset(),
}
_SOURCE_TYPES: dict[str, frozenset[str]] = {
    "model": frozenset({"huggingface", "github_release", "derived"}),
    "library": frozenset({"pypi", "git"}),
}
_SOURCE_KEYS: dict[str, frozenset[str]] = {
    "huggingface": frozenset({"type", "repo", "revision"}),
    "github_release": frozenset({"type", "repo", "tag", "commit"}),
    "derived": frozenset({"type", "from"}),
    "pypi": frozenset({"type", "package", "version", "index", "git"}),
    "git": frozenset({"type", "repo", "commit"}),
}
_UPSTREAM_KEYS: dict[str, frozenset[str]] = {
    "huggingface": frozenset({"type", "repo", "revision", "path", "sha256"}),
    "github": frozenset({"type", "repo", "commit", "path", "sha256"}),
    "manifest": frozenset({"type", "component", "sha256"}),
}
_PLATFORM_KEYS = frozenset({"name", "version", "source", "member", "sha256", "bytes", "licence"})
_RECIPE_REQUIRED = frozenset(
    {"id", "version", "pipeline", "models", "parameters", "post", "seed_policy"}
)
_RECIPE_KEYS = _RECIPE_REQUIRED | {"description"}
_POST_STEP_KEYS = frozenset({"op", "runner", "models", "parameters"})

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMPONENT_ID = re.compile(r"[a-z0-9][a-z0-9.+-]*")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_NETWORK = re.compile(r"\b(?:https?|hf|s3|gs|ftp)://|\b(?:huggingface\.co|hf\.co)/", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    """One reason the manifest or a recipe is refused.

    ``code`` is stable and machine-readable; ``where`` locates the offending node;
    ``detail`` says what is wrong in words.
    """

    code: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code} at {self.where}: {self.detail}"


@dataclass(frozen=True)
class Footprint:
    """Install size of a set of components, each distinct sha256 counted once.

    ``shared`` maps each sha256 listed by more than one selected component to those
    components' ids (the Qwen2.5-VL text encoder Qwen-Image-2512 and Qwen-Image-Edit-2511
    share is one such file).
    """

    unique_bytes: int
    unique_files: int
    listed_bytes: int
    listed_files: int
    shared: dict[str, tuple[str, ...]]


class SpdxError(ValueError):
    """An SPDX licence expression that does not parse."""


# --- YAML ------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that keeps timestamps as strings and refuses duplicate mapping keys.

    PyYAML silently keeps the last of two equal keys, which would let a second ``spdx:``
    line override the first unseen; and it turns ``2026-10-13T07:40:11Z`` into a datetime
    that canonical JSON cannot carry.
    """


def _construct_mapping(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_yaml(path: Path) -> Any:
    """Parse one YAML document with :class:`_StrictLoader`."""
    with open(path, encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_StrictLoader)  # nosec B506 - a SafeLoader subclass


def load_recipes(recipes_dir: Path) -> dict[str, Any]:
    """Every ``*.yaml`` recipe in ``recipes_dir``, keyed by file stem."""
    return {path.stem: load_yaml(path) for path in sorted(recipes_dir.glob("*.yaml"))}


# --- SPDX ------------------------------------------------------------------------------

_SPDX_TOKEN = re.compile(r"\(|\)|[A-Za-z0-9][A-Za-z0-9.:-]*\+?")
_SPDX_OPERATORS = frozenset({"AND", "OR", "WITH"})
#: A parsed expression: ("ID", id) | ("WITH", ("ID", id), exception) | ("AND" | "OR", [nodes]).
_Node = tuple[Any, ...]


def _tokenize_spdx(expression: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _SPDX_TOKEN.match(expression, position)
        if match is None:
            raise SpdxError(f"unexpected character {expression[position]!r} at {position}")
        tokens.append(match.group(0))
        position = match.end()
    if not tokens:
        raise SpdxError("empty licence expression")
    return tokens


class _SpdxParser:
    """Recursive-descent parser for SPDX expressions: OR binds loosest, then AND, then WITH."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> _Node:
        node = self._or()
        if self._peek() is not None:
            raise SpdxError(f"unexpected {self._peek()!r} after a complete expression")
        return node

    def _peek(self) -> str | None:
        return self._tokens[self._index] if self._index < len(self._tokens) else None

    def _take(self) -> str | None:
        token = self._peek()
        self._index += 1
        return token

    def _or(self) -> _Node:
        branches = [self._and()]
        while self._peek() == "OR":
            self._take()
            branches.append(self._and())
        return branches[0] if len(branches) == 1 else ("OR", branches)

    def _and(self) -> _Node:
        terms = [self._term()]
        while self._peek() == "AND":
            self._take()
            terms.append(self._term())
        return terms[0] if len(terms) == 1 else ("AND", terms)

    def _term(self) -> _Node:
        token = self._take()
        if token is None:
            raise SpdxError("the expression ends where a licence was expected")
        if token == "(":
            inner = self._or()
            if self._take() != ")":
                raise SpdxError("unbalanced parenthesis")
            return inner
        if token == ")" or token in _SPDX_OPERATORS:
            raise SpdxError(f"unexpected {token!r} where a licence was expected")
        leaf: _Node = ("ID", token)
        if self._peek() == "WITH":
            self._take()
            exception = self._take()
            if exception is None or exception in ("(", ")") or exception in _SPDX_OPERATORS:
                raise SpdxError("WITH must be followed by an exception identifier")
            leaf = ("WITH", leaf, exception)
        return leaf


def _base_id(identifier: str) -> str:
    return identifier[:-1] if identifier.endswith("+") else identifier


def _blockers(node: _Node) -> list[str]:
    kind = node[0]
    if kind == "ID":
        return [] if _base_id(node[1]) in ADMITTED_SPDX else [node[1]]
    if kind == "WITH":
        base = _base_id(node[1][1])
        if base not in ADMITTED_SPDX:
            return [node[1][1]]
        # An exception can withdraw rights the base licence grants, so only the
        # exceptions a5ec7670 v2 names, after the licence it names them for, pass.
        if node[2] not in ADMITTED_EXCEPTIONS.get(base, frozenset()):
            return [f"{node[1][1]} WITH {node[2]}"]
        return []
    branches = [_blockers(child) for child in node[1]]
    if kind == "OR" and any(not blocked for blocked in branches):
        return []
    return [identifier for blocked in branches for identifier in blocked]


def spdx_blockers(expression: str) -> list[str]:
    """What keeps ``expression`` out of the admitted ids; ``[]`` means admitted.

    ``AND`` needs every operand admitted and ``OR`` needs one. ``X WITH e`` is admitted
    only when ``X`` is admitted AND ``e`` is in :data:`ADMITTED_EXCEPTIONS` for ``X``; a
    blocked exception is reported as ``"X WITH e"``. Raises :class:`SpdxError` for an
    expression that does not parse.
    """
    return _blockers(_SpdxParser(_tokenize_spdx(expression)).parse())


def spdx_admitted(expression: str) -> bool:
    """Whether ``expression`` lies inside the admitted licence ids."""
    return not spdx_blockers(expression)


def crate_licence_expression(licence: str) -> str:
    """The SPDX expression a crates.io ``license`` field denotes.

    crates.io parses that field as an SPDX expression with ``/`` as a synonym for ``OR``
    (``allow_slash_as_or_operator: true`` in ``src/licenses.rs`` of rust-lang/crates.io at
    a7ade6b9a23bce78f86174d958409206158f6b15, read 2026-10-13; Cargo documents ``/`` as the
    deprecated separator). So ``MIT/Apache-2.0`` -- recorded verbatim, as crates.io serves
    it -- is checked as ``MIT OR Apache-2.0``.
    """
    return re.sub(r"\s*/\s*", " OR ", licence.strip())


def _licence_ids(expression: str) -> set[str]:
    """Every licence id (not exception id) an expression names."""
    ids: set[str] = set()

    def walk(node: _Node) -> None:
        if node[0] == "ID":
            ids.add(_base_id(node[1]))
        elif node[0] == "WITH":
            walk(node[1])
        else:
            for child in node[1]:
                walk(child)

    walk(_SpdxParser(_tokenize_spdx(expression)).parse())
    return ids


def normalise_name(name: str) -> str:
    """Lowercase, with every run of characters other than ``a-z0-9`` turned into one space."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def excluded_reason(name: str) -> str | None:
    """Why D2 excludes the component named ``name``, or ``None`` when it does not."""
    normalised = normalise_name(name)
    for pattern, reason in EXCLUDED_COMPONENTS:
        if pattern.search(normalised):
            return reason
    return None


# --- the checker -----------------------------------------------------------------------


class _Report:
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def add(self, code: str, where: str, detail: str) -> None:
        self.violations.append(Violation(code, where, detail))


def _is_str(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and value.strip() != ""


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_sha256(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _check_keys(
    node: Mapping[str, Any],
    allowed: frozenset[str],
    required: Iterable[str],
    where: str,
    report: _Report,
) -> None:
    for key in node:
        if key not in allowed:
            report.add("schema", where, f"unknown key {key!r}; allowed: {sorted(allowed)}")
    for key in required:
        if key not in node:
            report.add("schema", where, f"missing required key {key!r}")


def _relative_path_ok(path: Any) -> TypeGuard[str]:
    if not _is_str(path):
        return False
    pure = PurePosixPath(path)
    return not (pure.is_absolute() or ".." in pure.parts or "\\" in path or not pure.parts)


def _check_relative_path(path: Any, where: str, report: _Report) -> TypeGuard[str]:
    if not _relative_path_ok(path):
        report.add(
            "schema",
            where,
            f"path {path!r} must be a non-empty relative POSIX path that stays inside its root",
        )
        return False
    return True


def _check_runtime(runtime: Any, report: _Report) -> None:
    if not isinstance(runtime, Mapping):
        report.add("runtime_not_offline", "runtime", "the runtime contract is missing")
        return
    for section, expected in REQUIRED_RUNTIME.items():
        actual = runtime.get(section)
        if not isinstance(actual, Mapping):
            report.add("runtime_not_offline", f"runtime.{section}", "section is missing")
            continue
        for key, value in expected.items():
            if actual.get(key) != value or type(actual.get(key)) is not type(value):
                report.add(
                    "runtime_not_offline",
                    f"runtime.{section}.{key}",
                    f"must be {value!r}, found {actual.get(key)!r}",
                )


def _check_platform(platform: Any, report: _Report) -> None:
    if not isinstance(platform, Mapping) or platform.get("outside_d2_scope") is not True:
        report.add(
            "schema",
            "platform",
            "must declare outside_d2_scope: true, so the scope boundary is stated",
        )
        return
    items = platform.get("components")
    if not isinstance(items, list) or not items:
        report.add("schema", "platform.components", "must list each platform component")
        return
    for index, item in enumerate(items):
        where = f"platform.components[{index}]"
        if not isinstance(item, Mapping) or not _is_str(item.get("name")):
            report.add("schema", where, "a platform component is a mapping with a name")
            continue
        # A platform library a wheel bundles (``member``) is pinned like any file: the
        # archive it sits in, its sha256 and size, and the class of licence it ships under.
        required = (
            ("name", "source", "sha256", "bytes", "licence") if "member" in item else ("name",)
        )
        _check_keys(item, _PLATFORM_KEYS, required, where, report)
        if "member" in item and not _relative_path_ok(item["member"]):
            report.add("schema", f"{where}.member", "a member is a relative path in its archive")
        if "sha256" in item and not _is_sha256(item["sha256"]):
            report.add("unpinned_file", f"{where}.sha256", "a platform file is pinned by sha256")
        if "bytes" in item and not _is_count(item["bytes"]):
            report.add("unpinned_file", f"{where}.bytes", "a platform file has a positive size")


def _check_platform_members(
    manifest: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]], report: _Report
) -> None:
    """A platform library bundled in a wheel sits in a wheel a library component pins."""
    platform = manifest.get("platform")
    items = platform.get("components") if isinstance(platform, Mapping) else None
    pinned = {
        entry.get("path")
        for component in components.values()
        if component.get("kind") == "library"
        for entry in component.get("files") or []
        if isinstance(entry, Mapping)
    }
    for index, item in enumerate(items if isinstance(items, list) else []):
        if not (isinstance(item, Mapping) and "member" in item and _is_str(item.get("source"))):
            continue
        archive = urllib.parse.unquote(item["source"].rsplit("/", 1)[-1])
        if archive not in pinned:
            report.add(
                "unpinned_file",
                f"platform.components[{index}].source",
                f"{archive!r} is not a file any library component pins",
            )


def _check_header(manifest: Mapping[str, Any], report: _Report) -> None:
    _check_keys(manifest, _MANIFEST_KEYS, _MANIFEST_KEYS, "manifest", report)
    if manifest.get("schema_version") != 1:
        report.add("schema", "schema_version", "must be 1")
    record = manifest.get("decision_record")
    if not isinstance(record, Mapping) or (
        record.get("name") != DECISION_RECORD_NAME
        or record.get("id") != DECISION_RECORD_ID
        or record.get("version") != DECISION_RECORD_VERSION
    ):
        report.add(
            "schema",
            "decision_record",
            f"must name decision_record/{DECISION_RECORD_NAME} ({DECISION_RECORD_ID}) "
            f"version {DECISION_RECORD_VERSION}, the version this checker mirrors",
        )
    target = manifest.get("target")
    if not isinstance(target, Mapping) or not all(
        _is_str(target.get(key)) for key in ("os", "arch", "python", "cuda")
    ):
        report.add("schema", "target", "must state os, arch, python and cuda as strings")
    if not _is_str(manifest.get("scope")):
        report.add("schema", "scope", "the manifest must state its component scope")
    _check_platform(manifest.get("platform"), report)
    _check_runtime(manifest.get("runtime"), report)
    _check_relative_path(manifest.get("vendor_dir"), "vendor_dir", report)


def _check_expression(expression: Any, where: str, report: _Report, *, what: str) -> None:
    if not _is_str(expression):
        report.add("schema", where, f"{what} records an SPDX licence expression")
        return
    try:
        blockers = spdx_blockers(expression)
    except SpdxError as error:
        report.add("licence_unparseable", where, f"{expression!r}: {error}")
        return
    if blockers:
        report.add(
            "licence_not_admitted",
            where,
            f"{expression!r}: {', '.join(blockers)} not admitted by decision record "
            f"{DECISION_RECORD_ID[:8]} v{DECISION_RECORD_VERSION} (ids {sorted(ADMITTED_SPDX)}; "
            f"exceptions {sorted((base, sorted(e)) for base, e in ADMITTED_EXCEPTIONS.items())})",
        )


def _evidence_items(evidence: Any) -> list[Any]:
    return evidence if isinstance(evidence, list) else [evidence]


def _check_evidence_shape(evidence: Any, where: str, report: _Report) -> None:
    items = _evidence_items(evidence)
    if not items:
        report.add("schema", where, "at least one evidence item")
    for index, item in enumerate(items):
        item_where = where if not isinstance(evidence, list) else f"{where}[{index}]"
        ok = (
            isinstance(item, Mapping)
            and set(item) in ({"url", "sha256"}, {"url", "member", "sha256"})
            and _is_str(item.get("url"))
            and item["url"].startswith("https://")
            and _is_sha256(item.get("sha256"))
            and ("member" not in item or _relative_path_ok(item.get("member")))
        )
        if not ok:
            report.add(
                "schema",
                item_where,
                "evidence is {url, sha256} -- or {url, member, sha256} for a file inside an "
                "archive -- the exact text a licence was read from",
            )


def _pins(component: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """The pins a licence read must name: full 40-hex revisions and commits of this
    component (of its source model when it is derived; of its files' upstreams when it is
    vendored code), plus a release's exact tag."""
    pins: set[str] = set()
    source = component.get("source")
    if isinstance(source, Mapping):
        kind = source.get("type")
        if kind == "huggingface":
            pins.add(str(source.get("revision")))
        elif kind == "github_release":
            pins.add(str(source.get("commit")))
            if _is_str(source.get("tag")):
                pins.add(source["tag"])
        elif kind == "git":
            pins.add(str(source.get("commit")))
        elif kind == "pypi" and isinstance(source.get("git"), Mapping):
            pins.add(str(source["git"].get("commit")))
        elif kind == "derived":
            origin = components.get(str(source.get("from")))
            if origin is not None:
                origin_source = origin.get("source")
                if isinstance(origin_source, Mapping) and origin_source.get("type") != "derived":
                    pins |= _pins(origin, components)
    if component.get("kind") == "vendored_code":
        for entry in component.get("files") or []:
            if not isinstance(entry, Mapping):
                continue
            derived = entry.get("derived_from")
            locators = [entry.get("upstream")] + (
                list(derived.get("upstream") or []) if isinstance(derived, Mapping) else []
            )
            for locator in locators:
                if isinstance(locator, Mapping):
                    for key in ("revision", "commit"):
                        if _is_str(locator.get(key)):
                            pins.add(locator[key])
    tag = source.get("tag") if isinstance(source, Mapping) else None
    return {pin for pin in pins if _HEX40.fullmatch(pin) or pin == tag}


def _names_pin(url: str, pins: set[str]) -> bool:
    """Whether one of ``url``'s path segments IS a pin (a whole segment, not a substring)."""
    return bool(pins & set(urllib.parse.unquote(url).split("/")))


def _check_licence(
    component: Mapping[str, Any],
    where: str,
    components: Mapping[str, Mapping[str, Any]],
    report: _Report,
) -> None:
    _check_expression(component.get("spdx"), f"{where}.spdx", report, what="every component")
    url = component.get("licence_url")
    if not (_is_str(url) and url.startswith("https://")):
        report.add("schema", f"{where}.licence_url", "must be an https URL")
        url = None
    evidence = component.get("licence_evidence")
    _check_evidence_shape(evidence, f"{where}.licence_evidence", report)
    pins = _pins(component, components)
    files = {
        entry.get("path") for entry in component.get("files") or [] if isinstance(entry, Mapping)
    }
    if url is not None and not _names_pin(url, pins):
        report.add(
            "licence_unpinned",
            f"{where}.licence_url",
            f"{url!r} names none of the component's pins {sorted(pins)}: the licence must be "
            "read at the version that is pinned",
        )
    for index, item in enumerate(_evidence_items(evidence)):
        if not (isinstance(item, Mapping) and _is_str(item.get("url"))):
            continue
        item_url = item["url"]
        in_pinned_archive = (
            "member" in item and urllib.parse.unquote(item_url.rsplit("/", 1)[-1]) in files
        )
        if not (in_pinned_archive or _names_pin(item_url, pins)):
            report.add(
                "licence_unpinned",
                f"{where}.licence_evidence[{index}]",
                f"{item_url!r} is not read at a pin of this component ({sorted(pins)}) nor "
                "from a member of a pinned file",
            )
    read_at = component.get("licence_read_at")
    if not (isinstance(read_at, str) and _TIMESTAMP.fullmatch(read_at)):
        report.add(
            "schema", f"{where}.licence_read_at", "must be a UTC timestamp YYYY-MM-DDTHH:MM:SSZ"
        )
    else:
        try:
            datetime.strptime(read_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            report.add("schema", f"{where}.licence_read_at", f"{read_at!r} is not a real time")


def _names(component: Mapping[str, Any]) -> list[str]:
    """Every name that identifies a component or the upstream its bytes came from."""
    names: list[Any] = [component.get("id")]
    source = component.get("source")
    if isinstance(source, Mapping):
        names += [source.get("repo"), source.get("package"), source.get("from")]
        git = source.get("git")
        if isinstance(git, Mapping):
            names.append(git.get("repo"))
    for entry in component.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        derived = entry.get("derived_from")
        locators = [entry.get("upstream")] + (
            list(derived.get("upstream") or []) if isinstance(derived, Mapping) else []
        )
        for locator in locators:
            if isinstance(locator, Mapping):
                names += [locator.get("repo"), locator.get("component")]
    return [name for name in names if isinstance(name, str)]


def _check_exclusions(component: Mapping[str, Any], where: str, report: _Report) -> None:
    for name in _names(component):
        reason = excluded_reason(name)
        if reason:
            report.add("excluded_component", where, f"{name!r}: {reason}")
            return


def _check_source(component: Mapping[str, Any], where: str, report: _Report) -> None:
    kind = component.get("kind")
    if kind == "vendored_code":
        if "source" in component:
            report.add("schema", where, "vendored_code records its upstream per file")
        return
    source = component.get("source")
    if not isinstance(source, Mapping):
        report.add("schema", f"{where}.source", "missing")
        return
    source_type = source.get("type")
    if source_type not in _SOURCE_TYPES.get(str(kind), frozenset()):
        report.add(
            "schema",
            f"{where}.source.type",
            f"{source_type!r} is not a source type for kind {kind!r}",
        )
        return
    _check_keys(source, _SOURCE_KEYS[source_type], (), f"{where}.source", report)
    if source_type in ("huggingface", "github_release", "git") and not (
        isinstance(source.get("repo"), str) and _REPO.fullmatch(source["repo"])
    ):
        report.add("schema", f"{where}.source.repo", "must be owner/name")
    pins = {"huggingface": "revision", "github_release": "commit", "git": "commit"}
    if source_type in pins:
        pin = source.get(pins[source_type])
        if not (isinstance(pin, str) and _HEX40.fullmatch(pin)):
            report.add(
                "unpinned_revision",
                f"{where}.source.{pins[source_type]}",
                f"{pin!r} is not a full 40-hex commit id",
            )
    if source_type == "github_release" and not _is_str(source.get("tag")):
        report.add("schema", f"{where}.source.tag", "a release asset names its tag")
    if source_type == "derived" and not _is_str(source.get("from")):
        report.add("schema", f"{where}.source.from", "a derived component names its source")
    if source_type == "pypi":
        if not _is_str(source.get("package")) or not _is_str(source.get("version")):
            report.add("schema", f"{where}.source", "pypi needs package and version strings")
        index = source.get("index")
        if not (_is_str(index) and index.startswith("https://")):
            report.add("schema", f"{where}.source.index", "must be an https package index")
        git = source.get("git")
        if git is not None:
            if not isinstance(git, Mapping) or set(git) != {"repo", "commit"}:
                report.add("schema", f"{where}.source.git", "must be {repo, commit}")
            elif not (isinstance(git["commit"], str) and _HEX40.fullmatch(git["commit"])):
                report.add(
                    "unpinned_revision",
                    f"{where}.source.git.commit",
                    f"{git['commit']!r} is not a full 40-hex commit id",
                )


def _check_locator(locator: Any, where: str, report: _Report) -> None:
    if not isinstance(locator, Mapping) or locator.get("type") not in _UPSTREAM_KEYS:
        report.add("schema", where, f"upstream type must be one of {sorted(_UPSTREAM_KEYS)}")
        return
    _check_keys(
        locator, _UPSTREAM_KEYS[locator["type"]], _UPSTREAM_KEYS[locator["type"]], where, report
    )
    if not _is_sha256(locator.get("sha256")):
        report.add("unpinned_file", f"{where}.sha256", "upstream bytes must be pinned by sha256")
    if locator["type"] == "manifest":
        return
    pin_key = "revision" if locator["type"] == "huggingface" else "commit"
    pin = locator.get(pin_key)
    if not (isinstance(pin, str) and _HEX40.fullmatch(pin)):
        report.add("unpinned_revision", f"{where}.{pin_key}", f"{pin!r} is not a full commit id")
    if not (isinstance(locator.get("repo"), str) and _REPO.fullmatch(locator["repo"])):
        report.add("schema", f"{where}.repo", "must be owner/name")
    _check_relative_path(locator.get("path"), f"{where}.path", report)


def _check_files(
    component: Mapping[str, Any],
    where: str,
    vendor_dir: str,
    report: _Report,
) -> None:
    files = component.get("files")
    if not isinstance(files, list) or not files:
        report.add(
            "unpinned_file", f"{where}.files", "a component is admitted only with pinned files"
        )
        return
    kind = component.get("kind")
    source = component.get("source")
    derived_entry = isinstance(source, Mapping) and source.get("type") == "derived"
    seen: set[str] = set()
    for index, entry in enumerate(files):
        file_where = f"{where}.files[{index}]"
        if not isinstance(entry, Mapping):
            report.add("schema", file_where, "a file is a mapping")
            continue
        allowed = {"path", "sha256", "bytes"}
        if kind == "vendored_code":
            allowed |= {"upstream", "derived_from"}
        elif derived_entry:
            allowed |= {"derived_from"}
        _check_keys(entry, frozenset(allowed), ("path",), file_where, report)
        path = entry.get("path")
        if _check_relative_path(path, f"{file_where}.path", report):
            if path in seen:
                report.add("schema", f"{file_where}.path", f"{path!r} is listed twice")
            seen.add(path)
            if kind == "vendored_code" and not path.startswith(f"{vendor_dir}/"):
                report.add(
                    "schema", f"{file_where}.path", f"vendored files live under {vendor_dir}/"
                )
        sha = entry.get("sha256")
        if not _is_sha256(sha):
            report.add("unpinned_file", f"{file_where}.sha256", f"{path!r} is not pinned by sha256")
        if not _is_count(entry.get("bytes")):
            report.add(
                "unpinned_file", f"{file_where}.bytes", f"{path!r} has no positive byte size"
            )
        if kind == "vendored_code":
            has_upstream, has_derived = "upstream" in entry, "derived_from" in entry
            if has_upstream == has_derived:
                report.add(
                    "schema",
                    file_where,
                    "a vendored file names exactly one of upstream (byte-identical) or derived_from",
                )
            if has_upstream:
                _check_locator(entry["upstream"], f"{file_where}.upstream", report)
                upstream = entry["upstream"]
                if isinstance(upstream, Mapping) and upstream.get("type") == "manifest":
                    report.add(
                        "schema", f"{file_where}.upstream", "vendored code comes from a repo"
                    )
                elif isinstance(upstream, Mapping) and upstream.get("sha256") != sha:
                    report.add(
                        "vendored_mismatch",
                        f"{file_where}.upstream",
                        "claims byte-identity with upstream but the sha256s differ",
                    )
        if derived_entry and "derived_from" not in entry:
            report.add(
                "derived_invalid",
                file_where,
                "every file of a derived component names derived_from",
            )
        if "derived_from" in entry:
            derived = entry["derived_from"]
            if not (
                isinstance(derived, Mapping)
                and set(derived) == {"upstream", "conversion"}
                and _is_str(derived.get("conversion"))
                and isinstance(derived.get("upstream"), list)
                and derived["upstream"]
            ):
                report.add(
                    "derived_invalid",
                    f"{file_where}.derived_from",
                    "must be {upstream: [locators], conversion: what was done}",
                )
                continue
            for up_index, locator in enumerate(derived["upstream"]):
                _check_locator(locator, f"{file_where}.derived_from.upstream[{up_index}]", report)


def _check_bundled(component: Mapping[str, Any], where: str, report: _Report) -> None:
    """What a wheel ships: its licence files, the native libraries it bundles, and texts
    read but not terms (decision record a5ec7670 v2, "How a licence is read").

    A library installed from a wheel (``source.type: pypi``) must record this; every
    licence file it lists must be one whose bytes were read (its sha256 is among the
    component's ``licence_evidence``), so a wheel that ships an unread licence is refused.
    """
    bundled = component.get("bundled")
    source = component.get("source")
    if bundled is None:
        if (
            component.get("kind") == "library"
            and isinstance(source, Mapping)
            and source.get("type") == "pypi"
        ):
            report.add(
                "schema",
                f"{where}.bundled",
                "a wheel's licence files and bundled native libraries are recorded (bundled)",
            )
        return
    where = f"{where}.bundled"
    if not isinstance(bundled, Mapping):
        report.add("schema", where, "bundled is a mapping")
        return
    _check_keys(
        bundled,
        frozenset({"licence_files", "native_libraries", "not_terms"}),
        ("licence_files",),
        where,
        report,
    )
    licence_files = bundled.get("licence_files")
    if not isinstance(licence_files, list) or not licence_files:
        report.add("schema", f"{where}.licence_files", "list every licence file the wheel ships")
        licence_files = []
    read = {
        item.get("sha256")
        for item in _evidence_items(component.get("licence_evidence"))
        if isinstance(item, Mapping)
    }
    for index, item in enumerate(licence_files):
        item_where = f"{where}.licence_files[{index}]"
        if not (isinstance(item, Mapping) and set(item) == {"member", "sha256", "bytes"}):
            report.add("schema", item_where, "a licence file is {member, sha256, bytes}")
            continue
        if not (_is_sha256(item["sha256"]) and _is_count(item["bytes"])):
            report.add("unpinned_file", item_where, "a licence file is pinned by sha256 and size")
        elif item["sha256"] not in read:
            report.add(
                "licence_unread",
                item_where,
                f"the wheel ships {item['member']!r}, but no licence_evidence item has its "
                "sha256: read it and record it as evidence",
            )
    expression = component.get("spdx")
    try:
        component_ids = _licence_ids(expression) if _is_str(expression) else set()
    except SpdxError:
        component_ids = set()
    for index, item in enumerate(bundled.get("native_libraries") or []):
        item_where = f"{where}.native_libraries[{index}]"
        keys = frozenset({"member", "sha256", "bytes", "identity", "spdx", "licence_evidence"})
        if not isinstance(item, Mapping):
            report.add("schema", item_where, "a native library is a mapping")
            continue
        _check_keys(item, keys, keys, item_where, report)
        if not (_is_sha256(item.get("sha256")) and _is_count(item.get("bytes"))):
            report.add("unpinned_file", item_where, "a native library is pinned by sha256 and size")
        if not _is_str(item.get("identity")):
            report.add("schema", f"{item_where}.identity", "names the library and its version")
        _check_expression(item.get("spdx"), f"{item_where}.spdx", report, what="a native library")
        _check_evidence_shape(
            item.get("licence_evidence"), f"{item_where}.licence_evidence", report
        )
        for ev_index, evidence in enumerate(_evidence_items(item.get("licence_evidence"))):
            url = evidence.get("url") if isinstance(evidence, Mapping) else None
            if _is_str(url) and not any(
                _HEX40.fullmatch(segment) for segment in urllib.parse.unquote(url).split("/")
            ):
                report.add(
                    "licence_unpinned",
                    f"{item_where}.licence_evidence[{ev_index}]",
                    f"{url!r} names no commit: a library's licence is read at a pinned commit",
                )
        if _is_str(item.get("spdx")):
            try:
                missing = _licence_ids(item["spdx"]) - component_ids
            except SpdxError:
                continue
            if missing:
                report.add(
                    "licence_incomplete",
                    f"{where.rsplit('.bundled', 1)[0]}.spdx",
                    f"the wheel bundles {item.get('member')!r} under {sorted(missing)}, which the "
                    "component's expression does not name",
                )
    for index, item in enumerate(bundled.get("not_terms") or []):
        item_where = f"{where}.not_terms[{index}]"
        keys = frozenset({"text", "licence", "reason"})
        if not (
            isinstance(item, Mapping) and set(item) == keys and all(_is_str(item[k]) for k in keys)
        ):
            report.add(
                "schema",
                item_where,
                "a text read but not a term is {text, licence, reason}: say why it is not one",
            )


def _check_static_dependencies(component: Mapping[str, Any], where: str, report: _Report) -> None:
    """Statically linked dependencies, enumerated from the upstream lockfile (a5ec7670 v2).

    Every LINKED crate's licence must be admitted (a crates.io ``license`` field is read
    through :func:`crate_licence_expression`), and an unreadable one (``UNKNOWN``) is never
    assumed admitted. A crate the lockfile lists but the shipped binary does not link (a
    build-time proc-macro, another target's dependency) is recorded as read. The lockfile
    must come from the pinned version: for a PyPI package, that version's sdist.
    """
    deps = component.get("static_dependencies")
    if deps is None:
        return
    where = f"{where}.static_dependencies"
    keys = frozenset({"lockfile", "resolution", "read_at", "crates"})
    if not isinstance(deps, Mapping):
        report.add("schema", where, "static_dependencies is a mapping")
        return
    _check_keys(deps, keys, keys, where, report)
    lockfile = deps.get("lockfile")
    lock_keys = {"url", "sha256", "member", "member_sha256"}
    if not (
        isinstance(lockfile, Mapping)
        and set(lockfile) == lock_keys
        and _is_str(lockfile.get("url"))
        and lockfile["url"].startswith("https://")
        and _is_sha256(lockfile.get("sha256"))
        and _is_sha256(lockfile.get("member_sha256"))
        and _relative_path_ok(lockfile.get("member"))
    ):
        report.add(
            "schema",
            f"{where}.lockfile",
            "must be {url, sha256, member, member_sha256}: the archive the lockfile was read "
            "from and the lockfile inside it, each pinned by sha256",
        )
    else:
        _check_lockfile_version(component, lockfile, f"{where}.lockfile", report)
    if not _is_str(deps.get("resolution")):
        report.add("schema", f"{where}.resolution", "say how the linked set was determined")
    read_at = deps.get("read_at")
    if not (isinstance(read_at, str) and _TIMESTAMP.fullmatch(read_at)):
        report.add("schema", f"{where}.read_at", "must be a UTC timestamp YYYY-MM-DDTHH:MM:SSZ")
    crates = deps.get("crates")
    if not isinstance(crates, list) or not crates:
        report.add("schema", f"{where}.crates", "the lockfile's crates, each with its licence")
        return
    seen: set[tuple[str, str]] = set()
    crate_keys = frozenset({"name", "version", "checksum", "licence", "linked"})
    for index, crate in enumerate(crates):
        crate_where = f"{where}.crates[{index}]"
        if not isinstance(crate, Mapping):
            report.add("schema", crate_where, "a crate is a mapping")
            continue
        _check_keys(crate, crate_keys, crate_keys, crate_where, report)
        name, version = crate.get("name"), crate.get("version")
        if not (_is_str(name) and _is_str(version)):
            report.add("schema", crate_where, "a crate names its name and version")
            continue
        crate_where = f"{where}.crates[{name}@{version}]"
        if (name, version) in seen:
            report.add("schema", crate_where, "listed twice")
        seen.add((name, version))
        if not _is_sha256(crate.get("checksum")):
            report.add("unpinned_file", crate_where, "a crate is pinned by its lockfile checksum")
        linked = crate.get("linked")
        if not isinstance(linked, bool):
            report.add("schema", f"{crate_where}.linked", "linked is true or false")
            continue
        licence = crate.get("licence")
        if not _is_str(licence):
            report.add("schema", f"{crate_where}.licence", "the licence as read, or UNKNOWN")
        elif linked and licence == "UNKNOWN":
            report.add(
                "licence_unknown",
                crate_where,
                "a statically linked crate whose licence could not be read is never assumed "
                "admitted",
            )
        elif linked:
            _check_expression(
                crate_licence_expression(licence), crate_where, report, what="a linked crate"
            )


def _check_lockfile_version(
    component: Mapping[str, Any], lockfile: Mapping[str, Any], where: str, report: _Report
) -> None:
    """The lockfile is read from the pinned version: a PyPI package's sdist for exactly its
    version (``<name>-<version>.tar.gz``, holding ``<name>-<version>/...``), otherwise an
    archive URL that names the pinned commit."""
    source = component.get("source")
    if not isinstance(source, Mapping):
        return
    url, member = lockfile["url"], lockfile["member"]
    if source.get("type") == "pypi":
        package, version = source.get("package"), source.get("version")
        if not (_is_str(package) and _is_str(version)):
            return
        normalised = re.sub(r"[-_.]+", "_", package).lower()
        stems = {f"{package}-{version}", f"{normalised}-{version}"}
        archive = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if not any(archive == f"{stem}.tar.gz" and member.startswith(f"{stem}/") for stem in stems):
            report.add(
                "licence_unpinned",
                where,
                f"{archive!r} / {member!r} is not the sdist of {package} {version}: the "
                "lockfile must be read at the pinned version",
            )
    elif not _names_pin(url, {pin for pin in _pins(component, {}) if _HEX40.fullmatch(pin)}):
        report.add("licence_unpinned", where, f"{url!r} names no pinned commit of this component")


def _check_derivations(components: Mapping[str, Mapping[str, Any]], report: _Report) -> None:
    """Derived files point at real source bytes and carry the source's licence."""
    for component_id, component in components.items():
        where = f"components[{component_id}]"
        source = component.get("source")
        declared_from = source.get("from") if isinstance(source, Mapping) else None
        if isinstance(source, Mapping) and source.get("type") == "derived":
            origin = components.get(str(declared_from))
            if origin is None or origin.get("kind") != "model":
                report.add(
                    "derived_invalid",
                    f"{where}.source.from",
                    f"{declared_from!r} is not a model in the manifest",
                )
            elif origin.get("spdx") != component.get("spdx"):
                report.add(
                    "derived_invalid",
                    f"{where}.spdx",
                    f"a derived file carries its source's licence ({origin.get('spdx')!r}), "
                    f"not {component.get('spdx')!r}",
                )
        for index, entry in enumerate(component.get("files") or []):
            derived = entry.get("derived_from") if isinstance(entry, Mapping) else None
            if not isinstance(derived, Mapping) or not isinstance(derived.get("upstream"), list):
                continue
            for up_index, locator in enumerate(derived["upstream"]):
                if not isinstance(locator, Mapping) or locator.get("type") != "manifest":
                    continue
                locator_where = f"{where}.files[{index}].derived_from.upstream[{up_index}]"
                origin_id = locator.get("component")
                origin = components.get(str(origin_id))
                if origin is None:
                    report.add(
                        "derived_invalid", locator_where, f"{origin_id!r} is not in the manifest"
                    )
                    continue
                if declared_from is not None and origin_id != declared_from:
                    report.add(
                        "derived_invalid",
                        locator_where,
                        f"derives from {origin_id!r} but the component's source is {declared_from!r}",
                    )
                origin_hashes = {
                    item.get("sha256")
                    for item in origin.get("files") or []
                    if isinstance(item, Mapping)
                }
                if locator.get("sha256") not in origin_hashes:
                    report.add(
                        "derived_invalid",
                        locator_where,
                        f"no file of {origin_id!r} has sha256 {locator.get('sha256')!r}",
                    )
                if origin.get("spdx") != component.get("spdx"):
                    report.add(
                        "derived_invalid",
                        locator_where,
                        f"carries {component.get('spdx')!r} but its source {origin_id!r} is "
                        f"{origin.get('spdx')!r}",
                    )


def _check_hash_sizes(
    manifest: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]], report: _Report
) -> None:
    """One sha256 names one byte string, so it has one size wherever it is listed."""
    sizes: dict[str, tuple[int, str]] = {}

    def note(sha: Any, size: Any, where: str) -> None:
        if not (_is_sha256(sha) and _is_count(size)):
            return
        if sha in sizes and sizes[sha][0] != size:
            report.add(
                "hash_size_conflict",
                where,
                f"sha256 {sha} is listed as {size} bytes here and {sizes[sha][0]} bytes at "
                f"{sizes[sha][1]}",
            )
        sizes.setdefault(sha, (size, where))

    for component_id, component in components.items():
        for index, entry in enumerate(component.get("files") or []):
            if isinstance(entry, Mapping):
                note(
                    entry.get("sha256"),
                    entry.get("bytes"),
                    f"components[{component_id}].files[{index}]",
                )
        bundled = component.get("bundled")
        if isinstance(bundled, Mapping):
            for group in ("licence_files", "native_libraries"):
                for index, item in enumerate(bundled.get(group) or []):
                    if isinstance(item, Mapping):
                        note(
                            item.get("sha256"),
                            item.get("bytes"),
                            f"components[{component_id}].bundled.{group}[{index}]",
                        )
    platform = manifest.get("platform")
    if isinstance(platform, Mapping):
        for index, item in enumerate(platform.get("components") or []):
            if isinstance(item, Mapping):
                note(item.get("sha256"), item.get("bytes"), f"platform.components[{index}]")


def _check_runners(
    runners: Any,
    components: Mapping[str, Mapping[str, Any]],
    report: _Report,
) -> dict[str, Mapping[str, Any]]:
    declared: dict[str, Mapping[str, Any]] = {}
    if not isinstance(runners, list) or not runners:
        report.add("schema", "runners", "the manifest declares the runners recipes may name")
        return declared
    for index, runner in enumerate(runners):
        where = f"runners[{index}]"
        if not isinstance(runner, Mapping):
            report.add("schema", where, "a runner is a mapping")
            continue
        _check_keys(
            runner, frozenset({"id", "kind", "code"}), ("id", "kind", "code"), where, report
        )
        runner_id = runner.get("id")
        if not _is_str(runner_id):
            continue
        where = f"runners[{runner_id}]"
        if runner_id in declared:
            report.add("duplicate_id", where, "runner ids are unique")
        declared[runner_id] = runner
        if runner.get("kind") not in ("diffusers", "named"):
            report.add("schema", f"{where}.kind", "must be 'diffusers' or 'named'")
        code = runner.get("code")
        if not isinstance(code, list) or not code:
            report.add(
                "runner_unresolved", f"{where}.code", "a runner names the pinned code it executes"
            )
            continue
        for code_id in code:
            target = components.get(str(code_id))
            if target is None or target.get("kind") not in ("library", "vendored_code"):
                report.add(
                    "runner_unresolved",
                    f"{where}.code",
                    f"{code_id!r} is not a library or vendored_code component",
                )
        if runner.get("kind") == "diffusers" and "diffusers" not in code:
            report.add(
                "runner_unresolved",
                f"{where}.code",
                "a diffusers runner runs the diffusers library",
            )
    vendored_paths = {
        entry.get("path"): code_id
        for code_id, code in components.items()
        if code.get("kind") == "vendored_code"
        for entry in code.get("files") or []
        if isinstance(entry, Mapping)
    }
    for component_id, component in components.items():
        runner_id = component.get("runner")
        if runner_id is not None and runner_id not in declared:
            report.add(
                "runner_unresolved",
                f"components[{component_id}].runner",
                f"{runner_id!r} is not a declared runner",
            )
        architecture = component.get("architecture")
        module = architecture.get("module") if isinstance(architecture, Mapping) else None
        if module is None:
            continue
        owner = vendored_paths.get(module)
        runner_code = declared.get(str(runner_id), {}).get("code") or []
        if owner is None or owner not in runner_code:
            report.add(
                "runner_unresolved",
                f"components[{component_id}].architecture.module",
                f"{module!r} is not a vendored file that runner {runner_id!r} executes",
            )
    return declared


def _check_vendored_tree(
    components: Mapping[str, Mapping[str, Any]],
    vendor_dir: str,
    base_dir: Path,
    report: _Report,
) -> None:
    """Vendored bytes on disk match their pins, and nothing unlisted sits beside them.

    ``vendor_dir`` has already passed :func:`_relative_path_ok`, so the sweep stays under
    ``base_dir``.
    """
    listed: dict[str, tuple[Any, Any, str]] = {}
    for component_id, component in components.items():
        if component.get("kind") != "vendored_code":
            continue
        for index, entry in enumerate(component.get("files") or []):
            if isinstance(entry, Mapping) and _relative_path_ok(entry.get("path")):
                listed[entry["path"]] = (
                    entry.get("sha256"),
                    entry.get("bytes"),
                    f"components[{component_id}].files[{index}]",
                )
    for path, (sha, size, where) in sorted(listed.items()):
        on_disk = base_dir / path
        if not on_disk.is_file():
            report.add("vendored_mismatch", where, f"{path} is missing on disk")
            continue
        data = on_disk.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            report.add(
                "vendored_mismatch",
                where,
                f"{path} on disk is {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}; "
                f"the manifest pins {size} bytes, sha256 {sha}",
            )
    root = base_dir / vendor_dir
    if not root.is_dir():
        return
    for on_disk in sorted(root.rglob("*")):
        if on_disk.is_dir() or "__pycache__" in on_disk.parts or on_disk.suffix == ".pyc":
            continue
        relative = on_disk.relative_to(base_dir).as_posix()
        if relative in listed:
            continue
        if on_disk.name == "__init__.py" and on_disk.stat().st_size == 0:
            continue  # an empty package marker carries no third-party content
        report.add(
            "vendored_unlisted", relative, "a file under vendor_dir that the manifest does not pin"
        )


def _component_index(manifest: Mapping[str, Any], report: _Report) -> dict[str, Mapping[str, Any]]:
    components: dict[str, Mapping[str, Any]] = {}
    entries = manifest.get("components")
    if not isinstance(entries, list) or not entries:
        report.add("schema", "components", "the manifest lists its components")
        return components
    for index, component in enumerate(entries):
        if not isinstance(component, Mapping):
            report.add("schema", f"components[{index}]", "a component is a mapping")
            continue
        component_id = component.get("id")
        if not (isinstance(component_id, str) and _COMPONENT_ID.fullmatch(component_id)):
            report.add(
                "schema", f"components[{index}].id", f"{component_id!r} is not a lowercase id"
            )
            continue
        if component_id in components:
            report.add("duplicate_id", f"components[{component_id}]", "component ids are unique")
        components[component_id] = component
    return components


def check_manifest(manifest: Any, *, base_dir: Path | None = None) -> list[Violation]:
    """Every reason ``manifest`` is refused; ``[]`` admits it.

    ``base_dir`` is the directory holding the manifest. When given, vendored files are
    verified on disk and the vendor directory is swept for unlisted files -- unless
    ``vendor_dir`` is not a relative path inside ``base_dir``, which is itself reported.
    """
    report = _Report()
    if not isinstance(manifest, Mapping):
        report.add("schema", "manifest", "the manifest is a mapping")
        return report.violations
    _check_header(manifest, report)
    raw_vendor_dir = manifest.get("vendor_dir")
    vendor_dir = raw_vendor_dir if _relative_path_ok(raw_vendor_dir) else None
    components = _component_index(manifest, report)
    for component_id, component in components.items():
        where = f"components[{component_id}]"
        kind = component.get("kind")
        if kind not in _KIND_KEYS:
            report.add("schema", f"{where}.kind", f"must be one of {sorted(_KIND_KEYS)}")
            continue
        _check_keys(
            component,
            _COMMON_COMPONENT_KEYS | _KIND_KEYS[kind],
            ("id", "kind", "spdx", "licence_url", "licence_evidence", "licence_read_at", "files"),
            where,
            report,
        )
        _check_licence(component, where, components, report)
        _check_exclusions(component, where, report)
        _check_source(component, where, report)
        _check_files(component, where, vendor_dir or "vendor", report)
        _check_bundled(component, where, report)
        _check_static_dependencies(component, where, report)
        if kind == "model":
            size_multiple = component.get("size_multiple")
            if size_multiple is not None and not _is_count(size_multiple):
                report.add("schema", f"{where}.size_multiple", "must be a positive integer")
            for key in ("role", "runner", "scheduler_class"):
                if key in component and not _is_str(component[key]):
                    report.add("schema", f"{where}.{key}", "must be a non-empty string")
            if "architecture" in component and not isinstance(component["architecture"], Mapping):
                report.add("schema", f"{where}.architecture", "must be a mapping")
    _check_derivations(components, report)
    _check_hash_sizes(manifest, components, report)
    _check_platform_members(manifest, components, report)
    _check_runners(manifest.get("runners"), components, report)
    if base_dir is not None and vendor_dir is not None:
        _check_vendored_tree(components, vendor_dir, base_dir, report)
    return report.violations


# --- recipes ---------------------------------------------------------------------------


def _scan_network(node: Any, where: str, report: _Report) -> None:
    """A recipe may not enable a network fetch anywhere in its tree."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{where}.{key}"
            name = str(key)
            if name == "trust_remote_code" and value is not False:
                report.add(
                    "network_fetch", child, "trust_remote_code runs code fetched from the Hub"
                )
            elif name == "local_files_only" and value is not True:
                report.add("network_fetch", child, "loads are local_files_only")
            elif name.upper() == "HF_HUB_OFFLINE" and str(value) != "1":
                report.add("network_fetch", child, "the provider runs with HF_HUB_OFFLINE=1")
            elif name in HUB_LOADER_KEYS:
                report.add(
                    "network_fetch",
                    child,
                    f"{name!r} is a Hub loader argument; recipes name manifest ids",
                )
            _scan_network(value, child, report)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _scan_network(item, f"{where}[{index}]", report)
    elif isinstance(node, str) and _NETWORK.search(node):
        report.add("network_fetch", where, f"{node!r} points at the network")


def _resolve_models(
    models: Any,
    runner_id: Any,
    where: str,
    components: Mapping[str, Mapping[str, Any]],
    report: _Report,
    *,
    allow_empty: bool,
) -> list[Mapping[str, Any]]:
    resolved: list[Mapping[str, Any]] = []
    if not isinstance(models, list) or (not models and not allow_empty):
        report.add("recipe_schema", where, "must list manifest model ids")
        return resolved
    for model_id in models:
        component = components.get(str(model_id))
        if component is None or component.get("kind") != "model":
            report.add("model_unresolved", where, f"{model_id!r} is not a model in the manifest")
            continue
        if component.get("runner") != runner_id:
            report.add(
                "pipeline_mismatch",
                where,
                f"{model_id!r} runs under {component.get('runner')!r}, not {runner_id!r}",
            )
        resolved.append(component)
    return resolved


def _check_parameters(
    parameters: Any,
    models: list[Mapping[str, Any]],
    model_ids: list[Any],
    where: str,
    report: _Report,
) -> None:
    if not isinstance(parameters, Mapping):
        report.add("recipe_schema", where, "parameters is a mapping")
        return
    scheduler = parameters.get("scheduler")
    if scheduler is not None:
        if not (isinstance(scheduler, Mapping) and _is_str(scheduler.get("class"))):
            report.add("recipe_schema", f"{where}.scheduler", "must name its class")
        else:
            for model in models:
                expected = model.get("scheduler_class")
                if expected is not None and scheduler["class"] != expected:
                    report.add(
                        "pipeline_mismatch",
                        f"{where}.scheduler.class",
                        f"{scheduler['class']!r} differs from {model.get('id')!r}'s scheduler {expected!r}",
                    )
    scale_models = parameters.get("scale_models")
    if scale_models is not None:
        if not isinstance(scale_models, Mapping) or not scale_models:
            report.add("recipe_schema", f"{where}.scale_models", "maps a scale to a model id")
        else:
            for scale, model_id in scale_models.items():
                if model_id not in model_ids:
                    report.add(
                        "model_unresolved",
                        f"{where}.scale_models.{scale}",
                        f"{model_id!r} is not one of this recipe's models {model_ids}",
                    )
    presets = parameters.get("size_presets")
    if presets is None:
        return
    if not isinstance(presets, Mapping) or not presets:
        report.add(
            "recipe_schema", f"{where}.size_presets", "must map preset names to [width, height]"
        )
        return
    multiples = {model.get("size_multiple") for model in models if model.get("size_multiple")}
    for name, size in presets.items():
        size_where = f"{where}.size_presets.{name}"
        if not (
            isinstance(size, list) and len(size) == 2 and all(_is_count(side) for side in size)
        ):
            report.add("recipe_schema", size_where, "must be [width, height] in positive pixels")
            continue
        for multiple in multiples:
            if size[0] % multiple or size[1] % multiple:
                report.add(
                    "recipe_schema",
                    size_where,
                    f"{size[0]}x{size[1]} is not a multiple of {multiple}, which the pipeline requires",
                )
    default = parameters.get("default_size")
    if default not in presets:
        report.add(
            "recipe_schema", f"{where}.default_size", f"{default!r} is not one of the size presets"
        )


def _check_seed_policy(
    policy: Any, runner: Mapping[str, Any] | None, where: str, report: _Report
) -> None:
    if not isinstance(policy, Mapping):
        report.add("recipe_schema", where, "seed_policy is a mapping")
        return
    mode = policy.get("mode")
    if mode == "seeded":
        _check_keys(
            policy,
            frozenset({"mode", "base", "per_candidate"}),
            ("base", "per_candidate"),
            where,
            report,
        )
        if policy.get("base") not in SEED_BASES:
            report.add("recipe_schema", f"{where}.base", f"must be one of {sorted(SEED_BASES)}")
        if policy.get("per_candidate") not in SEED_PER_CANDIDATE:
            report.add(
                "recipe_schema",
                f"{where}.per_candidate",
                f"must be one of {sorted(SEED_PER_CANDIDATE)}",
            )
    elif mode == "deterministic":
        _check_keys(policy, frozenset({"mode"}), (), where, report)
        if runner is not None and runner.get("kind") == "diffusers":
            report.add("recipe_schema", f"{where}.mode", "a diffusion pipeline samples from a seed")
    else:
        report.add("recipe_schema", f"{where}.mode", "must be 'seeded' or 'deterministic'")


def check_recipe(name: str, recipe: Any, manifest: Mapping[str, Any]) -> list[Violation]:
    """Every reason the recipe in ``recipes/<name>.yaml`` is refused against ``manifest``."""
    report = _Report()
    where = f"recipes[{name}]"
    if not isinstance(recipe, Mapping):
        report.add("recipe_schema", where, "a recipe is a mapping")
        return report.violations
    _scan_network(recipe, where, report)
    _check_keys(recipe, _RECIPE_KEYS, _RECIPE_REQUIRED, where, report)
    if recipe.get("id") != name:
        report.add(
            "recipe_schema",
            f"{where}.id",
            f"{recipe.get('id')!r} must equal the file stem {name!r}",
        )
    if not _is_count(recipe.get("version")):
        report.add("recipe_schema", f"{where}.version", "a recipe version is a positive integer")
    silent = _Report()  # the manifest's own violations are reported by check_manifest
    components = _component_index(manifest, silent)
    runners = {
        str(runner.get("id")): runner
        for runner in manifest.get("runners") or []
        if isinstance(runner, Mapping)
    }
    weighted = {
        str(component.get("runner")) for component in components.values() if component.get("runner")
    }
    pipeline = recipe.get("pipeline")
    runner = runners.get(str(pipeline))
    if runner is None:
        report.add(
            "runner_unresolved", f"{where}.pipeline", f"{pipeline!r} is not a declared runner"
        )
    model_ids = recipe.get("models")
    models = _resolve_models(
        model_ids, pipeline, f"{where}.models", components, report, allow_empty=False
    )
    _check_parameters(
        recipe.get("parameters"),
        models,
        model_ids if isinstance(model_ids, list) else [],
        f"{where}.parameters",
        report,
    )
    post = recipe.get("post")
    if not isinstance(post, list):
        report.add("recipe_schema", f"{where}.post", "post is a (possibly empty) list of steps")
        post = []
    for index, step in enumerate(post):
        step_where = f"{where}.post[{index}]"
        if not isinstance(step, Mapping):
            report.add("recipe_schema", step_where, "a post step is a mapping")
            continue
        _check_keys(step, _POST_STEP_KEYS, ("op", "runner", "models"), step_where, report)
        if step.get("op") not in POST_OPS:
            report.add("recipe_schema", f"{step_where}.op", f"must be one of {sorted(POST_OPS)}")
        step_runner = step.get("runner")
        if str(step_runner) not in runners:
            report.add(
                "runner_unresolved",
                f"{step_where}.runner",
                f"{step_runner!r} is not a declared runner",
            )
        step_models = step.get("models")
        needs_weights = str(step_runner) in weighted
        if needs_weights and isinstance(step_models, list) and not step_models:
            report.add(
                "recipe_schema", f"{step_where}.models", f"{step_runner!r} loads weights; name them"
            )
        if not needs_weights and step_models:
            report.add("recipe_schema", f"{step_where}.models", f"{step_runner!r} loads no weights")
        resolved = _resolve_models(
            step_models, step_runner, f"{step_where}.models", components, report, allow_empty=True
        )
        if "parameters" in step:
            _check_parameters(
                step["parameters"],
                resolved,
                step_models if isinstance(step_models, list) else [],
                f"{step_where}.parameters",
                report,
            )
    _check_seed_policy(recipe.get("seed_policy"), runner, f"{where}.seed_policy", report)
    return report.violations


def check_recipes(recipes: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[Violation]:
    """Every reason any recipe is refused; ``recipes`` is keyed by file stem."""
    violations: list[Violation] = []
    if not recipes:
        violations.append(Violation("recipe_schema", "recipes", "no recipes found"))
    for name, recipe in sorted(recipes.items()):
        violations.extend(check_recipe(name, recipe, manifest))
    return violations


def check(manifest_path: Path = MANIFEST_PATH, recipes_dir: Path = RECIPES_DIR) -> list[Violation]:
    """Check the manifest (including vendored bytes on disk) and every recipe.

    ``recipes_dir`` holds ``*.yaml`` recipe files and nothing else: any other entry (a
    ``.yml`` recipe, a directory) would never be checked, so it is itself a violation.
    """
    try:
        manifest = load_yaml(manifest_path)
    except (OSError, yaml.YAMLError) as error:
        return [Violation("yaml", str(manifest_path), str(error))]
    violations = check_manifest(manifest, base_dir=manifest_path.parent)
    recipes: dict[str, Any] = {}
    for path in sorted(recipes_dir.iterdir()):
        if not (path.is_file() and path.suffix == ".yaml"):
            violations.append(
                Violation(
                    "recipe_schema",
                    f"recipes/{path.name}",
                    "recipes/ holds *.yaml recipe files only; this entry would never be checked",
                )
            )
            continue
        try:
            recipes[path.stem] = load_yaml(path)
        except (OSError, yaml.YAMLError) as error:
            violations.append(Violation("yaml", str(path), str(error)))
    if isinstance(manifest, Mapping):
        violations.extend(check_recipes(recipes, manifest))
    return violations


# --- provenance ------------------------------------------------------------------------


def canonical_json(document: Any) -> str:
    """The exact text :func:`recipe_hash` hashes (UTF-8 encoded); see the module docstring.

    ``json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False)`` -- not RFC 8785.
    """
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def recipe_hash(recipe: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``recipe`` plus the manifest entries it references.

    The referenced entries are the recipe's runners (its ``pipeline`` and every post-step
    runner), its models (its own and every post step's), and the code components those
    runners execute -- so a new weight file, a library upgrade or a parameter change each
    yield a new hash. Licence-only fields (:data:`HASH_EXCLUDED_COMPONENT_FIELDS`) and the
    recipe's description are left out: the hash identifies what generates the pixels, and
    licence provenance travels in the asset's sidecar. Raises ``KeyError`` for a reference
    the manifest does not contain; run :func:`check_recipe` first.
    """
    components = {
        component["id"]: component
        for component in manifest["components"]
        if isinstance(component, Mapping)
    }
    runners = {runner["id"]: runner for runner in manifest["runners"]}
    post = recipe.get("post") or []
    runner_ids = sorted({recipe["pipeline"], *(step["runner"] for step in post)})
    model_ids = [*recipe["models"], *(model for step in post for model in step.get("models") or [])]
    code_ids = [code_id for runner_id in runner_ids for code_id in runners[runner_id]["code"]]
    referenced = {
        "runners": {runner_id: runners[runner_id] for runner_id in runner_ids},
        "components": {
            component_id: {
                key: value
                for key, value in components[component_id].items()
                if key not in HASH_EXCLUDED_COMPONENT_FIELDS
            }
            for component_id in sorted({*model_ids, *code_ids})
        },
    }
    document = {
        "recipe": {
            key: value for key, value in recipe.items() if key not in HASH_EXCLUDED_RECIPE_FIELDS
        },
        "manifest": referenced,
    }
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def install_footprint(
    manifest: Mapping[str, Any], component_ids: Iterable[str] | None = None
) -> Footprint:
    """Bytes the selected components need on a content-addressed store.

    ``component_ids`` defaults to EVERY ``model`` component -- the generators, the matting,
    upscaling and restoration weights AND the judge-VLM pair -- not the generators alone;
    name ids to size a subset. Each distinct sha256 counts once, which is what a store that
    keeps files by sha256 needs. A stock Hugging Face cache keeps one blob store per repo
    (``models--<org>--<name>/blobs``), so there a file two repos share (Qwen-Image-2512's
    and Qwen-Image-Edit-2511's text encoder) is stored twice: size that layout from
    ``listed_bytes`` instead. A sha256 listed with two sizes is refused by
    :func:`check_manifest`; here it raises ``ValueError`` rather than answer by list order.
    """
    components = [
        component for component in manifest["components"] if isinstance(component, Mapping)
    ]
    if component_ids is None:
        selected = [component for component in components if component.get("kind") == "model"]
    else:
        wanted = list(component_ids)
        by_id = {component["id"]: component for component in components}
        missing = [component_id for component_id in wanted if component_id not in by_id]
        if missing:
            raise KeyError(f"not in the manifest: {missing}")
        selected = [by_id[component_id] for component_id in wanted]
    sizes: dict[str, int] = {}
    owners: dict[str, list[str]] = {}
    listed_bytes = listed_files = 0
    for component in selected:
        for entry in component["files"]:
            sha, size = entry["sha256"], entry["bytes"]
            if sizes.setdefault(sha, size) != size:
                raise ValueError(f"sha256 {sha} is listed as {sizes[sha]} and {size} bytes")
            listed_bytes += size
            listed_files += 1
            owners.setdefault(sha, [])
            if component["id"] not in owners[sha]:
                owners[sha].append(component["id"])
    return Footprint(
        unique_bytes=sum(sizes.values()),
        unique_files=len(sizes),
        listed_bytes=listed_bytes,
        listed_files=listed_files,
        shared={sha: tuple(ids) for sha, ids in owners.items() if len(ids) > 1},
    )


# --- command line ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qontinui_imagegen.manifest",
        description=(
            "Check the image provider's licence manifest and recipes "
            f"(decision record a5ec7670 v{DECISION_RECORD_VERSION})."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--recipes", type=Path, default=RECIPES_DIR)
    parser.add_argument("--hashes", action="store_true", help="print each recipe's provenance hash")
    parser.add_argument(
        "--footprint",
        nargs="*",
        metavar="COMPONENT",
        help=(
            "print the bytes the named components need, each sha256 counted once (a "
            "content-addressed store). With no ids: EVERY model component, the judge-VLM "
            "pair included -- name the generators to size them alone. A stock Hugging Face "
            "cache stores a file two repos share twice: size it from the listed bytes."
        ),
    )
    args = parser.parse_args(argv)
    violations = check(args.manifest, args.recipes)
    for violation in violations:
        print(violation)
    if violations:
        print(f"{len(violations)} violation(s)", file=sys.stderr)
        return 1
    manifest = load_yaml(args.manifest)
    if args.hashes:
        for name, recipe in load_recipes(args.recipes).items():
            print(f"{name} v{recipe['version']} {recipe_hash(recipe, manifest)}")
    if args.footprint is not None:
        footprint = install_footprint(manifest, args.footprint or None)
        print(
            f"unique {footprint.unique_bytes} bytes in {footprint.unique_files} files "
            f"(listed {footprint.listed_bytes} bytes in {footprint.listed_files} files; "
            f"{len(footprint.shared)} sha256 shared)"
        )
    print("manifest and recipes admitted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
