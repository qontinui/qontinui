"""Where the weights live, how they are verified, and the verify-on-download install.

Layout under the weights root (PD11: a mounted volume, relocatable to whichever volume has
room -- D1.5, rule R-disk):

* a Hugging Face component resolves in a stock HF hub cache,
  ``hub/models--<org>--<name>/snapshots/<revision>/<path>`` -- the layout
  ``huggingface_hub`` writes, and the one Phase 0 item 5 already filled on spaceship's D:;
* a GitHub release asset resolves at ``github/<owner>--<name>/<tag>/<path>``.

Verification (plan Phase 10, "Weight verification"): the full sha256 runs once, at install
(:func:`install`) or in a bulk-window ``verify --full``; each writes a STAMP per file
(sha256, size, mtime_ns, inode). Before a model's first load in a process the facade checks
that model's files against their stamps and re-hashes only a file whose stat changed or
that has no stamp. A file whose bytes are not the manifest's is never served. The reason
for stamps: llama-swap restarts the facade after every ``ttl`` unload, and re-hashing the
~132 GB of generators at every start would hold each cold request for minutes and can
exceed llama-swap's ``healthCheckTimeout``.

Run ``python -m qontinui_imagegen.weights --help`` from docker/imagegen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qontinui_imagegen.manifest import MANIFEST_PATH, load_yaml

HASH_CHUNK = 8 * 1024 * 1024
#: File types that carry weights. One of these beside a model's pinned files is refused:
#: a loader that globs a folder (diffusers, safetensors shards) would pick it up unverified.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"})
STAMPS_SCHEMA = 1


class WeightsError(Exception):
    """A model's weights are missing or are not the bytes the manifest pins."""

    def __init__(self, model: str, problems: Sequence[str]) -> None:
        super().__init__(f"{model}: " + "; ".join(problems))
        self.model = model
        self.problems = list(problems)


def component_dir(weights_root: Path, component: Mapping[str, Any]) -> Path:
    """The directory a model component's ``files[].path`` are relative to."""
    source = component.get("source") or {}
    kind = source.get("type")
    if kind == "huggingface":
        org, name = str(source["repo"]).split("/", 1)
        return (
            weights_root / "hub" / f"models--{org}--{name}" / "snapshots" / str(source["revision"])
        )
    if kind == "github_release":
        owner, name = str(source["repo"]).split("/", 1)
        return weights_root / "github" / f"{owner}--{name}" / str(source["tag"])
    raise ValueError(f"component {component.get('id')!r} has no installable source ({kind!r})")


def sha256_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return digest.hexdigest()


def _stat_record(path: Path) -> dict[str, int]:
    stat = path.stat()  # follows the HF cache's snapshot symlinks to the blob
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


@dataclass
class Stamps:
    """The per-file stamps file, keyed by the file's path relative to the weights root."""

    path: Path
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    writable: bool = True

    @classmethod
    def read(cls, path: Path) -> Stamps:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path)
        except (OSError, ValueError):
            # An unreadable stamps file vouches for nothing: every file is re-hashed.
            return cls(path)
        if not isinstance(document, dict) or document.get("schema") != STAMPS_SCHEMA:
            return cls(path)
        files = document.get("files")
        return cls(path, dict(files) if isinstance(files, dict) else {})

    def write(self) -> None:
        """Atomically replace the stamps file. A read-only volume keeps them in memory."""
        if not self.writable:
            return
        document = {"schema": STAMPS_SCHEMA, "files": dict(sorted(self.files.items()))}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".stamps-", dir=self.path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=1)
            os.replace(temp, self.path)
        except OSError:
            self.writable = False


@dataclass
class VerifyProgress:
    """What /health reports while a verification runs (llama-swap's loading body shape)."""

    total_bytes: int = 0
    done_bytes: int = 0
    active: int = 0

    @property
    def fraction(self) -> float:
        return 1.0 if not self.total_bytes else min(1.0, self.done_bytes / self.total_bytes)


class WeightVerifier:
    """Checks a model's files against their stamps once per process, re-hashing on change."""

    def __init__(
        self,
        weights_root: Path,
        components: Mapping[str, Mapping[str, Any]],
        stamps_path: Path,
        stamp_fields: Sequence[str],
        *,
        hasher: Callable[[Path, Callable[[int], None] | None], str] = sha256_file,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.weights_root = weights_root
        self.components = components
        self.stamps = Stamps.read(stamps_path)
        self.stamp_fields = tuple(stamp_fields)
        self.hasher = hasher
        self.clock = clock
        self.verified: set[str] = set()
        self.progress = VerifyProgress()
        self._lock = threading.Lock()

    def _key(self, path: Path) -> str:
        return path.relative_to(self.weights_root).as_posix()

    def _unchanged(
        self, stamp: Mapping[str, Any] | None, current: Mapping[str, int], sha: str
    ) -> bool:
        return (
            stamp is not None
            and stamp.get("sha256") == sha
            and all(stamp.get(name) == current[name] for name in self.stamp_fields)
        )

    def verify_model(self, model_id: str, *, full: bool = False) -> None:
        """Raise :class:`WeightsError` unless every file of ``model_id`` is the pinned bytes.

        Runs once per process per model unless ``full`` (re-hash everything, stamps ignored).
        """
        with self._lock:
            if model_id in self.verified and not full:
                return
            component = self.components.get(model_id)
            if component is None:
                raise WeightsError(model_id, ["not a model in the manifest"])
            base = component_dir(self.weights_root, component)
            problems: list[str] = []
            pending: list[tuple[Path, Mapping[str, Any], dict[str, int]]] = []
            for entry in component["files"]:
                path = base / str(entry["path"])
                try:
                    current = _stat_record(path)
                except FileNotFoundError:
                    problems.append(f"{entry['path']} is not installed")
                    continue
                if current["size"] != entry["bytes"]:
                    problems.append(
                        f"{entry['path']} is {current['size']} bytes, the manifest pins {entry['bytes']}"
                    )
                    continue
                stamp = self.stamps.files.get(self._key(path))
                if full or not self._unchanged(stamp, current, str(entry["sha256"])):
                    pending.append((path, entry, current))
            listed = {base / str(entry["path"]) for entry in component["files"]}
            if base.is_dir():
                for path in sorted(base.rglob("*")):
                    if path.suffix in WEIGHT_SUFFIXES and path not in listed:
                        problems.append(
                            f"{path.relative_to(base).as_posix()} is a weight file the "
                            "manifest does not pin; remove it (unpinned bytes are never loaded)"
                        )
            if problems:
                raise WeightsError(model_id, problems)
            self.progress = VerifyProgress(
                total_bytes=sum(int(entry["bytes"]) for _, entry, _ in pending), active=1
            )

            def advance(count: int) -> None:
                self.progress.done_bytes += count

            try:
                for path, entry, _current in pending:
                    actual = self.hasher(path, advance)
                    if actual != entry["sha256"]:
                        problems.append(
                            f"{entry['path']} hashes to {actual}, the manifest pins {entry['sha256']}"
                        )
                        self.stamps.files.pop(self._key(path), None)
                        continue
                    self.stamps.files[self._key(path)] = {
                        "sha256": actual,
                        **_stat_record(path),
                        "verified_at": datetime.fromtimestamp(self.clock(), timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
            finally:
                self.progress.active = 0
                if pending:
                    self.stamps.write()
            if problems:
                raise WeightsError(model_id, problems)
            self.verified.add(model_id)


# --- install ---------------------------------------------------------------------------------


Fetcher = Callable[[Mapping[str, Any], Mapping[str, Any], Path], Path]


def _fetch_huggingface(component: Mapping[str, Any], entry: Mapping[str, Any], root: Path) -> Path:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise OSError(
            "the install fetches from the Hugging Face Hub; run it without HF_HUB_OFFLINE=1 "
            "(the facade itself always runs offline)"
        )
    from huggingface_hub import hf_hub_download

    source = component["source"]
    return Path(
        hf_hub_download(
            repo_id=source["repo"],
            filename=entry["path"],
            revision=source["revision"],
            cache_dir=str(root / "hub"),
        )
    )


def _fetch_github_release(
    component: Mapping[str, Any], entry: Mapping[str, Any], root: Path
) -> Path:
    source = component["source"]
    target = component_dir(root, component) / str(entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{source['repo']}/releases/download/{source['tag']}/{entry['path']}"
    partial = target.with_name(target.name + ".partial")
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, HASH_CHUNK)
    os.replace(partial, target)
    return target


def default_fetcher(component: Mapping[str, Any], entry: Mapping[str, Any], root: Path) -> Path:
    kind = component["source"]["type"]
    if kind == "huggingface":
        return _fetch_huggingface(component, entry, root)
    if kind == "github_release":
        return _fetch_github_release(component, entry, root)
    raise ValueError(f"{component['id']}: no installer for source type {kind!r}")


@dataclass(frozen=True)
class InstallReport:
    installed: tuple[str, ...]
    already_present: tuple[str, ...]
    bytes_fetched: int


def plan_install(
    weights_root: Path, components: Iterable[Mapping[str, Any]], verifier: WeightVerifier
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], list[str]]:
    """Which files must be fetched, and which are already the pinned bytes."""
    missing: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    present: list[str] = []
    for component in components:
        try:
            verifier.verify_model(str(component["id"]))
            present.append(str(component["id"]))
            continue
        except WeightsError:
            pass
        base = component_dir(weights_root, component)
        for entry in component["files"]:
            path = base / str(entry["path"])
            stamp = verifier.stamps.files.get(verifier._key(path)) if path.exists() else None
            if stamp is not None and stamp.get("sha256") == entry["sha256"]:
                continue
            missing.append((component, entry))
    return missing, present


def install(
    weights_root: Path,
    components: Sequence[Mapping[str, Any]],
    verifier: WeightVerifier,
    *,
    fetcher: Fetcher = default_fetcher,
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
    log: Callable[[str], None] = print,
) -> InstallReport:
    """Fetch every missing file, verify its sha256, stamp it; refuse on too little space.

    The space check counts the LISTED bytes of what is missing: a stock HF cache keeps one
    blob store per repo, so a file two repos share is stored twice (manifest.yaml's sizing
    note). Nothing is fetched when the volume cannot hold all of it.
    """
    weights_root.mkdir(parents=True, exist_ok=True)
    missing, present = plan_install(weights_root, components, verifier)
    needed = sum(int(entry["bytes"]) for _, entry in missing)
    free = free_bytes(weights_root)
    if needed > free:
        raise OSError(
            f"refusing to install: {needed} bytes are missing and {weights_root} has {free} free "
            "(rule R-disk: name a volume with room)"
        )
    fetched = 0
    for component, entry in missing:
        log(f"fetching {component['id']}/{entry['path']} ({entry['bytes']} bytes)")
        path = fetcher(component, entry, weights_root)
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            path.unlink(missing_ok=True)
            raise WeightsError(
                str(component["id"]),
                [f"{entry['path']} downloaded as {actual}, the manifest pins {entry['sha256']}"],
            )
        verifier.stamps.files[
            verifier._key(component_dir(weights_root, component) / str(entry["path"]))
        ] = {
            "sha256": actual,
            **_stat_record(path),
            "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        verifier.stamps.write()
        fetched += int(entry["bytes"])
    installed = []
    for component in components:
        verifier.verified.discard(str(component["id"]))
        verifier.verify_model(str(component["id"]))
        if str(component["id"]) not in present:
            installed.append(str(component["id"]))
    return InstallReport(tuple(installed), tuple(present), fetched)


def main(argv: Sequence[str] | None = None) -> int:
    from qontinui_imagegen.config import load_config

    parser = argparse.ArgumentParser(
        prog="python -m qontinui_imagegen.weights",
        description="Install (verify-on-download) or verify the image provider's weights.",
    )
    parser.add_argument("command", choices=("install", "verify", "footprint"))
    parser.add_argument("models", nargs="+", help="manifest model ids")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--full", action="store_true", help="verify: re-hash every file, ignoring stamps"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest = load_yaml(args.manifest)
    components = {str(c["id"]): c for c in manifest["components"]}
    unknown = [model for model in args.models if components.get(model, {}).get("kind") != "model"]
    if unknown:
        print(f"not model components in the manifest: {unknown}", file=sys.stderr)
        return 2
    selected = [components[model] for model in args.models]
    verifier = WeightVerifier(
        config.weights_root,
        components,
        config.weights_root / config.stamps_file,
        config.stamp_fields,
    )
    if args.command == "footprint":
        listed = sum(int(entry["bytes"]) for c in selected for entry in c["files"])
        print(f"{listed} bytes listed for {args.models} (a stock HF cache stores each repo's copy)")
        return 0
    if args.command == "install":
        try:
            report = install(config.weights_root, selected, verifier)
        except (OSError, WeightsError) as error:
            print(str(error), file=sys.stderr)
            return 3
        print(
            f"installed {list(report.installed)}; already present {list(report.already_present)}; "
            f"fetched {report.bytes_fetched} bytes"
        )
        return 0
    failures = 0
    for model in args.models:
        try:
            verifier.verify_model(model, full=args.full)
            print(f"{model}: verified")
        except WeightsError as error:
            failures += 1
            print(str(error), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
