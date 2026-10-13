"""Make the declared owner win each top-level package that several locked distributions install.

Some distributions in poetry.lock install the SAME top-level package:

* ``Xlib/`` - ``python-xlib`` (the maintained library; ``pynput`` requires
  ``python-xlib>=0.17``) and ``python3-xlib`` 0.15, an abandoned 2014 fork that
  ``pyautogui`` and ``mouseinfo`` still require on Linux.
* ``cv2/`` - ``opencv-python-headless`` (what qontinui declares) and
  ``opencv-python``, the GUI build ``supervision`` (and the ``ml`` extra's
  ``ultralytics``) require: the same OpenCV version, different binaries. The
  ``omniparser`` extra adds a third, ``opencv-contrib-python`` 4.10.0.84, which
  paddlex (through paddleocr) pins exactly. Making the headless build win
  overrides that pin; paddlex 3.5.2 references no contrib-only ``cv2`` module.

Poetry installs a lock-version 2.1 lock (this repo's) as one group on a thread
pool, and its wheel writer only warns ("Installing ... over existing file")
before it opens a path for writing. So the distributions that share a package
write the same files at the same time, and the result varies from run to run:

* One distribution wins every file. That imports fine.
* A whole-file mix: every file is intact, but some come from each distribution.
  It is the more common failure. For Xlib it is ``AttributeError: module
  'Xlib.X' has no attribute 'FamilyServerInterpreted'``, python-xlib's code
  next to the fork's ``X.py``.
* A torn file: both writers had it open, so it holds the later writer's bytes
  followed by the tail of the earlier, longer copy. For
  ``Xlib/protocol/__init__.py`` that is ``IndentationError: unexpected indent``
  at line 26. For ``cv2/cv2.abi3.so`` it is a corrupt shared object, and
  ``import cv2`` segfaults or aborts.

pynput reports either Xlib failure as ``ImportError: this platform is not
supported: (...)``, raised through ``qontinui.__init__`` -> healing -> hal ->
keyboard_operations, so every test module that imports qontinui errors at
collection.

The odds depend on timing. On a cold cache Poetry first builds python3-xlib from
its sdist (it publishes no wheel). That delays its writes past python-xlib's,
the fork cleanly replaces every shared file, and everything imports: qontinui's
CI passed that way on 2026-10-07 (run 37704199471). With the artifact cache
warm, the built wheel is reused and both installs start within ~50 ms of each
other. The tests job of run 38394722804 (2026-10-13) then failed on both
attempts, once with each Xlib failure. A cached virtualenv is reused as it is,
damage included. A minimal local reproduction (just the two distributions of a
pair, 20 fresh ``poetry install`` runs with Poetry 2.5.1) left ``import Xlib``
broken 6 times and ``import cv2`` failing 11 times.

No install-time setting prevents this. Poetry cannot exclude or override a
transitive dependency, and ``installer.parallel = false`` installs the group in
name order, so the abandoned fork would then win every time. So the repair
happens after the install. For each package below, this script hashes every file
the owner's RECORD lists. If any byte differs, it reinstalls exactly the locked
owner (``pip install --force-reinstall --no-deps --require-hashes`` with
poetry.lock's hashes) and verifies again. It installs the wheel Poetry already
downloaded into its artifact cache when one matches the lock. It asks the package
index when none does. It also asks the index, once and still hash-pinned, when
pip cannot install the cached wheel, as with another platform's wheel in a
shared cache. On a healthy environment it changes nothing. Files that only a
rival ships (opencv-python's Qt plugin under ``cv2/qt/``) stay in place. Nothing
loads them; cv2 only imports ``cv2.qt`` as an empty namespace package.

Run it after every ``poetry install`` (CI does, in every job that installs):

    poetry run python scripts/enforce_package_owners.py          # verify, repair
    poetry run python scripts/enforce_package_owners.py --check  # verify only

A project that installs qontinui as a dependency inherits these collisions. It can
run this script from its own root, with ``--lock poetry.lock`` naming its own lock.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_LOCK = Path(__file__).resolve().parent.parent / "poetry.lock"


@dataclass(frozen=True)
class Contest:
    """A top-level package that more than one locked distribution installs."""

    package: str
    owner: str
    rivals: tuple[str, ...]


CONTESTS: tuple[Contest, ...] = (
    # pyautogui and mouseinfo only use Xlib.display, X, XK and ext.xtest, which
    # python-xlib provides unchanged; it also ships nine extensions the fork lacks.
    Contest(package="Xlib", owner="python-xlib", rivals=("python3-xlib",)),
    # qontinui declares the headless build and calls no highgui (window) function.
    # opencv-contrib-python comes only with the omniparser extra (paddleocr ->
    # paddlex[ocr-core] pins 4.10.0.84); the headless build wins over that pin too.
    Contest(
        package="cv2",
        owner="opencv-python-headless",
        rivals=("opencv-python", "opencv-contrib-python"),
    ),
)


def canonical(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _file_digest(path: Path, algorithm: str) -> bytes:
    """The raw ``algorithm`` digest of a file, read in chunks."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def record_rows(dist: metadata.Distribution) -> list[tuple[PurePosixPath, str]]:
    """``(path, hash field)`` for every row of ``dist``'s RECORD.

    Read directly: ``Distribution.files`` drops rows whose file is missing
    (Python 3.12+), and a missing file is damage this script has to see.
    """
    text = dist.read_text("RECORD") or ""
    rows = csv.reader(text.splitlines())
    return [(PurePosixPath(row[0]), row[1] if len(row) > 1 else "") for row in rows if row]


def damaged_files(dist: metadata.Distribution, package: str) -> tuple[int, list[str]]:
    """Hash every file ``dist``'s RECORD lists under ``package/``.

    Returns how many files were checked and which ones are missing or differ
    from the recorded hash. Bytecode is skipped: it is derived, and Python
    revalidates it against its source on import.
    """
    checked = 0
    damaged: list[str] = []
    for path, hash_field in record_rows(dist):
        algorithm, _, expected = hash_field.partition("=")
        if path.parts[:1] != (package,) or not expected or path.suffix == ".pyc":
            continue
        checked += 1
        try:
            digest = _file_digest(Path(str(dist.locate_file(path))), algorithm)
            actual: str | None = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        except FileNotFoundError:
            actual = None
        if actual != expected:
            damaged.append(str(path))
    return checked, damaged


def other_writers(package: str, owner: str) -> list[str]:
    """Installed distributions besides ``owner`` whose RECORD lists files under ``package/``.

    A dist-info without a ``Name`` (half-written, or edited by hand) is skipped: it
    cannot be named, and it must not fail the check of an intact owner.
    """
    writers = set()
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or canonical(name) == canonical(owner):
            continue
        if any(path.parts[:1] == (package,) for path, _ in record_rows(dist)):
            writers.add(f"{name} {dist.metadata.get('Version')}")
    return sorted(writers)


class LockError(Exception):
    """poetry.lock cannot give the owner's pin. The message starts with the lock's path."""


# The only algorithms pip accepts in --hash; anything else in a lock cannot be used.
PIP_HASH_ALGORITHMS = ("sha256", "sha384", "sha512")


def locked_entry(lock_file: Path, owner: str, installed_version: str) -> dict[str, Any]:
    """The poetry.lock entry for ``owner`` at the installed version, its artifacts checked.

    Refuses a lock that pins another version than the installed one: reinstalling
    from it would change the environment rather than repair it. A lock that cannot
    be read or parsed, or whose entry lacks what the repair reads, is refused too.
    Each refusal is a LockError naming the lock.
    """
    try:
        lock = tomllib.loads(lock_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise LockError(f"{lock_file}: cannot be read ({error.strerror or error})") from error
    except UnicodeDecodeError as error:
        raise LockError(f"{lock_file}: is not UTF-8 text ({error.reason})") from error
    except (tomllib.TOMLDecodeError, RecursionError) as error:
        # tomllib recurses once per nesting level, and some releases (3.12.14 and 3.12.15,
        # not 3.13.5) refuse a key of more than 1000 dotted parts with RecursionError.
        raise LockError(f"{lock_file}: is not valid TOML ({error})") from error
    if "package" not in lock:
        raise LockError(f"{lock_file}: has no [[package]] table, so it locks nothing")
    try:
        packages = lock["package"]
        entries = [entry for entry in packages if canonical(entry["name"]) == canonical(owner)]
        matching = [entry for entry in entries if entry["version"] == installed_version]
        if len(matching) > 1:
            raise LockError(
                f"{lock_file}: several matching lock entries for {owner} {installed_version} "
                f"({len(matching)})"
            )
        if not matching:
            locked = ", ".join(sorted(str(entry["version"]) for entry in entries))
            raise LockError(
                f"{lock_file}: no matching lock entry for {owner} {installed_version} "
                f"(it locks {locked or 'none'}); pass the poetry.lock this environment "
                "was installed from (--lock)"
            )
        entry = matching[0]
        where = f"{lock_file}: the {owner} {installed_version} entry"
        if not isinstance(entry["files"], list):
            raise LockError(f"{where} has files that are not a list: {entry['files']!r}")
        for artifact in entry["files"]:
            if not (isinstance(artifact, dict) and "file" in artifact and "hash" in artifact):
                raise LockError(
                    f"{where} lists an artifact that lacks a file or a hash: {artifact!r}"
                )
            algorithm, _, digest = str(artifact["hash"]).partition(":")
            if algorithm not in PIP_HASH_ALGORITHMS:
                raise LockError(
                    f"{where} locks {artifact['file']} with hash {artifact['hash']!r}; "
                    f"pip accepts only {', '.join(PIP_HASH_ALGORITHMS)}"
                )
            if not digest:
                raise LockError(
                    f"{where} locks {artifact['file']} with an empty {algorithm} digest"
                )
    except (KeyError, TypeError, AttributeError) as error:
        raise LockError(
            f"{lock_file}: a [[package]] entry is malformed ({type(error).__name__}: {error})"
        ) from error
    return entry


def requirement_line(entry: dict[str, Any]) -> str:
    """A pip requirement pinning the entry's version and every locked artifact hash."""
    hashes = " ".join(f"--hash={artifact['hash']}" for artifact in entry["files"])
    return f"{entry['name']}=={entry['version']} {hashes}"


def poetry_artifact_cache() -> Path:
    """Poetry's artifact cache: ``$POETRY_CACHE_DIR/artifacts``, else its default.

    The default is platformdirs' user cache path for "pypoetry", as Poetry uses.
    A cache-dir set with ``poetry config`` is not read; the repair then falls back
    to the package index, and says so.
    """
    override = os.environ.get("POETRY_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(local) / "pypoetry" / "Cache"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "pypoetry"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pypoetry"
    return root / "artifacts"


def cached_wheels(entry: dict[str, Any], cache: Path) -> list[Path]:
    """Wheels under ``cache`` that are one of the entry's locked artifacts, by name AND hash."""
    wanted = {
        str(artifact["file"]): str(artifact["hash"])
        for artifact in entry["files"]
        if str(artifact["file"]).endswith(".whl")
    }
    found: list[Path] = []
    if not wanted or not cache.is_dir():
        return found
    for directory, _, names in os.walk(cache):
        for name in names:
            if name in wanted:
                algorithm, _, expected = wanted[name].partition(":")
                path = Path(directory, name)
                if _file_digest(path, algorithm).hex() == expected:
                    found.append(path)
    return sorted(found)


def pip_install_command(requirements: Path, find_links: Sequence[Path]) -> list[str]:
    """pip force-reinstalling the hash-pinned requirement in ``requirements``, no deps.

    With ``find_links`` pip installs from those directories only (``--no-index``).
    """
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--force-reinstall",
        "--no-deps",
        "--require-hashes",
    ]
    if find_links:
        command.append("--no-index")
        for directory in find_links:
            command += ["--find-links", str(directory)]
    return command + ["-r", str(requirements)]


def reinstall(requirement: str, find_links: Sequence[Path]) -> None:
    """Force-reinstall one hash-pinned distribution, leaving its dependencies alone."""
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(f"pip is not installed in {sys.prefix}; it is needed to reinstall")
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "owner.txt"
        requirements.write_text(requirement + "\n", encoding="utf-8")
        subprocess.run(
            pip_install_command(requirements, find_links),
            check=True,
        )


def enforce(contest: Contest, lock_file: Path, check_only: bool) -> bool:
    """Make every file the owner's RECORD lists under the package hold the owner's bytes.

    True when they do, or when the owner is not installed here. Files that only a
    rival ships are left in place.
    """
    try:
        dist = metadata.distribution(contest.owner)
    except metadata.PackageNotFoundError:
        print(f"{contest.package}/: {contest.owner} is not installed here - nothing to enforce")
        return True
    version = dist.version
    writers = other_writers(contest.package, contest.owner)
    label = (
        f"{contest.package}/ (owner {contest.owner} {version}; "
        f"also written by: {', '.join(writers) or 'nothing else installed'})"
    )
    checked, damaged = damaged_files(dist, contest.package)
    if checked == 0:
        print(f"{label}: the owner's RECORD lists no hashed file under {contest.package}/")
        return False
    if not damaged:
        print(f"{label}: intact - all {checked} files match the owner's RECORD")
        return True
    print(
        f"{label}: {len(damaged)} of {checked} files differ from the owner's RECORD, "
        f"e.g. {', '.join(damaged[:3])}"
    )
    if check_only:
        return False
    try:
        entry = locked_entry(lock_file, contest.owner, version)
    except LockError as error:
        print(f"{label}: {error}")
        return False
    requirement = requirement_line(entry)
    cache = poetry_artifact_cache()
    try:
        wheels = cached_wheels(entry, cache)
        from_cache = False
        if wheels:
            print(
                f"{label}: reinstalling from Poetry's artifact cache: {', '.join(map(str, wheels))}"
            )
            try:
                reinstall(requirement, sorted({wheel.parent for wheel in wheels}))
                from_cache = True
            except subprocess.CalledProcessError:
                print(
                    f"{label}: pip could not install the cached wheel (a cache shared across "
                    "platforms can hold another platform's); retrying once from the package "
                    "index, still hash-pinned"
                )
        else:
            print(f"{label}: reinstalling from the package index (no locked wheel under {cache})")
        if not from_cache:
            reinstall(requirement, [])
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"{label}: reinstalling {contest.owner} failed: {error}")
        return False
    source = "Poetry's artifact cache" if from_cache else "the package index"
    importlib.invalidate_caches()
    dist = metadata.distribution(contest.owner)
    checked, damaged = damaged_files(dist, contest.package)
    if checked == 0 or damaged:
        print(f"{label}: still damaged after reinstalling: {', '.join(damaged[:5])}")
        return False
    print(
        f"{label}: repaired - reinstalled {contest.owner} {dist.version} from {source}; "
        f"all {checked} files match its RECORD"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify (and repair) the owners of top-level packages that several "
        "locked distributions install. See the module docstring for why."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify; exit 1 if an owner's files are damaged (changes nothing)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help="poetry.lock this environment was installed from; the owner is reinstalled from "
        "its pin and hashes (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    # Keep this script's lines in order with pip's output in a piped CI log.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    print(f"enforce_package_owners: checking {sys.prefix} (Python {sys.version.split()[0]})")
    results = [enforce(contest, args.lock, args.check) for contest in CONTESTS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
