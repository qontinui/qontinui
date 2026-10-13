"""Make the declared owner win each top-level package that two locked distributions install.

Two pairs of distributions in poetry.lock install the SAME top-level package:

* ``Xlib/`` - ``python-xlib`` (the maintained library; ``pynput`` requires
  ``python-xlib>=0.17``) and ``python3-xlib`` 0.15, an abandoned 2014 fork that
  ``pyautogui`` and ``mouseinfo`` still require on Linux.
* ``cv2/`` - ``opencv-python-headless`` (what qontinui declares) and
  ``opencv-python``, the GUI build ``supervision`` (and the ``ml`` extra's
  ``ultralytics``) require: the same OpenCV version, different binaries.

Poetry installs each dependency-depth group on a thread pool, and its wheel
writer only warns ("Installing ... over existing file") before opening the path
for writing. When both distributions of a pair land in one group, two threads
write the same files at once, and a file both have open ends up holding the
later writer's bytes followed by the tail of the earlier, longer copy. For
``Xlib/protocol/__init__.py`` that is ``IndentationError: unexpected indent``
at line 26; pynput reports it as ``ImportError: this platform is not
supported``, raised through ``qontinui.__init__`` -> healing -> hal ->
keyboard_operations, so every test module that imports qontinui errors at
collection. For ``cv2/cv2.abi3.so`` it is a corrupt shared object, and
``import cv2`` segfaults or aborts.

Whether the writes overlap is timing, which is why it looked like a flake:

* On a cold cache poetry first builds python3-xlib from its sdist (it publishes
  no wheel). That delays its writes past python-xlib's, the fork cleanly
  replaces every shared file, and everything imports. qontinui's CI passed this
  way on 2026-10-07 (run 37704199471).
* With poetry's artifact cache warm, the built wheel is reused, both installs
  start within ~50 ms of each other, and the writes interleave. The tests job
  then failed on both attempts of run 38394722804 (2026-10-13).
* A cached virtualenv is reused as it is, damage included.

A minimal local reproduction (just the two distributions of a pair, 20 fresh
``poetry install`` runs with Poetry 2.5.1) left ``import Xlib`` broken 6 times
and ``import cv2`` failing 11 times.

No install-time setting prevents this. Poetry cannot exclude or override a
transitive dependency. ``installer.parallel = false`` would make the order
deterministic, but operations in a group sort by name, so the abandoned fork
would then win every time. So the repair happens after the install. For each
package below, this script hashes every file the owner's RECORD lists. If any
byte differs, it reinstalls exactly the locked owner (``pip install
--force-reinstall --no-deps`` with poetry.lock's hashes) and verifies again. On
a healthy environment it changes nothing.

Run it after every ``poetry install`` (CI does, in every job that installs):

    poetry run python scripts/enforce_package_owners.py          # verify, repair
    poetry run python scripts/enforce_package_owners.py --check  # verify only
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.util
import io
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath

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
    Contest(package="cv2", owner="opencv-python-headless", rivals=("opencv-python",)),
)


def canonical(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _digest(path: Path, algorithm: str) -> str:
    """A file's hash in RECORD form: urlsafe base64, padding stripped."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


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
            actual: str | None = _digest(Path(str(dist.locate_file(path))), algorithm)
        except FileNotFoundError:
            actual = None
        if actual != expected:
            damaged.append(str(path))
    return checked, damaged


def other_writers(package: str, owner: str) -> list[str]:
    """Installed distributions besides ``owner`` whose RECORD lists files under ``package/``."""
    writers = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if canonical(name) == canonical(owner):
            continue
        if any(path.parts[:1] == (package,) for path, _ in record_rows(dist)):
            writers.add(f"{name} {dist.version}")
    return sorted(writers)


def locked_requirement(lock_file: Path, owner: str, installed_version: str) -> str:
    """The owner's pin from poetry.lock, carrying every locked artifact hash."""
    packages = tomllib.loads(lock_file.read_text(encoding="utf-8"))["package"]
    entries = [entry for entry in packages if canonical(entry["name"]) == canonical(owner)]
    if len(entries) > 1:
        entries = [entry for entry in entries if entry["version"] == installed_version]
    if len(entries) != 1:
        raise LookupError(f"cannot pick one locked {owner} from {lock_file}")
    entry = entries[0]
    hashes = " ".join(f"--hash={artifact['hash']}" for artifact in entry["files"])
    return f"{entry['name']}=={entry['version']} {hashes}"


def reinstall(requirement: str) -> None:
    """Force-reinstall one hash-pinned distribution, leaving its dependencies alone."""
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(f"pip is not installed in {sys.prefix}; it is needed to reinstall")
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "owner.txt"
        requirements.write_text(requirement + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--force-reinstall",
                "--no-deps",
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            check=True,
        )


def enforce(contest: Contest, lock_file: Path, check_only: bool) -> bool:
    """Leave ``contest.package`` holding exactly its owner's files; True when it does."""
    try:
        dist = metadata.distribution(contest.owner)
    except metadata.PackageNotFoundError:
        print(f"{contest.package}/: {contest.owner} is not installed here - nothing to enforce")
        return True
    writers = other_writers(contest.package, contest.owner)
    label = (
        f"{contest.package}/ (owner {contest.owner} {dist.version}; "
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
        reinstall(locked_requirement(lock_file, contest.owner, dist.version))
    except (LookupError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"{label}: reinstalling {contest.owner} failed: {error}")
        return False
    importlib.invalidate_caches()
    dist = metadata.distribution(contest.owner)
    checked, damaged = damaged_files(dist, contest.package)
    if checked == 0 or damaged:
        print(f"{label}: still damaged after reinstalling: {', '.join(damaged[:5])}")
        return False
    print(
        f"{label}: repaired - reinstalled {contest.owner} {dist.version} from poetry.lock; "
        f"all {checked} files match its RECORD"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify (and repair) the owners of top-level packages that two "
        "locked distributions both install. See the module docstring for why."
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
        help="poetry.lock to take the owner's pin and hashes from (default: %(default)s)",
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
