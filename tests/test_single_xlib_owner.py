"""Guard: ``poetry.lock`` locks at most one distribution that installs ``Xlib/``.

``python-xlib`` (required by pynput) and the abandoned ``python3-xlib`` fork
(required by pyautogui / mouseinfo) both install a top-level ``Xlib/``
package. Poetry installs them concurrently, which tears or mixes the
package and breaks ``import pynput`` nondeterministically. pyautogui was
removed from qontinui so that only ``python-xlib`` is locked; this test keeps
it that way.

The lock path defaults to the repo's ``poetry.lock`` and can be pointed at
another lock with ``QONTINUI_XLIB_GUARD_LOCK`` (used to show the guard fails
on the pre-fix lock).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

#: Distributions known to install a top-level ``Xlib/`` package.
XLIB_DISTRIBUTIONS = frozenset({"python-xlib", "python3-xlib", "xlib"})

#: The one allowed owner: pynput's dependency.
ALLOWED_XLIB_OWNERS = frozenset({"python-xlib"})

#: Distributions whose return would bring the ``python3-xlib`` fork back.
FORBIDDEN_DISTRIBUTIONS = frozenset({"pyautogui", "mouseinfo"})

_DEFAULT_LOCK = Path(__file__).resolve().parent.parent / "poetry.lock"


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


@pytest.fixture(scope="module")
def locked_names() -> set[str]:
    lock_path = Path(os.environ.get("QONTINUI_XLIB_GUARD_LOCK", _DEFAULT_LOCK))
    with lock_path.open("rb") as f:
        lock = tomllib.load(f)
    names = {_normalize(pkg["name"]) for pkg in lock.get("package", [])}
    assert names, f"{lock_path} locks no packages; refusing to pass vacuously"
    return names


def test_only_python_xlib_owns_xlib(locked_names: set[str]) -> None:
    owners = locked_names & XLIB_DISTRIBUTIONS
    assert owners <= ALLOWED_XLIB_OWNERS, (
        f"poetry.lock locks {sorted(owners)} as Xlib/ installers; only "
        f"{sorted(ALLOWED_XLIB_OWNERS)} may be locked "
        f"(offender: {sorted(owners - ALLOWED_XLIB_OWNERS)})."
    )


def test_pyautogui_closure_is_not_locked(locked_names: set[str]) -> None:
    offenders = sorted(locked_names & FORBIDDEN_DISTRIBUTIONS)
    assert not offenders, (
        f"poetry.lock locks {offenders}, which require the python3-xlib fork "
        "and reintroduce a second Xlib/ owner."
    )
