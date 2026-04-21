"""Cross-repo parity test for VGA canonical JSON.

The canonical JSON produced by :meth:`VgaStateMachine.to_canonical_json`
is the input to ``content_hash`` (SHA-256) used for content-addressable
storage and round-trip equality checks through ``POST /api/vga/state/import``.

A TypeScript twin (``qontinui-web/frontend/src/lib/vga/canonical.ts``) emits
the *same byte string* for the same semantic input. If either side adds,
renames, or re-orders a field without the other, the hash diverges
silently — imported state machines look "modified" when they are not.

This test freezes:

1. An input fixture at
   ``test-fixtures/vga/canonical-state-machine.json``.
2. The byte-exact canonical serialization at
   ``test-fixtures/vga/canonical-state-machine.canonical.json``.
3. The SHA-256 digest of that serialization
   (see :data:`FROZEN_SHA256` below).

The sibling TS test ``canonical.parity.test.ts`` asserts the same byte
string from the TS encoder. Intentional schema changes must update the
fixture *and* both encoders in the same commit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from qontinui.vga.state_machine import VgaStateMachine

# Repo layout: <qontinui>/tests/test_vga_canonical_parity.py
#   qontinui repo root = parents[1]. Fixtures live inside the qontinui
#   repo (and are duplicated under qontinui-web/frontend/test-fixtures/)
#   so each repo can run its own parity test without external deps.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = _REPO_ROOT / "test-fixtures" / "vga"
INPUT_FIXTURE = _FIXTURE_DIR / "canonical-state-machine.json"
EXPECTED_FIXTURE = _FIXTURE_DIR / "canonical-state-machine.canonical.json"

# SHA-256 of the frozen canonical output. If this digest changes, the
# content_hash of every state machine previously persisted also changes
# — treat it as a breaking schema event and coordinate with the web /
# TS side in the same commit.
FROZEN_SHA256 = "1a43fefeca0fe4ec3398bfee78fff41ab37fa05e673db83856e6db21bb070cb4"


def _read_expected() -> str:
    """Read the expected canonical output verbatim.

    The file is stored without a trailing newline; we do NOT strip —
    the test asserts byte-exact equality.
    """
    return EXPECTED_FIXTURE.read_text(encoding="utf-8")


def test_fixtures_exist() -> None:
    assert INPUT_FIXTURE.is_file(), f"missing input fixture: {INPUT_FIXTURE}"
    assert EXPECTED_FIXTURE.is_file(), (
        f"missing expected canonical fixture: {EXPECTED_FIXTURE}"
    )


def test_expected_has_no_trailing_newline() -> None:
    """Byte-exactness sentinel: a stray newline silently breaks parity."""
    raw = EXPECTED_FIXTURE.read_bytes()
    assert not raw.endswith(b"\n"), (
        "expected canonical fixture must not end with a newline — "
        "json.dumps output has no trailing newline and the TS side "
        "emits none either"
    )


def test_python_encoder_matches_expected_bytes() -> None:
    """Python's canonical encoder reproduces the frozen byte string."""
    sm = VgaStateMachine.from_canonical_json(
        INPUT_FIXTURE.read_text(encoding="utf-8")
    )
    actual = sm.to_canonical_json()
    expected = _read_expected()
    assert actual == expected, (
        "Python canonical output drifted from the frozen fixture. "
        "If this is intentional, regenerate "
        f"{EXPECTED_FIXTURE} AND update the TS encoder AND refresh "
        "FROZEN_SHA256 in this file AND coordinate the same fixture "
        "update with canonical.parity.test.ts."
    )


def test_frozen_sha256_stable() -> None:
    """Freeze the SHA-256 digest so ``content_hash`` can't drift unnoticed."""
    sm = VgaStateMachine.from_canonical_json(
        INPUT_FIXTURE.read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(sm.to_canonical_json().encode("utf-8")).hexdigest()
    assert digest == FROZEN_SHA256, (
        f"content_hash digest changed: {digest} != {FROZEN_SHA256}. "
        "Any change here breaks content-addressable storage; update "
        "FROZEN_SHA256 and the expected fixture together, and confirm "
        "the TS side emits the same bytes."
    )


def test_sm_sha256_method_matches_frozen() -> None:
    """The public :meth:`VgaStateMachine.sha256` helper agrees with the digest."""
    sm = VgaStateMachine.from_canonical_json(
        INPUT_FIXTURE.read_text(encoding="utf-8")
    )
    assert sm.sha256() == FROZEN_SHA256
