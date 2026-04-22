"""VGA state machine Pydantic models.

This module defines the **DTO** shape of a VGA state machine: what gets
persisted to PG's ``runner.vga_state_machines.state_graph`` JSONB column,
what the web's ``GET /api/vga/state/{id}.json`` endpoint returns, and what
the Python worker loads at runtime.

Design decision (plan §13): ``VgaState`` is *not* the same class as
:class:`qontinui.model.state.state.State`. Instead, it carries the same
``blocking`` flag, and the RUNTIME (see :mod:`qontinui.vga.runtime`) uses
:class:`~qontinui.state_management.enhanced_active_state_set.EnhancedActiveStateSet`
— the existing abstraction — to enforce blocking-state policy. No new
overlay / modal abstraction is introduced.

All models are Pydantic v2 :class:`BaseModel`. Canonical JSON export is
deterministic (sorted keys, no timestamps) so hashes are stable across
machines. Round-trip is explicit via :meth:`VgaStateMachine.to_canonical_json`
+ :meth:`VgaStateMachine.from_canonical_json`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates.

    Matches the shape persisted by the builder UI and used by the Rust
    handler's event stream.
    """

    model_config = ConfigDict(frozen=False)

    x: int = Field(..., description="Top-left x in pixels.")
    y: int = Field(..., description="Top-left y in pixels.")
    w: int = Field(..., ge=0, description="Width in pixels.")
    h: int = Field(..., ge=0, description="Height in pixels.")

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def as_xyxy(self) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` — useful for IoU math."""
        return self.x, self.y, self.x + self.w, self.y + self.h


class VgaElement(BaseModel):
    """A persisted element inside a :class:`VgaState`.

    Attributes:
        id: Stable UUID. Used by :class:`~qontinui.vga.runtime.VgaAction`
            to reference targets.
        label: Human-readable label (e.g. ``"Save button"``).
        prompt: Natural-language description passed to v5 at runtime. The
            builder UI exposes this for user refinement.
        bbox: Last-confirmed bounding box.
        last_confirmed_at: When the user last confirmed (or corrected)
            the bbox.
        correction_count: How many times the user has corrected this
            element. Used to surface drift hotspots.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    label: str
    prompt: str
    bbox: BBox
    last_confirmed_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    correction_count: int = 0


class VgaState(BaseModel):
    """A persisted state (node) in the VGA state graph.

    The ``blocking`` field mirrors
    :attr:`qontinui.model.state.state.State.blocking` semantics. At runtime
    the VGA runtime pushes blocking states into the existing
    :class:`EnhancedActiveStateSet` so pathfinding / enforcement reuses
    the canonical code path — see plan §13 "modals reuse existing
    `blocking` states".
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str
    elements: list[VgaElement] = Field(default_factory=list)
    blocking: bool = False


class VgaTransition(BaseModel):
    """Directed edge between two :class:`VgaState` objects.

    Attributes:
        trigger_element_id: Element whose click (or activation) in
            ``from_state_id`` causes the move to ``to_state_id``.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    from_state_id: UUID
    to_state_id: UUID
    trigger_element_id: UUID


class VgaStateMachine(BaseModel):
    """Top-level VGA state-machine DTO.

    Attributes:
        target_process: OS-level process name used by HAL for window
            focusing (``"notepad++.exe"``, ``"obs64.exe"``, ...).
        target_os: ``"windows"`` | ``"macos"`` | ``"linux"``.
        grounding_model: Model name pinned to this SM. Defaults to
            ``qontinui-grounding-v5``. Rolling forward to v6 is an
            explicit migration, not a side effect of swapping llama-swap
            default (plan §13 recommendation A).
        private: When True, corrections logged against this SM may not
            be exported off-machine. Default True (plan §13 rec. D).
        created_at / updated_at: Populated by the persistence layer.
            Excluded from :meth:`to_canonical_json` output so hashes are
            content-addressable.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str
    target_process: str
    target_os: str
    grounding_model: str = "qontinui-grounding-v5"
    private: bool = True
    states: list[VgaState] = Field(default_factory=list)
    transitions: list[VgaTransition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # ------------------------------------------------------------------
    # Canonical JSON for content-addressable export / hashing
    # ------------------------------------------------------------------

    _CANONICAL_EXCLUDE: tuple[str, ...] = ("created_at", "updated_at")
    """Fields stripped from canonical output — their values are
    environment-dependent and would make the hash non-reproducible."""

    def to_canonical_json(self) -> str:
        """Return stable-key-order JSON with no timestamps.

        Used by the web export route ``GET /state/[id].json`` and by
        :meth:`sha256`.

        This encoder has a parity twin at
        ``qontinui-web/frontend/src/lib/vga/canonical.ts``. Shared
        fixtures live at ``test-fixtures/vga/canonical-state-machine{,
        .canonical}.json``. Tests must update both sides OR canonical
        JSON (and thus ``content_hash``) will diverge silently; CI will
        fail (see ``tests/test_vga_canonical_parity.py``).
        """
        data = self.model_dump(mode="json", exclude_none=False)
        for key in self._CANONICAL_EXCLUDE:
            data.pop(key, None)

        # Strip timestamps from nested elements so the hash reflects only
        # structural + semantic content.
        for state in data.get("states", []):
            for element in state.get("elements", []):
                element.pop("last_confirmed_at", None)
                element.pop("correction_count", None)

        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, data: str | dict[str, Any]) -> VgaStateMachine:
        """Round-trip a canonical JSON payload back into a model.

        Missing timestamps are re-synthesized at load time — they don't
        participate in the hash anyway.
        """
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = dict(data)

        now = datetime.now(tz=UTC).isoformat()
        parsed.setdefault("created_at", now)
        parsed.setdefault("updated_at", now)

        # Element timestamps are similarly optional in canonical form.
        for state in parsed.get("states", []):
            for element in state.get("elements", []):
                element.setdefault("last_confirmed_at", now)
                element.setdefault("correction_count", 0)

        return cls.model_validate(parsed)

    def sha256(self) -> str:
        """SHA-256 hex digest of the canonical JSON.

        Deterministic across machines — two state machines with identical
        structure + labels + bboxes hash to the same value.
        """
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Lookup helpers used by the runtime
    # ------------------------------------------------------------------

    def find_element(self, element_id: UUID) -> tuple[VgaState, VgaElement] | None:
        """Locate an element by ID plus the state that owns it."""
        for state in self.states:
            for element in state.elements:
                if element.id == element_id:
                    return state, element
        return None

    def blocking_states(self) -> list[VgaState]:
        """All states flagged as blocking. Mirrors
        :meth:`EnhancedActiveStateSet.get_blocking_states`."""
        return [s for s in self.states if s.blocking]
