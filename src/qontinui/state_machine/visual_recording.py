"""Visual GUI automation recording via screenshots + accessibility tree.

Records non-SDK desktop apps by periodically capturing screenshots and the
OS accessibility tree, then produces a CooccurrenceExport-compatible dict
for consumption by the RecordingPipeline — the same format the TypeScript
SDK recording produces.

Uses PairedCaptureService for atomic screenshot + a11y capture and FNV-1a
hashing for element fingerprints (matching the TypeScript implementation).
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qontinui.hal.services.paired_capture import PairedCaptureService

logger = logging.getLogger(__name__)


# =============================================================================
# FNV-1a hash (matches TypeScript computeHashSync)
# =============================================================================


def _fnv1a_fingerprint_hash(
    structural_path: str,
    position_zone: str,
    role: str,
    accessible_name: str | None,
    size_category: str,
) -> str:
    """FNV-1a hash matching the TypeScript computeHashSync().

    Input format: "structuralPath|positionZone|role|accessibleName|sizeCategory"
    Output: 16-char hex string (two 32-bit FNV-1a hashes concatenated).
    """
    input_str = f"{structural_path}|{position_zone}|{role}|{accessible_name or ''}|{size_category}"

    h1 = 0x811C9DC5
    h2 = 0x811C9DC5

    for char in input_str:
        c = ord(char)
        h1 ^= c
        h1 = ctypes.c_int32(h1 * 0x01000193).value
        h2 ^= ctypes.c_int32(c * 31).value
        h2 = ctypes.c_int32(h2 * 0x01000193).value

    hex1 = format(ctypes.c_uint32(h1).value, "08x")
    hex2 = format(ctypes.c_uint32(h2).value, "08x")
    return hex1 + hex2


# =============================================================================
# Position zone and size category classifiers
# =============================================================================


def _classify_position_zone(
    x: int,
    y: int,
    w: int,
    h: int,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> str:
    """Classify an element's position zone from its screen bounds."""
    cx = x + w // 2
    cy = y + h // 2

    # Vertical band
    if cy < screen_h * 0.1:
        v = "header"
    elif cy > screen_h * 0.9:
        v = "footer"
    else:
        v = "main"

    # Horizontal band
    if cx < screen_w * 0.2:
        h_zone = "sidebar-left"
    elif cx > screen_w * 0.8:
        h_zone = "sidebar-right"
    else:
        h_zone = ""

    if h_zone:
        return h_zone if v == "main" else f"{v}"
    return v


def _classify_size_category(w: int, h: int) -> str:
    """Classify an element's size from its bounds."""
    area = w * h
    if area < 400:
        return "icon"
    if area < 2500:
        return "button"
    if area < 10000:
        return "small"
    if area < 50000:
        return "medium"
    if area < 200000:
        return "large"
    return "panel"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class VisualRecordingConfig:
    """Configuration for visual recording sessions."""

    capture_interval_ms: int = 2000
    """Interval between periodic captures in milliseconds."""

    screen_width: int = 1920
    """Monitor width for position zone classification."""

    screen_height: int = 1080
    """Monitor height for position zone classification."""

    monitor: int | None = None
    """Monitor index for capture (None = primary)."""


@dataclass
class _CaptureSnapshot:
    """Internal snapshot from a single capture."""

    id: str
    timestamp: float
    content_hash: str
    fingerprint_hashes: list[str]
    app_name: str | None = None
    window_title: str | None = None
    url: str | None = None


@dataclass
class _ElementFingerprint:
    """Fingerprint data for a single accessibility element."""

    hash: str
    structural_path: str
    position_zone: str
    role: str
    accessible_name: str | None
    size_category: str


@dataclass
class _RecordedInteraction:
    """A user interaction recorded at screen coordinates."""

    id: str
    timestamp: float
    action_type: str
    x: int
    y: int
    before_capture_id: str
    after_capture_id: str | None = None
    target_fingerprint: str | None = None


# =============================================================================
# Visual Recording Session
# =============================================================================


class VisualRecordingSession:
    """Recording session for non-SDK desktop apps using screenshots + a11y.

    Periodically captures the screen and accessibility tree via PairedCaptureService,
    tracks state changes via content hash deduplication, and builds element
    fingerprints from accessibility tree nodes. Produces the same CooccurrenceExport
    format as the SDK recording for compatibility with RecordingPipeline.
    """

    def __init__(
        self,
        capture_service: PairedCaptureService,
        config: VisualRecordingConfig | None = None,
    ) -> None:
        self._capture = capture_service
        self._config = config or VisualRecordingConfig()

        # Session state
        self._session_id: str = ""
        self._start_time: float = 0.0
        self._active: bool = False
        self._capture_task: asyncio.Task[None] | None = None

        # Collected data
        self._snapshots: list[_CaptureSnapshot] = []
        self._all_fingerprints: dict[str, _ElementFingerprint] = {}
        self._interactions: list[_RecordedInteraction] = []
        self._last_content_hash: str | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start periodic capture loop."""
        if self._active:
            return

        self._session_id = f"visual-{uuid.uuid4().hex[:12]}"
        self._start_time = time.time()
        self._active = True
        self._snapshots = []
        self._all_fingerprints = {}
        self._interactions = []
        self._last_content_hash = None

        # Take initial capture
        await self._take_capture()

        # Start periodic capture loop
        self._capture_task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> dict[str, Any]:
        """Stop and return CooccurrenceExport-compatible dict."""
        if not self._active:
            raise RuntimeError("No active visual recording session")

        self._active = False

        # Cancel the capture loop
        if self._capture_task is not None:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        # Take final capture
        await self._take_capture()

        return self._build_export()

    async def record_interaction(self, action_type: str, x: int, y: int) -> None:
        """Record a user interaction at screen coordinates.

        Takes a before-capture (or uses the latest if recent), records the
        interaction, then takes an after-capture to detect state changes.
        """
        if not self._active:
            return

        # Use the most recent capture as "before" if it's less than 500ms old
        before_id: str
        if self._snapshots and (time.time() * 1000 - self._snapshots[-1].timestamp) < 500:
            before_id = self._snapshots[-1].id
        else:
            before_id = await self._take_capture()

        interaction = _RecordedInteraction(
            id=f"action-{uuid.uuid4().hex[:8]}",
            timestamp=int(time.time() * 1000),
            action_type=action_type,
            x=x,
            y=y,
            before_capture_id=before_id,
        )

        # Find nearest fingerprint to the interaction coordinates
        interaction.target_fingerprint = self._find_nearest_fingerprint(x, y)

        # Wait briefly for the UI to settle, then take "after" capture
        await asyncio.sleep(0.3)
        after_id = await self._take_capture()
        interaction.after_capture_id = after_id

        self._interactions.append(interaction)

    # =========================================================================
    # Capture Loop
    # =========================================================================

    async def _capture_loop(self) -> None:
        """Periodic capture loop running in background."""
        interval = self._config.capture_interval_ms / 1000.0
        while self._active:
            await asyncio.sleep(interval)
            if not self._active:
                break
            try:
                await self._take_capture()
            except Exception:
                logger.exception("Visual recording capture failed")

    async def _take_capture(self) -> str:
        """Take a single capture, returning the capture ID.

        Uses content hash deduplication: if the content hash matches the
        previous capture, we still record it but note the dedup.
        """
        result = await self._capture.capture(monitor=self._config.monitor)

        # Content hash deduplication — skip if identical to last capture
        if self._last_content_hash and result.content_hash == self._last_content_hash:
            # Return the previous capture ID since nothing changed
            if self._snapshots:
                return self._snapshots[-1].id
            # Fallback: still record it
        self._last_content_hash = result.content_hash

        capture_id = f"cap-{uuid.uuid4().hex[:8]}"

        # Build fingerprints from accessibility tree if available
        fingerprint_hashes: list[str] = []
        if result.source_type == "accessibility" and result.metadata.get("total_nodes", 0) > 0:
            fingerprint_hashes = await self._build_fingerprints_from_a11y()

        snapshot = _CaptureSnapshot(
            id=capture_id,
            timestamp=int(time.time() * 1000),
            content_hash=result.content_hash,
            fingerprint_hashes=fingerprint_hashes,
            app_name=result.app_name,
            window_title=result.window_title,
            url=result.url,
        )
        self._snapshots.append(snapshot)
        return capture_id

    async def _build_fingerprints_from_a11y(self) -> list[str]:
        """Build element fingerprints from the current accessibility tree."""
        hashes: list[str] = []

        try:
            # Access the a11y backend through the capture service
            a11y = self._capture._a11y
            if not a11y or not a11y.is_connected():
                return hashes

            snapshot = await a11y.capture_tree()
            self._walk_a11y_node(snapshot.root, [], hashes)
        except Exception:
            logger.debug("Failed to build a11y fingerprints", exc_info=True)

        return hashes

    def _walk_a11y_node(
        self,
        node: Any,
        path_parts: list[str],
        out_hashes: list[str],
        depth: int = 0,
    ) -> None:
        """Recursively walk a11y tree and build fingerprints for each node."""
        if depth > 100:
            return

        role = getattr(node, "role", "") or ""
        name = getattr(node, "name", None)
        bounds = getattr(node, "bounds", None)

        # Build structural path
        current_path = [*path_parts, role] if role else path_parts
        structural_path = " > ".join(current_path) if current_path else ""

        # Classify position and size from bounds
        if bounds and hasattr(bounds, "x"):
            x = int(getattr(bounds, "x", 0))
            y = int(getattr(bounds, "y", 0))
            w = int(getattr(bounds, "width", 0))
            h = int(getattr(bounds, "height", 0))
            position_zone = _classify_position_zone(
                x,
                y,
                w,
                h,
                self._config.screen_width,
                self._config.screen_height,
            )
            size_category = _classify_size_category(w, h)
        else:
            position_zone = "main"
            size_category = "medium"

        # Only fingerprint nodes with a role (skip generic containers)
        if role:
            fp_hash = _fnv1a_fingerprint_hash(
                structural_path,
                position_zone,
                role,
                name,
                size_category,
            )
            if fp_hash not in self._all_fingerprints:
                self._all_fingerprints[fp_hash] = _ElementFingerprint(
                    hash=fp_hash,
                    structural_path=structural_path,
                    position_zone=position_zone,
                    role=role,
                    accessible_name=name,
                    size_category=size_category,
                )
            out_hashes.append(fp_hash)

        # Recurse children
        children = getattr(node, "children", []) or []
        for child in children:
            self._walk_a11y_node(child, current_path, out_hashes, depth + 1)

    def _find_nearest_fingerprint(self, x: int, y: int) -> str | None:
        """Find the fingerprint hash of the element nearest to (x, y)."""
        # Simple approach: check the most recent snapshot's fingerprints
        # and find the one whose position zone best matches the coordinates
        target_zone = _classify_position_zone(
            x,
            y,
            1,
            1,
            self._config.screen_width,
            self._config.screen_height,
        )

        # Find fingerprints in the same zone
        candidates = [
            fp for fp in self._all_fingerprints.values() if fp.position_zone == target_zone
        ]

        if not candidates:
            return None

        # Return the first matching candidate (heuristic — a more sophisticated
        # approach would use exact bounds overlap)
        return candidates[0].hash

    # =========================================================================
    # Export Building
    # =========================================================================

    def _build_export(self) -> dict[str, Any]:
        """Build a CooccurrenceExport-compatible dict from collected data."""
        all_fp_hashes = list(self._all_fingerprints.keys())

        # Fingerprint details
        fingerprint_details: dict[str, dict[str, Any]] = {}
        for fp_hash, fp in self._all_fingerprints.items():
            fingerprint_details[fp_hash] = {
                "structuralPath": fp.structural_path,
                "positionZone": fp.position_zone,
                "role": fp.role,
                "accessibleName": fp.accessible_name,
                "sizeCategory": fp.size_category,
                "hash": fp.hash,
            }

        # Presence matrix
        presence_matrix: list[dict[str, Any]] = []
        for snap in self._snapshots:
            presence_matrix.append(
                {
                    "captureId": snap.id,
                    "url": snap.url or snap.window_title or "",
                    "fingerprints": snap.fingerprint_hashes,
                }
            )

        # Co-occurrence counts
        cooccurrence_counts: dict[str, dict[str, int]] = {}
        for snap in self._snapshots:
            hashes = snap.fingerprint_hashes
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    a, b = hashes[i], hashes[j]
                    cooccurrence_counts.setdefault(a, {})
                    cooccurrence_counts.setdefault(b, {})
                    cooccurrence_counts[a][b] = cooccurrence_counts[a].get(b, 0) + 1
                    cooccurrence_counts[b][a] = cooccurrence_counts[b].get(a, 0) + 1

        # Build inverted index: hash -> list of (capture_id, timestamp)
        hash_to_captures: dict[str, list[tuple[str, float]]] = {}
        for snap in self._snapshots:
            for h in set(snap.fingerprint_hashes):
                hash_to_captures.setdefault(h, []).append((snap.id, snap.timestamp))

        # Fingerprint stats
        fingerprint_stats: dict[str, dict[str, Any]] = {}
        for fp_hash in all_fp_hashes:
            entries = hash_to_captures.get(fp_hash, [])
            capture_ids = [e[0] for e in entries]
            timestamps = [e[1] for e in entries]
            fingerprint_stats[fp_hash] = {
                "totalAppearances": len(capture_ids),
                "captureIds": capture_ids,
                "firstSeen": min(timestamps) if timestamps else 0,
                "lastSeen": max(timestamps) if timestamps else 0,
            }

        # Transitions from recorded interactions
        transitions: list[dict[str, Any]] = []
        for interaction in self._interactions:
            before_snap = next(
                (s for s in self._snapshots if s.id == interaction.before_capture_id), None
            )
            after_snap = (
                next((s for s in self._snapshots if s.id == interaction.after_capture_id), None)
                if interaction.after_capture_id
                else None
            )

            before_set = set(before_snap.fingerprint_hashes) if before_snap else set()
            after_set = set(after_snap.fingerprint_hashes) if after_snap else set()

            transitions.append(
                {
                    "actionId": interaction.id,
                    "actionType": interaction.action_type,
                    "targetFingerprint": interaction.target_fingerprint,
                    "beforeCaptureId": interaction.before_capture_id,
                    "afterCaptureId": interaction.after_capture_id or "",
                    "appearedFingerprints": list(after_set - before_set),
                    "disappearedFingerprints": list(before_set - after_set),
                    "timestamp": interaction.timestamp,
                }
            )

        # State candidates: group fingerprints by identical presence signature
        sig_groups: dict[str, list[str]] = {}
        for fp_hash in all_fp_hashes:
            entries = hash_to_captures.get(fp_hash, [])
            sig = ",".join(sorted(e[0] for e in entries))
            sig_groups.setdefault(sig, []).append(fp_hash)

        state_candidates: list[dict[str, Any]] = []
        for group in sig_groups.values():
            if len(group) < 2:
                continue
            # Determine dominant position zone
            zones = [
                self._all_fingerprints[h].position_zone
                for h in group
                if h in self._all_fingerprints
            ]
            dominant_zone = max(set(zones), key=zones.count) if zones else None

            state_candidates.append(
                {
                    "fingerprints": sorted(group),
                    "cooccurrenceRate": 1.0,
                    "positionZone": dominant_zone,
                }
            )

        return {
            "sessionId": self._session_id,
            "exportedAt": int(time.time() * 1000),
            "allFingerprints": all_fp_hashes,
            "fingerprintDetails": fingerprint_details,
            "presenceMatrix": presence_matrix,
            "cooccurrenceCounts": cooccurrence_counts,
            "fingerprintStats": fingerprint_stats,
            "transitions": transitions,
            "stateCandidates": state_candidates,
        }
