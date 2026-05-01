"""LineZone — count detections crossing a line segment.

Tracks detections crossing a directed line segment, counting entries
in the "in" direction (left-to-right relative to the line vector) and
"out" direction (right-to-left).  Requires ``tracker_id`` on the
``Detections`` container to maintain per-object crossing state.

Example::

    from qontinui.find.line_zone import LineZone, Point
    from qontinui.find.detections import Detections

    line = LineZone(start=Point(100, 300), end=Point(500, 300))

    # Per-frame update (detections must have tracker_id)
    in_mask, out_mask = line.trigger(detections)
    print(f"in={line.in_count}, out={line.out_count}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .zones import Position, _get_anchors

if TYPE_CHECKING:
    from .detections import Detections


@dataclass(frozen=True)
class Point:
    """2D point."""

    x: float
    y: float


class LineZone:
    """Directed line segment for crossing-based counting.

    Detections are tracked across frames via ``tracker_id``.  When an
    anchor point crosses from one side of the line to the other between
    two consecutive ``trigger()`` calls, it is counted as either *in*
    (positive side → negative side relative to line normal) or *out*.

    Args:
        start: Start point of the line segment.
        end: End point of the line segment.
        triggering_position: Which bounding-box anchor to track.
    """

    def __init__(
        self,
        start: Point,
        end: Point,
        triggering_position: Position = Position.BOTTOM_CENTER,
    ) -> None:
        self.start = start
        self.end = end
        self.triggering_position = triggering_position
        self._in_count: int = 0
        self._out_count: int = 0
        # Maps tracker_id → signed distance to line on previous frame
        self._prev_signs: dict[int, float] = {}

    @property
    def in_count(self) -> int:
        """Cumulative count of detections crossing *into* the zone."""
        return self._in_count

    @property
    def out_count(self) -> int:
        """Cumulative count of detections crossing *out of* the zone."""
        return self._out_count

    def reset(self) -> None:
        """Reset all counters and tracking state."""
        self._in_count = 0
        self._out_count = 0
        self._prev_signs.clear()

    def trigger(
        self, detections: Detections
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Detect line crossings for the current frame.

        Args:
            detections: Detections container.  Must have ``tracker_id`` set
                (non-None) for crossing tracking to work.

        Returns:
            ``(in_mask, out_mask)`` — boolean arrays of shape ``(n,)``
            indicating which detections crossed *in* or *out* on this frame.
        """
        n = len(detections)
        in_mask = np.zeros(n, dtype=np.bool_)
        out_mask = np.zeros(n, dtype=np.bool_)

        if n == 0:
            return in_mask, out_mask

        # Require explicit tracker_id in data dict — class_id is not a
        # valid substitute since it identifies object *type*, not instance.
        if "tracker_id" not in detections.data:
            return in_mask, out_mask

        tracker_ids = np.asarray(detections.data["tracker_id"])

        anchors = _get_anchors(detections, self.triggering_position)
        signs = self._signed_distances(anchors)

        new_prev: dict[int, float] = {}
        for i in range(n):
            tid = int(tracker_ids[i])
            current_sign = signs[i]
            new_prev[tid] = current_sign

            if tid in self._prev_signs:
                prev_sign = self._prev_signs[tid]
                # Crossing detected: sign change (excluding zero).
                # "In" = crossing from negative to positive side.
                # "Out" = crossing from positive to negative side.
                if prev_sign < 0 and current_sign >= 0:
                    in_mask[i] = True
                    self._in_count += 1
                elif prev_sign > 0 and current_sign <= 0:
                    out_mask[i] = True
                    self._out_count += 1

        self._prev_signs = new_prev
        return in_mask, out_mask

    def _signed_distances(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute signed distance of each point from the line.

        Positive = left of the line direction (start→end),
        Negative = right of the line direction.

        Args:
            points: (n, 2) array of [x, y].

        Returns:
            (n,) array of signed distances.
        """
        # Line vector
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y

        # Cross product gives signed distance (unnormalised)
        px = points[:, 0] - self.start.x
        py = points[:, 1] - self.start.y
        return dx * py - dy * px

    def __repr__(self) -> str:
        return (
            f"LineZone(start=({self.start.x}, {self.start.y}), "
            f"end=({self.end.x}, {self.end.y}), "
            f"in={self._in_count}, out={self._out_count})"
        )
