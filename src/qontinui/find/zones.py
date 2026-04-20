"""PolygonZone — region-of-interest membership testing for detections.

Defines arbitrary polygon zones on screen and tests which detections
fall inside.  Inspired by roboflow/supervision's PolygonZone, adapted
for GUI automation with screen-coordinate semantics.

Example::

    import numpy as np
    from qontinui.find.zones import PolygonZone, Position
    from qontinui.find.detections import Detections

    # Define a rectangular zone in the top-left quadrant
    zone = PolygonZone(
        polygon=np.array([[0, 0], [960, 0], [960, 540], [0, 540]]),
        triggering_position=Position.CENTER,
    )

    inside_mask = zone.trigger(detections)
    print(f"{zone.current_count} detections in zone")
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .detections import Detections


class Position(Enum):
    """Anchor point on a bounding box used for zone membership testing."""

    CENTER = "center"
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


def _get_anchors(detections: Detections, position: Position) -> NDArray[np.floating]:
    """Extract anchor points from detections based on position.

    Args:
        detections: Detections container with xyxy boxes.
        position: Which anchor point to use.

    Returns:
        (n, 2) array of [x, y] anchor coordinates.
    """
    xyxy = detections.xyxy.astype(np.float64)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]

    if position == Position.CENTER:
        return np.column_stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    elif position == Position.BOTTOM_CENTER:
        return np.column_stack([(x1 + x2) / 2.0, y2])
    elif position == Position.TOP_CENTER:
        return np.column_stack([(x1 + x2) / 2.0, y1])
    elif position == Position.TOP_LEFT:
        return np.column_stack([x1, y1])
    elif position == Position.TOP_RIGHT:
        return np.column_stack([x2, y1])
    elif position == Position.BOTTOM_LEFT:
        return np.column_stack([x1, y2])
    elif position == Position.BOTTOM_RIGHT:
        return np.column_stack([x2, y2])
    else:
        return np.column_stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


class PolygonZone:
    """Arbitrary polygon zone for detection membership testing.

    Tests whether detection anchor points fall inside a polygon defined
    by screen-coordinate vertices.  After each ``trigger()`` call the
    ``current_count`` is updated.

    Args:
        polygon: Vertices as ``(N, 2)`` array of ``[x, y]`` integer coordinates.
            The polygon is automatically closed (last→first edge).
        triggering_position: Which bounding-box anchor to test for membership.
    """

    def __init__(
        self,
        polygon: NDArray[np.int_],
        triggering_position: Position = Position.BOTTOM_CENTER,
    ) -> None:
        polygon = np.asarray(polygon, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"polygon must have shape (N, 2), got {polygon.shape}")
        if len(polygon) < 3:
            raise ValueError(f"polygon needs at least 3 vertices, got {len(polygon)}")
        self.polygon = polygon
        self.triggering_position = triggering_position
        self._current_count: int = 0
        # Pre-compute contour for cv2 operations (immutable after init)
        self._contour = polygon.reshape((-1, 1, 2)).astype(np.float32)

    @property
    def current_count(self) -> int:
        """Number of detections inside the zone after the last ``trigger()``."""
        return self._current_count

    def trigger(self, detections: Detections) -> NDArray[np.bool_]:
        """Test which detections are inside this zone.

        Updates ``current_count`` as a side-effect.

        Args:
            detections: Detections container.

        Returns:
            Boolean mask of shape ``(n,)`` — ``True`` for detections inside.
        """
        n = len(detections)
        if n == 0:
            self._current_count = 0
            return np.empty(0, dtype=np.bool_)

        anchors = _get_anchors(detections, self.triggering_position)
        contour = self._contour

        inside = np.empty(n, dtype=np.bool_)
        for i in range(n):
            pt = (float(anchors[i, 0]), float(anchors[i, 1]))
            # pointPolygonTest returns: +1 inside, 0 on edge, -1 outside
            dist = cv2.pointPolygonTest(contour, pt, False)
            inside[i] = dist >= 0  # inside or on edge

        self._current_count = int(inside.sum())
        return inside

    def contains_point(self, x: float, y: float) -> bool:
        """Test if a single point is inside this zone.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if the point is inside or on the edge.
        """
        contour = self._contour
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    @property
    def area(self) -> float:
        """Area of the polygon in pixels² (via cv2.contourArea)."""
        contour = self._contour
        return float(cv2.contourArea(contour))

    @property
    def bounding_rect(self) -> tuple[int, int, int, int]:
        """Axis-aligned bounding rectangle ``(x, y, width, height)``."""
        contour = self.polygon.reshape((-1, 1, 2))
        return cast(tuple[int, int, int, int], cv2.boundingRect(contour))

    def __repr__(self) -> str:
        n_verts = len(self.polygon)
        return (
            f"PolygonZone(vertices={n_verts}, "
            f"position={self.triggering_position.value}, "
            f"current_count={self._current_count})"
        )
