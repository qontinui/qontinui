"""Detections — universal container for detection results.

Inspired by roboflow/supervision's sv.Detections, adapted for GUI automation.
Provides numpy-backed batch operations, filtering, merging, and NMS on
detection results from any cascade backend.

Example::

    from qontinui.find.detections import Detections

    # From cascade backend results
    dets = Detections.from_detection_results(results)

    # Filter by confidence
    high_conf = dets[dets.confidence > 0.9]

    # Merge results from multiple backends
    merged = Detections.merge([dets1, dets2])

    # Apply non-maximum suppression
    deduped = merged.with_nms(iou_threshold=0.5)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class Detections:
    """Universal container for detection results from any backend.

    All fields are numpy arrays of length ``n`` (the number of detections).
    Optional fields are ``None`` when not available.

    Attributes:
        xyxy: Bounding boxes as ``(n, 4)`` array of ``[x1, y1, x2, y2]`` in pixels.
        confidence: Confidence scores as ``(n,)`` float array (0.0–1.0).
        backend_name: Backend names as ``(n,)`` object array of strings.
        label: Optional ``(n,)`` object array of string labels.
        class_id: Optional ``(n,)`` int array of class identifiers.
        normalized_xyxy: Optional ``(n, 4)`` array in 0.0–1.0 screen-relative coords.
        data: Extensible dict mapping string keys to ``(n,)`` arrays or lists.
    """

    __slots__ = (
        "xyxy",
        "confidence",
        "backend_name",
        "label",
        "class_id",
        "normalized_xyxy",
        "data",
    )

    def __init__(
        self,
        xyxy: NDArray[np.int_],
        confidence: NDArray[np.floating],
        backend_name: NDArray[np.object_],
        label: NDArray[np.object_] | None = None,
        class_id: NDArray[np.int_] | None = None,
        normalized_xyxy: NDArray[np.floating] | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        n = len(xyxy)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise ValueError(f"xyxy must have shape (n, 4), got {xyxy.shape}")
        if len(confidence) != n:
            raise ValueError(f"confidence length {len(confidence)} != xyxy length {n}")
        if len(backend_name) != n:
            raise ValueError(f"backend_name length {len(backend_name)} != xyxy length {n}")
        if label is not None and len(label) != n:
            raise ValueError(f"label length {len(label)} != xyxy length {n}")
        if class_id is not None and len(class_id) != n:
            raise ValueError(f"class_id length {len(class_id)} != xyxy length {n}")
        if normalized_xyxy is not None:
            if normalized_xyxy.shape != (n, 4):
                raise ValueError(f"normalized_xyxy shape {normalized_xyxy.shape} != ({n}, 4)")

        self.xyxy = xyxy
        self.confidence = confidence
        self.backend_name = backend_name
        self.label = label
        self.class_id = class_id
        self.normalized_xyxy = normalized_xyxy
        self.data = data if data is not None else {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> Detections:
        """Create an empty Detections container."""
        return cls(
            xyxy=np.empty((0, 4), dtype=np.int_),
            confidence=np.empty(0, dtype=np.float64),
            backend_name=np.empty(0, dtype=object),
        )

    @classmethod
    def from_detection_results(
        cls,
        results: list,
        *,
        screen_width: int = 0,
        screen_height: int = 0,
    ) -> Detections:
        """Convert a list of ``DetectionResult`` dataclasses into Detections.

        This is the primary adapter for integrating with the existing cascade
        detector output.

        Args:
            results: List of DetectionResult instances.
            screen_width: Screen width for normalization (0 to skip).
            screen_height: Screen height for normalization (0 to skip).

        Returns:
            A new Detections container.
        """
        if not results:
            return cls.empty()

        n = len(results)
        xyxy = np.empty((n, 4), dtype=np.int_)
        confidence = np.empty(n, dtype=np.float64)
        backend_name = np.empty(n, dtype=object)
        label = np.empty(n, dtype=object)
        has_label = False

        for i, r in enumerate(results):
            xyxy[i] = [r.x, r.y, r.x + r.width, r.y + r.height]
            confidence[i] = r.confidence
            backend_name[i] = r.backend_name
            if r.label is not None:
                has_label = True
            label[i] = r.label

        # Build normalized_xyxy if coordinates are available
        normalized_xyxy: NDArray[np.floating] | None = None
        first = results[0]
        if first.normalized_x is not None:
            normalized_xyxy = np.empty((n, 4), dtype=np.float64)
            for i, r in enumerate(results):
                nx = r.normalized_x or 0.0
                ny = r.normalized_y or 0.0
                nw = r.normalized_width or 0.0
                nh = r.normalized_height or 0.0
                normalized_xyxy[i] = [nx, ny, nx + nw, ny + nh]
        elif screen_width > 0 and screen_height > 0:
            normalized_xyxy = np.empty((n, 4), dtype=np.float64)
            normalized_xyxy[:, 0] = xyxy[:, 0] / screen_width
            normalized_xyxy[:, 1] = xyxy[:, 1] / screen_height
            normalized_xyxy[:, 2] = xyxy[:, 2] / screen_width
            normalized_xyxy[:, 3] = xyxy[:, 3] / screen_height

        # Collect metadata into data dict
        data: dict[str, Any] = {}
        metadata_list = [r.metadata for r in results]
        if any(metadata_list):
            data["metadata"] = metadata_list

        return cls(
            xyxy=xyxy,
            confidence=confidence,
            backend_name=backend_name,
            label=label if has_label else None,
            normalized_xyxy=normalized_xyxy,
            data=data,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.xyxy)

    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def area(self) -> NDArray[np.int_]:
        """Pixel area of each bounding box, shape ``(n,)``."""
        widths = self.xyxy[:, 2] - self.xyxy[:, 0]
        heights = self.xyxy[:, 3] - self.xyxy[:, 1]
        return cast(NDArray[np.int_], widths * heights)

    @property
    def center(self) -> NDArray[np.floating]:
        """Center point of each bounding box, shape ``(n, 2)``."""
        return np.column_stack(
            [
                (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0,
                (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0,
            ]
        )

    @property
    def width(self) -> NDArray[np.int_]:
        """Width of each bounding box, shape ``(n,)``."""
        return self.xyxy[:, 2] - self.xyxy[:, 0]

    @property
    def height(self) -> NDArray[np.int_]:
        """Height of each bounding box, shape ``(n,)``."""
        return self.xyxy[:, 3] - self.xyxy[:, 1]

    # ------------------------------------------------------------------
    # Indexing and filtering
    # ------------------------------------------------------------------

    def __getitem__(self, index: NDArray[np.bool_] | slice | int | list[int]) -> Detections:
        """Filter detections by boolean mask, slice, integer index, or index list.

        Examples::

            # Boolean mask (confidence filter)
            high = dets[dets.confidence > 0.9]

            # Slice
            top5 = dets[:5]

            # Single detection
            first = dets[0]

            # Index list
            selected = dets[[0, 2, 4]]
        """
        if isinstance(index, int | np.integer):
            index = [int(index)]

        # Resolve data dict indexing — lists need explicit index expansion
        new_data: dict[str, Any] = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                new_data[k] = v[index]
            elif isinstance(v, list):
                if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                    new_data[k] = [v[i] for i, keep in enumerate(index) if keep]
                elif isinstance(index, slice):
                    new_data[k] = v[index]
                else:
                    idx = index if isinstance(index, list) else list(index)
                    new_data[k] = [v[i] for i in idx]
            else:
                new_data[k] = v

        return Detections(
            xyxy=self.xyxy[index],
            confidence=self.confidence[index],
            backend_name=self.backend_name[index],
            label=self.label[index] if self.label is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            normalized_xyxy=(
                self.normalized_xyxy[index] if self.normalized_xyxy is not None else None
            ),
            data=new_data,
        )

    def sort_by_confidence(self, descending: bool = True) -> Detections:
        """Return a copy sorted by confidence score."""
        order = np.argsort(self.confidence)
        if descending:
            order = order[::-1]
        return self[order.tolist()]

    # ------------------------------------------------------------------
    # Merge and NMS
    # ------------------------------------------------------------------

    @classmethod
    def merge(cls, detections_list: list[Detections]) -> Detections:
        """Merge multiple Detections into one by concatenation.

        Args:
            detections_list: List of Detections to merge.

        Returns:
            A single Detections with all entries concatenated.
        """
        non_empty = [d for d in detections_list if not d.is_empty()]
        if not non_empty:
            return cls.empty()
        if len(non_empty) == 1:
            return non_empty[0]

        xyxy = np.concatenate([d.xyxy for d in non_empty])
        confidence = np.concatenate([d.confidence for d in non_empty])
        backend_name = np.concatenate([d.backend_name for d in non_empty])

        # Optional fields — only merge if all have them
        label: NDArray[np.object_] | None = None
        if all(d.label is not None for d in non_empty):
            label = np.concatenate([d.label for d in non_empty])

        class_id: NDArray[np.int_] | None = None
        if all(d.class_id is not None for d in non_empty):
            class_id = np.concatenate([d.class_id for d in non_empty])

        normalized_xyxy: NDArray[np.floating] | None = None
        if all(d.normalized_xyxy is not None for d in non_empty):
            normalized_xyxy = np.concatenate([d.normalized_xyxy for d in non_empty])

        # Merge data dicts — concatenate lists and arrays for shared keys
        all_keys: set[str] = set()
        for d in non_empty:
            all_keys.update(d.data.keys())

        data: dict[str, Any] = {}
        for key in all_keys:
            values = []
            for d in non_empty:
                if key in d.data:
                    v = d.data[key]
                    if isinstance(v, list):
                        values.extend(v)
                    elif isinstance(v, np.ndarray):
                        values.extend(v.tolist())
                    else:
                        values.extend([v] * len(d))
                else:
                    values.extend([None] * len(d))
            data[key] = values

        return cls(
            xyxy=xyxy,
            confidence=confidence,
            backend_name=backend_name,
            label=label,
            class_id=class_id,
            normalized_xyxy=normalized_xyxy,
            data=data,
        )

    def with_nms(self, iou_threshold: float = 0.5) -> Detections:
        """Apply non-maximum suppression and return filtered Detections.

        Keeps the highest-confidence detection when boxes overlap above
        the IoU threshold.

        Args:
            iou_threshold: IoU threshold for suppression (0.0–1.0).

        Returns:
            Filtered Detections with overlapping lower-confidence boxes removed.
        """
        if len(self) <= 1:
            return self

        keep = _nms(self.xyxy, self.confidence, iou_threshold)
        return self[keep]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, screen_width: int, screen_height: int) -> Detections:
        """Return a copy with normalized_xyxy computed from xyxy.

        Args:
            screen_width: Full screen width in pixels.
            screen_height: Full screen height in pixels.

        Returns:
            New Detections with normalized_xyxy populated.
        """
        if screen_width <= 0 or screen_height <= 0:
            return self
        norm = np.empty_like(self.xyxy, dtype=np.float64)
        norm[:, 0] = self.xyxy[:, 0] / screen_width
        norm[:, 1] = self.xyxy[:, 1] / screen_height
        norm[:, 2] = self.xyxy[:, 2] / screen_width
        norm[:, 3] = self.xyxy[:, 3] / screen_height
        return Detections(
            xyxy=self.xyxy,
            confidence=self.confidence,
            backend_name=self.backend_name,
            label=self.label,
            class_id=self.class_id,
            normalized_xyxy=norm,
            data=self.data,
        )

    # ------------------------------------------------------------------
    # Conversion back to DetectionResult list
    # ------------------------------------------------------------------

    def to_detection_results(self) -> list:
        """Convert back to a list of DetectionResult dataclasses.

        This enables backwards compatibility with code that expects the
        original DetectionResult format.

        Returns:
            List of DetectionResult instances.
        """
        from .backends.base import DetectionResult

        results: list[DetectionResult] = []
        metadata_list = self.data.get("metadata")

        for i in range(len(self)):
            x1, y1, x2, y2 = self.xyxy[i]
            w = int(x2 - x1)
            h = int(y2 - y1)

            nx: float | None = None
            ny: float | None = None
            nw: float | None = None
            nh: float | None = None
            if self.normalized_xyxy is not None:
                nx1, ny1, nx2, ny2 = self.normalized_xyxy[i]
                nx = float(nx1)
                ny = float(ny1)
                nw = float(nx2 - nx1)
                nh = float(ny2 - ny1)

            meta: dict[str, Any] = {}
            if metadata_list is not None and i < len(metadata_list):
                m = metadata_list[i]
                if isinstance(m, dict):
                    meta = m

            results.append(
                DetectionResult(
                    x=int(x1),
                    y=int(y1),
                    width=w,
                    height=h,
                    confidence=float(self.confidence[i]),
                    backend_name=str(self.backend_name[i]),
                    label=(
                        str(self.label[i])
                        if self.label is not None and self.label[i] is not None
                        else None
                    ),
                    metadata=meta,
                    normalized_x=nx,
                    normalized_y=ny,
                    normalized_width=nw,
                    normalized_height=nh,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fields = [f"n={len(self)}"]
        if len(self) > 0:
            fields.append(f"confidence=[{self.confidence.min():.2f}..{self.confidence.max():.2f}]")
            unique_backends = np.unique(self.backend_name)
            fields.append(f"backends={list(unique_backends)}")
        return f"Detections({', '.join(fields)})"


# ======================================================================
# Private helpers
# ======================================================================


def _iou_matrix(boxes_a: NDArray, boxes_b: NDArray) -> NDArray[np.floating]:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (M, 4) array.
        boxes_b: (N, 4) array.

    Returns:
        (M, N) IoU matrix.
    """
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms(
    xyxy: NDArray[np.int_],
    confidence: NDArray[np.floating],
    iou_threshold: float,
) -> list[int]:
    """Greedy non-maximum suppression.

    Args:
        xyxy: (n, 4) bounding boxes.
        confidence: (n,) confidence scores.
        iou_threshold: Overlap threshold above which lower-confidence boxes
            are suppressed.

    Returns:
        List of indices to keep.
    """
    order = np.argsort(confidence)[::-1].tolist()
    keep: list[int] = []

    while order:
        i = order.pop(0)
        keep.append(i)

        if not order:
            break

        remaining = np.array(order)
        ious = _iou_matrix(xyxy[np.array([i])], xyxy[remaining])[0]
        order = [order[j] for j in range(len(order)) if ious[j] < iou_threshold]

    return keep
