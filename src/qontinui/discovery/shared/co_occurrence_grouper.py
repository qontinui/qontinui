"""Shared Co-occurrence Grouper for state machine creation.

This module provides a unified consumer for co-occurrence data from both:
1. Extraction feature (StateImages with screenshot_ids populated during extraction)
2. Input Capture feature (templates with frame presence detected via template matching)

The grouper is agnostic to the source - it just needs to know which images/templates
appear in which frames/screenshots.

Data Format:
    image_frame_map: dict[str, set[int]]
        Maps image_id to the set of frame/screenshot indices where it appears.
        Example: {"img_001": {0, 1, 5, 6}, "img_002": {0, 1, 5}}

This format can be produced by:
- Extraction: Directly from StateImage.screenshot_ids
- Input Capture: By inverting CoOccurrenceAnalyzer's frame_template_map
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CoOccurrenceGroup:
    """A group of images that co-occur (appear together).

    Attributes:
        group_id: Unique identifier for this group
        image_ids: Set of image IDs in this group
        common_frames: Set of frame indices where ALL images appear together
        confidence: Confidence score for this grouping (based on co-occurrence ratio)
        metadata: Additional metadata
    """

    group_id: str
    image_ids: set[str]
    common_frames: set[int]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_count(self) -> int:
        return len(self.image_ids)

    @property
    def frame_count(self) -> int:
        return len(self.common_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "image_ids": list(self.image_ids),
            "common_frames": list(self.common_frames),
            "confidence": self.confidence,
            "image_count": self.image_count,
            "frame_count": self.frame_count,
            "metadata": self.metadata,
        }


@dataclass
class CoOccurrenceGroupingResult:
    """Result of co-occurrence grouping.

    Attributes:
        groups: List of image groups
        ungrouped_images: Images not assigned to any group
        co_occurrence_matrix: NxN matrix of co-occurrence ratios
        image_ids: List of image IDs (index matches matrix position)
        total_frames: Total number of frames analyzed
        processing_time_ms: Processing time
    """

    groups: list[CoOccurrenceGroup] = field(default_factory=list)
    ungrouped_images: set[str] = field(default_factory=set)
    co_occurrence_matrix: np.ndarray | None = None
    image_ids: list[str] = field(default_factory=list)
    total_frames: int = 0
    processing_time_ms: float = 0.0

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "ungrouped_images": list(self.ungrouped_images),
            "image_ids": self.image_ids,
            "total_frames": self.total_frames,
            "group_count": self.group_count,
            "processing_time_ms": self.processing_time_ms,
        }


class CoOccurrenceGrouper:
    """Groups images by co-occurrence patterns.

    This is the shared consumer for both extraction and input capture features.
    It takes a mapping of image_id -> frame_ids and groups images that appear
    together.

    Two grouping strategies:
    1. Strict: Images must appear in EXACTLY the same frames
    2. Threshold: Images that co-occur above a threshold ratio

    Example:
        >>> grouper = CoOccurrenceGrouper(threshold=0.8)
        >>> # From extraction (StateImages)
        >>> image_frame_map = {si.id: set(si.screenshot_ids) for si in state_images}
        >>> result = grouper.group(image_frame_map)
        >>>
        >>> # From input capture (after template matching)
        >>> image_frame_map = invert_frame_template_map(frame_template_map)
        >>> result = grouper.group(image_frame_map)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the grouper.

        Args:
            threshold: Minimum co-occurrence ratio to group images (0-1).
                Two images are grouped if they appear together in at least
                this fraction of frames where either appears.
            strict_mode: If True, only group images that appear in EXACTLY
                the same frames (ignores threshold).
        """
        self.threshold = threshold
        self.strict_mode = strict_mode

    def group(
        self,
        image_frame_map: dict[str, set[int]],
        total_frames: int | None = None,
    ) -> CoOccurrenceGroupingResult:
        """Group images by co-occurrence.

        Args:
            image_frame_map: Maps image_id to set of frame indices where
                the image appears.
            total_frames: Total number of frames (for metadata). If not
                provided, inferred from the data.

        Returns:
            CoOccurrenceGroupingResult with groups and metadata
        """
        import time

        start_time = time.time()

        if not image_frame_map:
            return CoOccurrenceGroupingResult(processing_time_ms=0.0)

        image_ids = list(image_frame_map.keys())

        # Infer total frames if not provided
        if total_frames is None:
            all_frames: set[int] = set()
            for frames in image_frame_map.values():
                all_frames.update(frames)
            total_frames = len(all_frames) if all_frames else 0

        # Build co-occurrence matrix
        co_occurrence_ratio = self._build_co_occurrence_matrix(image_frame_map, image_ids)

        # Group images
        if self.strict_mode:
            groups = self._group_strict(image_frame_map, image_ids)
        else:
            groups = self._group_by_threshold(image_frame_map, image_ids, co_occurrence_ratio)

        # Find ungrouped images
        grouped_ids: set[str] = set()
        for group in groups:
            grouped_ids.update(group.image_ids)
        ungrouped = set(image_ids) - grouped_ids

        processing_time = (time.time() - start_time) * 1000

        return CoOccurrenceGroupingResult(
            groups=groups,
            ungrouped_images=ungrouped,
            co_occurrence_matrix=co_occurrence_ratio,
            image_ids=image_ids,
            total_frames=total_frames,
            processing_time_ms=processing_time,
        )

    def _build_co_occurrence_matrix(
        self,
        image_frame_map: dict[str, set[int]],
        image_ids: list[str],
    ) -> np.ndarray:
        """Build co-occurrence ratio matrix.

        Returns NxN matrix where [i,j] is the co-occurrence ratio:
        count(frames where both appear) / min(count_i, count_j)
        """
        n = len(image_ids)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i, id_i in enumerate(image_ids):
            frames_i = image_frame_map[id_i]
            count_i = len(frames_i)

            for j, id_j in enumerate(image_ids):
                if i == j:
                    matrix[i, j] = 1.0
                    continue

                frames_j = image_frame_map[id_j]
                count_j = len(frames_j)

                if count_i == 0 or count_j == 0:
                    matrix[i, j] = 0.0
                    continue

                common = len(frames_i & frames_j)
                matrix[i, j] = common / min(count_i, count_j)

        return matrix

    def _group_strict(
        self,
        image_frame_map: dict[str, set[int]],
        image_ids: list[str],
    ) -> list[CoOccurrenceGroup]:
        """Group images that appear in EXACTLY the same frames."""
        groups: list[CoOccurrenceGroup] = []
        used: set[str] = set()

        for i, id_i in enumerate(image_ids):
            if id_i in used:
                continue

            frames_i = image_frame_map[id_i]
            group_ids = {id_i}
            used.add(id_i)

            for j in range(i + 1, len(image_ids)):
                id_j = image_ids[j]
                if id_j in used:
                    continue

                frames_j = image_frame_map[id_j]

                # Strict: must be exactly the same frames
                if frames_i == frames_j:
                    group_ids.add(id_j)
                    used.add(id_j)

            groups.append(
                CoOccurrenceGroup(
                    group_id=f"group_{len(groups)}",
                    image_ids=group_ids,
                    common_frames=frames_i.copy(),
                    confidence=1.0,  # Strict mode = perfect match
                    metadata={"grouping_mode": "strict"},
                )
            )

        return groups

    def _group_by_threshold(
        self,
        image_frame_map: dict[str, set[int]],
        image_ids: list[str],
        co_occurrence_ratio: np.ndarray,
    ) -> list[CoOccurrenceGroup]:
        """Group images by co-occurrence threshold."""
        groups: list[CoOccurrenceGroup] = []
        used: set[str] = set()
        id_to_idx = {id_: i for i, id_ in enumerate(image_ids)}

        for i, id_i in enumerate(image_ids):
            if id_i in used:
                continue

            frames_i = image_frame_map[id_i]
            group_ids = {id_i}
            used.add(id_i)
            common_frames = frames_i.copy()

            # Find all images that co-occur above threshold
            for j in range(i + 1, len(image_ids)):
                id_j = image_ids[j]
                if id_j in used:
                    continue

                if co_occurrence_ratio[i, j] >= self.threshold:
                    group_ids.add(id_j)
                    used.add(id_j)
                    # Common frames = intersection of all members
                    common_frames &= image_frame_map[id_j]

            # Calculate average confidence for the group
            if len(group_ids) > 1:
                total_ratio = 0.0
                count = 0
                group_list = list(group_ids)
                for gi in range(len(group_list)):
                    for gj in range(gi + 1, len(group_list)):
                        idx_i = id_to_idx[group_list[gi]]
                        idx_j = id_to_idx[group_list[gj]]
                        total_ratio += co_occurrence_ratio[idx_i, idx_j]
                        count += 1
                confidence = total_ratio / count if count > 0 else 1.0
            else:
                confidence = 1.0

            groups.append(
                CoOccurrenceGroup(
                    group_id=f"group_{len(groups)}",
                    image_ids=group_ids,
                    common_frames=common_frames,
                    confidence=confidence,
                    metadata={
                        "grouping_mode": "threshold",
                        "threshold": self.threshold,
                    },
                )
            )

        return groups


def invert_frame_template_map(frame_template_map: dict[int, set[str]]) -> dict[str, set[int]]:
    """Convert frame→templates to template→frames mapping.

    Input Capture's CoOccurrenceAnalyzer produces frame_template_map.
    This function converts it to the format needed by CoOccurrenceGrouper.

    Args:
        frame_template_map: Maps frame_number to set of template_ids present

    Returns:
        Maps template_id to set of frame_numbers where it appears
    """
    template_frame_map: dict[str, set[int]] = defaultdict(set)

    for frame_num, template_ids in frame_template_map.items():
        for template_id in template_ids:
            template_frame_map[template_id].add(frame_num)

    return dict(template_frame_map)


def from_state_images_screenshot_ids(
    state_images: list[Any],  # list[StateImage] - using Any to avoid import
) -> dict[str, set[int]]:
    """Convert StateImages to the format needed by CoOccurrenceGrouper.

    Extraction feature produces StateImages with screenshot_ids populated.
    This converts them to the common format.

    Args:
        state_images: List of StateImage objects with screenshot_ids

    Returns:
        Maps image_id to set of screenshot indices
    """
    image_frame_map: dict[str, set[int]] = {}

    for si in state_images:
        # screenshot_ids are strings, convert to int indices if needed
        frame_ids: set[int] = set()
        for sid in si.screenshot_ids:
            if isinstance(sid, int):
                frame_ids.add(sid)
            elif isinstance(sid, str) and sid.isdigit():
                frame_ids.add(int(sid))
            else:
                # Use hash for non-numeric IDs
                frame_ids.add(hash(sid) % 10000)

        image_frame_map[si.id] = frame_ids

    return image_frame_map
