"""Co-occurrence Analyzer for detecting template presence across video frames.

This module analyzes a video to determine which approved templates appear
together in the same frames. This co-occurrence data is essential for
grouping templates into states based on their visual co-presence.

The key insight: templates that frequently appear together in the same
frames likely belong to the same state (or related states that are
often active simultaneously).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .approved_template import ApprovedTemplate


@dataclass
class CoOccurrenceResult:
    """Result of co-occurrence analysis.

    Attributes:
        frame_template_map: Dict mapping frame_number to set of template IDs
            present in that frame
        template_frame_counts: Dict mapping template_id to number of frames
            it appears in
        co_occurrence_matrix: NxN matrix where [i,j] is count of frames where
            templates i and j both appear
        template_ids: List of template IDs (index corresponds to matrix position)
        frames_analyzed: Total number of frames analyzed
        processing_time_ms: Time taken to process
        metadata: Additional analysis metadata
    """

    frame_template_map: dict[int, set[str]] = field(default_factory=dict)
    template_frame_counts: dict[str, int] = field(default_factory=dict)
    co_occurrence_matrix: np.ndarray | None = None
    template_ids: list[str] = field(default_factory=list)
    frames_analyzed: int = 0
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_co_occurrence_ratio(self, template_id_1: str, template_id_2: str) -> float:
        """Get co-occurrence ratio between two templates.

        Returns the fraction of frames where both templates appear together,
        relative to the minimum appearance count of either template.

        Args:
            template_id_1: First template ID
            template_id_2: Second template ID

        Returns:
            Co-occurrence ratio (0-1), or 0 if either template not found
        """
        if self.co_occurrence_matrix is None:
            return 0.0

        if template_id_1 not in self.template_ids or template_id_2 not in self.template_ids:
            return 0.0

        i = self.template_ids.index(template_id_1)
        j = self.template_ids.index(template_id_2)

        count_1 = self.template_frame_counts.get(template_id_1, 0)
        count_2 = self.template_frame_counts.get(template_id_2, 0)

        if count_1 == 0 or count_2 == 0:
            return 0.0

        co_count = float(self.co_occurrence_matrix[i, j])
        return co_count / min(count_1, count_2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_template_map": {str(k): list(v) for k, v in self.frame_template_map.items()},
            "template_frame_counts": self.template_frame_counts,
            "co_occurrence_matrix": (
                self.co_occurrence_matrix.tolist()
                if self.co_occurrence_matrix is not None
                else None
            ),
            "template_ids": self.template_ids,
            "frames_analyzed": self.frames_analyzed,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class CoOccurrenceAnalyzer:
    """Analyzes video frames to detect template co-occurrence.

    This class scans video frames and uses template matching to detect
    which approved templates are visible in each frame. The resulting
    co-occurrence data can be used to group templates into states.

    Uses parallel processing for efficiency when analyzing many frames.

    Example:
        >>> analyzer = CoOccurrenceAnalyzer(similarity_threshold=0.8)
        >>> result = analyzer.analyze_video(
        ...     video_path=Path("session.mp4"),
        ...     templates=approved_templates,
        ...     sample_interval=30,  # Check every 30 frames
        ... )
        >>> # Use result.frame_template_map with StateGrouper
        >>> grouper = StateGrouper()
        >>> grouping = grouper.group_by_co_occurrence(
        ...     templates=approved_templates,
        ...     frames=result.frame_template_map,
        ... )
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        match_method: int = cv2.TM_CCOEFF_NORMED,
        max_workers: int = 4,
    ) -> None:
        """Initialize the analyzer.

        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider
                a template as present in a frame
            match_method: OpenCV template matching method
            max_workers: Maximum number of parallel workers for frame processing
        """
        self.similarity_threshold = similarity_threshold
        self.match_method = match_method
        self.max_workers = max_workers

    def analyze_video(
        self,
        video_path: Path,
        templates: list[ApprovedTemplate],
        sample_interval: int = 30,
        max_frames: int | None = None,
    ) -> CoOccurrenceResult:
        """Analyze video frames to detect template presence.

        Samples frames from the video at regular intervals and checks
        which templates are visible in each frame using template matching.
        Uses parallel processing for efficiency.

        Args:
            video_path: Path to the video file
            templates: List of approved templates to search for
            sample_interval: Check every N frames (default: 30 = ~1 per second at 30fps)
            max_frames: Maximum number of frames to analyze (None = all)

        Returns:
            CoOccurrenceResult with frame-to-template mapping and co-occurrence matrix
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not templates:
            return CoOccurrenceResult(
                processing_time_ms=0.0,
                metadata={"error": "No templates provided"},
            )

        # Prepare template images for matching
        template_images = self._prepare_template_images(templates)
        if not template_images:
            return CoOccurrenceResult(
                processing_time_ms=0.0,
                metadata={"error": "No valid template images"},
            )

        # Open video and extract frames to analyze
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Extract frames at sample intervals
            frames_to_analyze: list[tuple[int, np.ndarray]] = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    if max_frames and len(frames_to_analyze) >= max_frames:
                        break

                    # Convert to grayscale for matching
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames_to_analyze.append((frame_count, gray_frame))

                frame_count += 1

        finally:
            cap.release()

        if not frames_to_analyze:
            return CoOccurrenceResult(
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "No frames extracted from video"},
            )

        # Process frames in parallel
        frame_template_map: dict[int, set[str]] = {}
        template_frame_counts: dict[str, int] = defaultdict(int)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all frame analysis tasks
            future_to_frame = {
                executor.submit(
                    self._detect_templates_in_frame, gray_frame, template_images
                ): frame_num
                for frame_num, gray_frame in frames_to_analyze
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_num = future_to_frame[future]
                try:
                    present_templates = future.result()
                    if present_templates:
                        frame_template_map[frame_num] = present_templates
                        for tid in present_templates:
                            template_frame_counts[tid] += 1
                except Exception:
                    # Skip frames that fail
                    pass

        # Build co-occurrence matrix
        template_ids = [t.id for t in templates]
        co_occurrence_matrix = self._build_co_occurrence_matrix(frame_template_map, template_ids)

        processing_time = (time.time() - start_time) * 1000

        return CoOccurrenceResult(
            frame_template_map=frame_template_map,
            template_frame_counts=dict(template_frame_counts),
            co_occurrence_matrix=co_occurrence_matrix,
            template_ids=template_ids,
            frames_analyzed=len(frames_to_analyze),
            processing_time_ms=processing_time,
            metadata={
                "video_path": str(video_path),
                "total_video_frames": total_frames,
                "sample_interval": sample_interval,
                "similarity_threshold": self.similarity_threshold,
                "parallel_workers": self.max_workers,
            },
        )

    def analyze_screenshots(
        self,
        screenshots: list[np.ndarray],
        templates: list[ApprovedTemplate],
        frame_numbers: list[int] | None = None,
    ) -> CoOccurrenceResult:
        """Analyze a list of screenshots to detect template presence.

        Alternative to video analysis when screenshots are already available.

        Args:
            screenshots: List of screenshot images (BGR or grayscale)
            templates: List of approved templates to search for
            frame_numbers: Optional frame numbers for each screenshot
                          (defaults to 0, 1, 2, ...)

        Returns:
            CoOccurrenceResult with frame-to-template mapping
        """
        import time

        start_time = time.time()

        if not templates or not screenshots:
            return CoOccurrenceResult(
                processing_time_ms=0.0,
                metadata={"error": "No templates or screenshots provided"},
            )

        # Prepare template images
        template_images = self._prepare_template_images(templates)
        if not template_images:
            return CoOccurrenceResult(
                processing_time_ms=0.0,
                metadata={"error": "No valid template images"},
            )

        if frame_numbers is None:
            frame_numbers = list(range(len(screenshots)))

        frame_template_map: dict[int, set[str]] = {}
        template_frame_counts: dict[str, int] = defaultdict(int)

        for i, screenshot in enumerate(screenshots):
            frame_num = frame_numbers[i]

            # Convert to grayscale if needed
            if len(screenshot.shape) == 3:
                gray_frame = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = screenshot

            # Detect templates
            present_templates = self._detect_templates_in_frame(gray_frame, template_images)

            if present_templates:
                frame_template_map[frame_num] = present_templates
                for tid in present_templates:
                    template_frame_counts[tid] += 1

        # Build co-occurrence matrix
        template_ids = [t.id for t in templates]
        co_occurrence_matrix = self._build_co_occurrence_matrix(frame_template_map, template_ids)

        processing_time = (time.time() - start_time) * 1000

        return CoOccurrenceResult(
            frame_template_map=frame_template_map,
            template_frame_counts=dict(template_frame_counts),
            co_occurrence_matrix=co_occurrence_matrix,
            template_ids=template_ids,
            frames_analyzed=len(screenshots),
            processing_time_ms=processing_time,
            metadata={
                "source": "screenshots",
                "screenshot_count": len(screenshots),
                "similarity_threshold": self.similarity_threshold,
            },
        )

    def _prepare_template_images(self, templates: list[ApprovedTemplate]) -> dict[str, np.ndarray]:
        """Prepare template images for matching.

        Args:
            templates: List of approved templates

        Returns:
            Dict mapping template_id to grayscale template image
        """
        template_images: dict[str, np.ndarray] = {}

        for template in templates:
            if template.pixel_data is None:
                continue

            # Convert to grayscale if needed
            if len(template.pixel_data.shape) == 3:
                gray = cv2.cvtColor(template.pixel_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = template.pixel_data

            # Ensure it's a valid template size
            if gray.shape[0] < 5 or gray.shape[1] < 5:
                continue

            template_images[template.id] = gray

        return template_images

    def _detect_templates_in_frame(
        self,
        frame: np.ndarray,
        template_images: dict[str, np.ndarray],
    ) -> set[str]:
        """Detect which templates are present in a frame.

        Args:
            frame: Grayscale frame image
            template_images: Dict mapping template_id to grayscale template

        Returns:
            Set of template IDs that are detected in the frame
        """
        present_templates: set[str] = set()

        for template_id, template_img in template_images.items():
            # Skip if template is larger than frame
            if template_img.shape[0] > frame.shape[0] or template_img.shape[1] > frame.shape[1]:
                continue

            try:
                result = cv2.matchTemplate(frame, template_img, self.match_method)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val >= self.similarity_threshold:
                    present_templates.add(template_id)

            except cv2.error:
                # Template matching failed (e.g., invalid dimensions)
                continue

        return present_templates

    def _build_co_occurrence_matrix(
        self,
        frame_template_map: dict[int, set[str]],
        template_ids: list[str],
    ) -> np.ndarray:
        """Build co-occurrence matrix from frame-template mapping.

        Args:
            frame_template_map: Dict mapping frame_number to set of template IDs
            template_ids: List of all template IDs (determines matrix order)

        Returns:
            NxN numpy array where [i,j] is count of frames where both
            templates i and j appear
        """
        n = len(template_ids)
        matrix = np.zeros((n, n), dtype=np.int32)

        id_to_idx = {tid: i for i, tid in enumerate(template_ids)}

        for templates_in_frame in frame_template_map.values():
            # Convert to indices
            indices = [id_to_idx[tid] for tid in templates_in_frame if tid in id_to_idx]

            # Increment co-occurrence counts
            for i in indices:
                for j in indices:
                    matrix[i, j] += 1

        return matrix
