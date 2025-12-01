"""
EXPERIMENTAL: Hybrid detector combining multiple detection strategies.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector combines multiple computer vision detection methods and uses
a voting/consensus mechanism to improve detection accuracy. It's experimental
and performance may vary significantly depending on the UI being analyzed.
"""

from typing import Any

from .base_detector import BaseDetector
from .contour_detector import ContourDetector
from .edge_detector import EdgeBasedDetector
from .mser_detector import MSERDetector
from .types import BBox


class HybridDetector(BaseDetector):
    """Combine multiple detection strategies using consensus voting.

    EXPERIMENTAL: This detector is experimental and combines edge detection,
    contour detection, and MSER detection using a voting mechanism.

    The hybrid approach attempts to improve precision by only keeping
    detections that are confirmed by multiple detection methods.
    """

    def __init__(self):
        super().__init__("Hybrid Detector")
        self.edge_detector = EdgeBasedDetector()
        self.contour_detector = ContourDetector()
        self.mser_detector = MSERDetector()

    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect elements using multiple methods and combine results

        Parameters:
            use_edge: Use edge detector (default: True)
            use_contour: Use contour detector (default: True)
            use_mser: Use MSER detector (default: True)
            consensus_threshold: Minimum overlap to keep (default: 2)
            merge_threshold: IoU threshold for merging (default: 0.7)
        """
        use_edge = params.get("use_edge", True)
        use_contour = params.get("use_contour", True)
        use_mser = params.get("use_mser", True)
        consensus_threshold = params.get("consensus_threshold", 2)
        merge_threshold = params.get("merge_threshold", 0.7)

        all_boxes = []

        # Run each detector
        if use_edge:
            edge_params = params.get("edge_params", {})
            edge_boxes = self.edge_detector.detect(image_path, **edge_params)
            all_boxes.extend([(box, "edge") for box in edge_boxes])

        if use_contour:
            contour_params = params.get("contour_params", {})
            contour_boxes = self.contour_detector.detect(image_path, **contour_params)
            all_boxes.extend([(box, "contour") for box in contour_boxes])

        if use_mser:
            mser_params = params.get("mser_params", {})
            mser_boxes = self.mser_detector.detect(image_path, **mser_params)
            all_boxes.extend([(box, "mser") for box in mser_boxes])

        if not all_boxes:
            return []

        # Voting/consensus approach
        boxes_only = [box for box, _ in all_boxes]

        # Group overlapping boxes
        groups = self._group_overlapping_boxes(boxes_only, iou_threshold=0.5)

        # Keep groups that have consensus
        final_boxes = []
        for group in groups:
            if len(group) >= consensus_threshold:
                # Merge boxes in group
                merged = self._merge_box_group(group)
                final_boxes.append(merged)

        # Final merge of very similar boxes
        final_boxes = self.merge_overlapping_boxes(
            final_boxes, iou_threshold=merge_threshold
        )
        final_boxes = self.remove_contained_boxes(final_boxes)

        return final_boxes

    def _group_overlapping_boxes(
        self, boxes: list[BBox], iou_threshold: float
    ) -> list[list[BBox]]:
        """Group boxes that overlap significantly"""
        if not boxes:
            return []

        groups = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            group = [box1]
            used.add(i)

            for j, box2 in enumerate(boxes[i + 1 :], start=i + 1):
                if j in used:
                    continue

                # Check if box2 overlaps with any box in current group
                for box_in_group in group:
                    iou = self._compute_iou(box_in_group, box2)
                    if iou >= iou_threshold:
                        group.append(box2)
                        used.add(j)
                        break

            groups.append(group)

        return groups

    def _compute_iou(self, box1: BBox, box2: BBox) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_box_group(self, group: list[BBox]) -> BBox:
        """Merge a group of boxes by taking bounding box"""
        min_x = min(b.x1 for b in group)
        min_y = min(b.y1 for b in group)
        max_x = max(b.x2 for b in group)
        max_y = max(b.y2 for b in group)
        return BBox(min_x, min_y, max_x, max_y)

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Parameter grid for hyperparameter search"""
        return [
            # All methods, different consensus levels
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 2,
            },
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 3,
            },
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 1,
            },
            # Pairs of methods
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": False,
                "consensus_threshold": 2,
            },
            {
                "use_edge": True,
                "use_contour": False,
                "use_mser": True,
                "consensus_threshold": 2,
            },
            {
                "use_edge": False,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 2,
            },
            # Different merge thresholds
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 2,
                "merge_threshold": 0.5,
            },
            {
                "use_edge": True,
                "use_contour": True,
                "use_mser": True,
                "consensus_threshold": 2,
                "merge_threshold": 0.8,
            },
        ]
