"""ProcessorManager - Orchestrates multiple semantic processors."""

from __future__ import annotations

import concurrent.futures
from enum import Enum
from typing import Any

import numpy as np

from ...model.element.region import Region
from ..core import SemanticScene
from .base import ProcessingHints, SemanticProcessor


class ProcessingStrategy(Enum):
    """Processing strategies for combining multiple processors."""

    BEST = "best"  # Use the best processor for the context
    ALL = "all"  # Use all processors and merge results
    ADAPTIVE = "adaptive"  # Start fast, refine uncertain areas
    SEQUENTIAL = "sequential"  # Run processors in sequence
    PARALLEL = "parallel"  # Run processors in parallel


class ProcessorManager:
    """Manages multiple processors and processing strategies.

    Orchestrates different semantic processors to provide flexible
    and optimized scene analysis.
    """

    def __init__(self) -> None:
        """Initialize the processor manager."""
        self.processors: dict[str, SemanticProcessor] = {}
        self.strategy = ProcessingStrategy.BEST
        self._default_processor: str | None = None

    def register_processor(self, name: str, processor: SemanticProcessor) -> None:
        """Register a processor.

        Args:
            name: Name to identify the processor
            processor: SemanticProcessor instance
        """
        self.processors[name] = processor

        # Set as default if it's the first
        if self._default_processor is None:
            self._default_processor = name

    def unregister_processor(self, name: str) -> bool:
        """Unregister a processor.

        Args:
            name: Name of processor to remove

        Returns:
            True if processor was removed
        """
        if name in self.processors:
            del self.processors[name]
            if self._default_processor == name:
                self._default_processor = next(iter(self.processors.keys()), None)
            return True
        return False

    def set_default_processor(self, name: str) -> bool:
        """Set the default processor.

        Args:
            name: Name of processor to use as default

        Returns:
            True if processor exists and was set
        """
        if name in self.processors:
            self._default_processor = name
            return True
        return False

    def set_strategy(self, strategy: ProcessingStrategy) -> None:
        """Set the processing strategy.

        Args:
            strategy: ProcessingStrategy to use
        """
        self.strategy = strategy

    def process(
        self,
        screenshot: np.ndarray[Any, Any],
        strategy: ProcessingStrategy | None = None,
        hints: ProcessingHints | None = None,
    ) -> SemanticScene:
        """Process screenshot using configured strategy.

        Args:
            screenshot: Screenshot to process
            strategy: Optional override for processing strategy
            hints: Optional processing hints

        Returns:
            SemanticScene with detected objects
        """
        if not self.processors:
            return SemanticScene(source_image=screenshot)

        strategy = strategy or self.strategy

        if strategy == ProcessingStrategy.BEST:
            return self.process_with_best_processor(screenshot, hints)
        elif strategy == ProcessingStrategy.ALL:
            return self.process_with_all(screenshot, hints)
        elif strategy == ProcessingStrategy.ADAPTIVE:
            return self.process_adaptive(screenshot, hints)
        elif strategy == ProcessingStrategy.SEQUENTIAL:
            return self.process_sequential(screenshot, hints)
        elif strategy == ProcessingStrategy.PARALLEL:
            return self.process_parallel(screenshot, hints)
        else:
            # Default to best processor
            return self.process_with_best_processor(screenshot, hints)

    def process_with_best_processor(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints | None = None
    ) -> SemanticScene:
        """Process with the best processor for the context.

        Args:
            screenshot: Screenshot to process
            hints: Optional processing hints

        Returns:
            SemanticScene from best processor
        """
        processor_name = self._select_best_processor(hints)
        processor = self.processors.get(processor_name)

        if processor is None:
            return SemanticScene(source_image=screenshot)

        if hints:
            return processor.process_with_hints(screenshot, hints)
        else:
            return processor.process(screenshot)

    def _select_best_processor(self, hints: ProcessingHints | None) -> str:
        """Select the best processor based on hints.

        Args:
            hints: Processing hints

        Returns:
            Name of best processor
        """
        if not hints or not hints.context:
            return self._default_processor or next(iter(self.processors.keys()))

        # Context-based selection
        context_mapping = {
            "game_inventory": "ocr",  # OCR good for item names
            "web_page": "ocr",  # OCR for text content
            "desktop_app": "ocr",  # OCR for menus and labels
            "dialog": "ocr",  # OCR for button text
        }

        suggested = context_mapping.get(hints.context)
        if suggested and suggested in self.processors:
            return suggested

        # Check if any processor supports the expected types
        if hints.expected_object_types:
            for name, processor in self.processors.items():
                supported = processor.get_supported_object_types()
                if any(obj_type in supported for obj_type in hints.expected_object_types):
                    return name

        return self._default_processor or next(iter(self.processors.keys()))

    def process_with_all(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints | None = None
    ) -> SemanticScene:
        """Process with all processors and merge results.

        Args:
            screenshot: Screenshot to process
            hints: Optional processing hints

        Returns:
            Merged SemanticScene from all processors
        """
        scenes = []

        for processor in self.processors.values():
            if hints:
                scene = processor.process_with_hints(screenshot, hints)
            else:
                scene = processor.process(screenshot)
            scenes.append(scene)

        return self._merge_scenes(scenes, screenshot)

    def process_sequential(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints | None = None
    ) -> SemanticScene:
        """Process sequentially, passing results between processors.

        Args:
            screenshot: Screenshot to process
            hints: Optional processing hints

        Returns:
            SemanticScene with accumulated results
        """
        if not self.processors:
            return SemanticScene(source_image=screenshot)

        # Start with first processor
        current_scene = None
        current_hints = hints

        for processor in self.processors.values():
            if current_scene:
                # Update hints with previous scene
                if current_hints:
                    current_hints.previous_scene = current_scene
                else:
                    current_hints = ProcessingHints(previous_scene=current_scene)

            if current_hints:
                scene = processor.process_with_hints(screenshot, current_hints)
            else:
                scene = processor.process(screenshot)

            if current_scene:
                # Merge with previous results
                current_scene = self._merge_scenes([current_scene, scene], screenshot)
            else:
                current_scene = scene

        return current_scene or SemanticScene(source_image=screenshot)

    def process_parallel(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints | None = None
    ) -> SemanticScene:
        """Process with all processors in parallel.

        Args:
            screenshot: Screenshot to process
            hints: Optional processing hints

        Returns:
            Merged SemanticScene from all processors
        """
        if not self.processors:
            return SemanticScene(source_image=screenshot)

        scenes = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.processors)) as executor:
            # Submit all processing tasks
            futures = {}
            for name, processor in self.processors.items():
                if hints:
                    future = executor.submit(processor.process_with_hints, screenshot, hints)
                else:
                    future = executor.submit(processor.process, screenshot)
                futures[future] = name

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    scene = future.result(timeout=10)  # 10 second timeout
                    scenes.append(scene)
                except Exception as e:
                    print(f"Processor {futures[future]} failed: {e}")

        return self._merge_scenes(scenes, screenshot)

    def process_adaptive(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints | None = None
    ) -> SemanticScene:
        """Adaptive processing - start fast, refine uncertain areas.

        Args:
            screenshot: Screenshot to process
            hints: Optional processing hints

        Returns:
            SemanticScene with adaptive refinement
        """
        # Start with fast processor
        fast_processor = self._get_fastest_processor()
        if not fast_processor:
            return self.process_with_best_processor(screenshot, hints)

        # Quick hints for initial pass
        quick_hints = ProcessingHints(quick_mode=True) if not hints else hints
        quick_hints.quick_mode = True

        initial_scene = fast_processor.process_with_hints(screenshot, quick_hints)

        # Identify uncertain regions (low confidence objects)
        uncertain_regions = self._identify_uncertain_regions(initial_scene)

        if not uncertain_regions:
            return initial_scene

        # Use detailed processor on uncertain regions
        detailed_processor = self._get_most_accurate_processor()
        if not detailed_processor or detailed_processor == fast_processor:
            return initial_scene

        # Process each uncertain region
        for region in uncertain_regions:
            detailed_scene = detailed_processor.process_region(screenshot, region)

            # Merge detailed results back
            initial_scene = self._merge_scenes([initial_scene, detailed_scene], screenshot)

        return initial_scene

    def _get_fastest_processor(self) -> SemanticProcessor | None:
        """Get the processor with lowest average processing time.

        Returns:
            Fastest processor or None
        """
        fastest = None
        min_time = float("inf")

        for processor in self.processors.values():
            avg_time = processor.get_average_processing_time()
            if avg_time > 0 and avg_time < min_time:
                min_time = avg_time
                fastest = processor

        # If no timing data, return first processor
        if fastest is None and self.processors:
            fastest = next(iter(self.processors.values()))

        return fastest

    def _get_most_accurate_processor(self) -> SemanticProcessor | None:
        """Get the most accurate processor (typically non-OCR).

        Returns:
            Most accurate processor or None
        """
        # For now, return the default or first non-OCR processor
        # In a full implementation, this could track accuracy metrics
        for name, processor in self.processors.items():
            if "ocr" not in name.lower():
                return processor

        return self.processors.get(self._default_processor) if self._default_processor else None

    def _identify_uncertain_regions(
        self, scene: SemanticScene, confidence_threshold: float = 0.7
    ) -> list[Region]:
        """Identify regions with low-confidence detections.

        Args:
            scene: Scene to analyze
            confidence_threshold: Threshold for uncertainty

        Returns:
            List of regions needing refinement
        """
        uncertain_regions = []

        for obj in scene.objects:
            if obj.confidence < confidence_threshold:
                # Add some padding around uncertain object
                box = obj.get_bounding_box()
                padded = Region(
                    x=max(0, box.x - 10),
                    y=max(0, box.y - 10),
                    width=box.width + 20,
                    height=box.height + 20,
                )
                uncertain_regions.append(padded)

        # Merge overlapping regions
        return self._merge_overlapping_regions(uncertain_regions)

    def _merge_overlapping_regions(self, regions: list[Region]) -> list[Region]:
        """Merge overlapping regions.

        Args:
            regions: List of regions

        Returns:
            List of merged regions
        """
        if not regions:
            return []

        merged = []
        used = set()

        for i, region1 in enumerate(regions):
            if i in used:
                continue

            # Start with this region
            x1, y1 = region1.x, region1.y
            x2 = region1.x + region1.width
            y2 = region1.y + region1.height

            # Merge with overlapping regions
            for j, region2 in enumerate(regions[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check for overlap
                if (
                    region2.x < x2
                    and region2.x + region2.width > x1
                    and region2.y < y2
                    and region2.y + region2.height > y1
                ):
                    # Merge
                    x1 = min(x1, region2.x)
                    y1 = min(y1, region2.y)
                    x2 = max(x2, region2.x + region2.width)
                    y2 = max(y2, region2.y + region2.height)
                    used.add(j)

            merged.append(Region(x1, y1, x2 - x1, y2 - y1))
            used.add(i)

        return merged

    def _merge_scenes(
        self, scenes: list[SemanticScene], source_image: np.ndarray[Any, Any]
    ) -> SemanticScene:
        """Merge multiple scenes into one.

        Args:
            scenes: List of scenes to merge
            source_image: Original screenshot

        Returns:
            Merged SemanticScene
        """
        if not scenes:
            return SemanticScene(source_image=source_image)

        if len(scenes) == 1:
            return scenes[0]

        merged = SemanticScene(source_image=source_image)

        # Collect all objects
        all_objects = []
        for scene in scenes:
            all_objects.extend(scene.objects)

        # Remove duplicates based on location and type
        seen = set()
        for obj in all_objects:
            # Create a key based on approximate location and type
            centroid = obj.location.get_centroid()
            key = (centroid.x // 10, centroid.y // 10, obj.object_type)

            if key not in seen:
                merged.add_object(obj)
                seen.add(key)

        return merged
