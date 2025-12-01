"""Pattern loading from configuration.

Handles loading patterns from config, distinguishing between state images
and direct image references.
"""

import logging
from pathlib import Path
from typing import Any

from ..model.element import Pattern
from .pattern_builder import PatternBuilder

logger = logging.getLogger(__name__)


class PatternLoader:
    """Loads Pattern objects from configuration.

    Responsibilities:
    - Distinguish state images from direct images
    - Navigate config.states hierarchy for state image lookup
    - Delegate pattern building to PatternBuilder
    """

    def __init__(self, config: Any, pattern_builder: PatternBuilder):
        """Initialize loader with configuration.

        Args:
            config: Configuration object with states and images
            pattern_builder: PatternBuilder for creating Pattern objects
        """
        self.config = config
        self.pattern_builder = pattern_builder

    def load_patterns(self, image_ids: list[str]) -> list[Pattern]:
        """Load multiple patterns from config.

        Args:
            image_ids: List of image IDs to load

        Returns:
            List of Pattern objects (may be empty if none found)
        """
        import os
        import tempfile
        from datetime import datetime

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(
                    tempfile.gettempdir(), "qontinui_find_debug.log"
                )
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] PATTERN_LOADER: {msg}\n")
            except Exception:
                pass

        log_debug("load_patterns() called")
        log_debug(f"  image_ids: {image_ids}")
        log_debug(f"  config type: {type(self.config)}")
        log_debug(f"  config has 'states': {hasattr(self.config, 'states')}")
        log_debug(f"  config has 'images': {hasattr(self.config, 'images')}")

        if hasattr(self.config, "states"):
            log_debug(f"  config.states: {self.config.states}")
            log_debug(
                f"  config.states count: {len(self.config.states) if self.config.states else 0}"
            )

        if hasattr(self.config, "images"):
            log_debug(
                f"  config.images count: {len(self.config.images) if self.config.images else 0}"
            )

        patterns = []

        for image_id in image_ids:
            log_debug(f"  Loading pattern for image_id: {image_id}")
            loaded_patterns = self.load_pattern(image_id)
            log_debug(f"    Got {len(loaded_patterns)} patterns")
            patterns.extend(loaded_patterns)

        log_debug(f"  Total patterns loaded: {len(patterns)}")
        logger.debug(f"Loaded {len(patterns)} patterns from {len(image_ids)} image IDs")
        return patterns

    def load_pattern(self, image_id: str) -> list[Pattern]:
        """Load pattern(s) for a single image ID.

        State images may have multiple pattern variations, so this returns a list.

        Args:
            image_id: Image ID to load

        Returns:
            List of Pattern objects for this image ID
        """
        # Check if this is a state image
        if self._is_state_image_id(image_id):
            return self._load_state_image_patterns(image_id)
        else:
            pattern = self._load_direct_image_pattern(image_id)
            return [pattern] if pattern else []

    def _is_state_image_id(self, image_id: str) -> bool:
        """Check if image_id refers to a state image.

        Args:
            image_id: Image ID to check

        Returns:
            True if ID matches a state image pattern
        """
        import os
        import tempfile
        from datetime import datetime

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                debug_log = os.path.join(
                    tempfile.gettempdir(), "qontinui_find_debug.log"
                )
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] PATTERN_LOADER: {msg}\n")
            except Exception:
                pass

        log_debug(f"_is_state_image_id() checking: {image_id}")

        if not hasattr(self.config, "states") or not self.config.states:
            log_debug("  Config has no states or states is empty")
            return False

        log_debug(f"  Checking {len(self.config.states)} states")

        # Search all states for this image ID
        for i, state in enumerate(self.config.states):
            log_debug(f"    State {i}: {state.id if hasattr(state, 'id') else 'no-id'}")

            if not hasattr(state, "state_images") or not state.state_images:
                log_debug("      No state_images")
                continue

            log_debug(f"      Has {len(state.state_images)} state_images")
            for j, state_image in enumerate(state.state_images):
                log_debug(f"        State image {j}: id={state_image.id}")
                if state_image.id == image_id:
                    log_debug("        MATCH! Returning True")
                    return True

        log_debug("  No match found, returning False")
        return False

    def _load_state_image_patterns(self, image_id: str) -> list[Pattern]:
        """Load all pattern variations for a state image.

        Args:
            image_id: State image ID

        Returns:
            List of Pattern objects for all variations
        """
        patterns = []

        for state in self.config.states:
            if not hasattr(state, "state_images") or not state.state_images:
                continue

            for state_image in state.state_images:
                if state_image.id != image_id:
                    continue

                # Load all patterns for this state image
                for i, pattern_config in enumerate(state_image.patterns):
                    pattern_id = f"{image_id}_variation_{i}"
                    pattern = self._build_pattern_from_config(
                        pattern_config=pattern_config.model_dump(), image_id=pattern_id
                    )
                    if pattern:
                        patterns.append(pattern)

        if not patterns:
            logger.warning(f"No patterns loaded for state image: {image_id}")

        return patterns

    def _load_direct_image_pattern(self, image_id: str) -> Pattern | None:
        """Load pattern for a direct image reference.

        Args:
            image_id: Direct image ID

        Returns:
            Pattern object or None if not found
        """
        # Search in config.images
        if hasattr(self.config, "images") and self.config.images:
            for image in self.config.images:
                if image.id == image_id:
                    return self._build_pattern_from_config(
                        pattern_config=image.model_dump(), image_id=image_id
                    )

        logger.error(f"Image not found in config: {image_id}")
        return None

    def _build_pattern_from_config(
        self, pattern_config: dict, image_id: str
    ) -> Pattern | None:
        """Build pattern from config dict.

        Args:
            pattern_config: Dict with 'path' and optional 'mask'
            image_id: Identifier for this pattern

        Returns:
            Pattern object or None if building fails
        """
        # Get image assets directory from config
        if hasattr(self.config, "image_assets_dir"):
            image_assets_dir = Path(self.config.image_assets_dir)
        else:
            logger.error("Config missing image_assets_dir")
            return None

        return self.pattern_builder.build_from_config(
            pattern_config=pattern_config,
            image_id=image_id,
            image_assets_dir=image_assets_dir,
        )
