"""Adapter layer for converting qontinui-schemas config to runtime types.

This module bridges the gap between:
- Configuration format (qontinui_schemas): Pydantic models with ID references
- Runtime format (qontinui.model): Dataclasses with actual object instances

The adapter converts QontinuiConfig (v2.x format) from qontinui-schemas into
runtime State, StateImage, and Transition objects that can be used by the
qontinui library for automation execution.

Key conversions:
- ImageAsset (config) -> Image (runtime)
- StateImage (config with patterns) -> StateImage (runtime with actual images)
- State (config) -> State (runtime)
- Transition (config) -> StateTransition (runtime)
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

from qontinui import registry
from qontinui.model.element import Image, Region
from qontinui.model.search_regions import SearchRegions
from qontinui.model.state.state import StateBuilder
from qontinui.model.state.state_image import StateImage as RuntimeStateImage
from qontinui.model.state.state_location import StateLocation
from qontinui.model.state.state_region import StateRegion
from qontinui.model.state.state_service import StateService
from qontinui.model.state.state_string import StateString

if TYPE_CHECKING:
    from qontinui_schemas import (
        ImageAsset,
        QontinuiConfig,
        SearchRegion,
    )
    from qontinui_schemas import State as ConfigState
    from qontinui_schemas import StateImage as ConfigStateImage
    from qontinui_schemas import StateLocation as ConfigStateLocation
    from qontinui_schemas import StateRegion as ConfigStateRegion
    from qontinui_schemas import StateString as ConfigStateString

logger = logging.getLogger(__name__)


class ConfigAdapter:
    """Adapter for converting qontinui-schemas config to runtime types.

    This class handles the conversion of QontinuiConfig (v2.x format) from
    qontinui-schemas into runtime objects that can be used by the qontinui
    library for automation execution.

    Usage:
        adapter = ConfigAdapter()
        adapter.load_config(qontinui_config, state_service)

    The adapter:
    1. Registers all ImageAssets in the registry
    2. Creates runtime State objects with proper StateImages
    3. Handles the monitors field for multi-monitor support
    4. Converts patterns with image ID references to actual Pattern objects
    """

    def __init__(self) -> None:
        """Initialize the config adapter."""
        self._image_lookup: dict[str, Image] = {}
        self._state_id_mapping: dict[str, int] = {}
        self._cached_monitor_manager: Any = None

    @property
    def _monitor_manager(self) -> Any:
        """Get or create MonitorManager (lazy initialization)."""
        if self._cached_monitor_manager is None:
            from qontinui.monitor.monitor_manager import MonitorManager

            self._cached_monitor_manager = MonitorManager()
        return self._cached_monitor_manager

    def _get_monitor_for_point(self, x: int, y: int) -> int | None:
        """Detect which monitor contains a point.

        Args:
            x: X coordinate (absolute screen coordinate)
            y: Y coordinate (absolute screen coordinate)

        Returns:
            Monitor index, or None if not found
        """
        result: int = self._monitor_manager.get_monitor_at_point(x, y)
        return result

    def _translate_point_to_monitors(
        self, x: int, y: int, target_monitors: list[int]
    ) -> tuple[int, int]:
        """Translate a point from its original monitor to target monitors.

        Detects which monitor the point is on, calculates relative offset,
        and applies that offset to the first target monitor.

        Args:
            x: Original X coordinate (absolute)
            y: Original Y coordinate (absolute)
            target_monitors: List of target monitor indices

        Returns:
            Translated (x, y) coordinates on the target monitor
        """
        if not target_monitors:
            return (x, y)

        # Find original monitor
        original_monitor_idx = self._get_monitor_for_point(x, y)
        if original_monitor_idx is None:
            logger.warning(f"Could not find monitor for point ({x}, {y})")
            return (x, y)

        original_monitor = self._monitor_manager.get_monitor_info(original_monitor_idx)
        if not original_monitor:
            return (x, y)

        # Calculate relative offset within original monitor
        rel_x = x - original_monitor.x
        rel_y = y - original_monitor.y

        # Get target monitor (use first one)
        target_monitor_idx = target_monitors[0]
        target_monitor = self._monitor_manager.get_monitor_info(target_monitor_idx)
        if not target_monitor:
            logger.warning(f"Target monitor {target_monitor_idx} not found")
            return (x, y)

        # Apply relative offset to target monitor
        new_x = target_monitor.x + rel_x
        new_y = target_monitor.y + rel_y

        logger.debug(
            f"Translated point ({x}, {y}) from monitor {original_monitor_idx} "
            f"to ({new_x}, {new_y}) on monitor {target_monitor_idx}"
        )

        return (new_x, new_y)

    def _translate_region_to_monitors(self, region: Region, target_monitors: list[int]) -> Region:
        """Translate a region from its original monitor to target monitors.

        Args:
            region: Original region (absolute coordinates)
            target_monitors: List of target monitor indices

        Returns:
            Translated region on the target monitor
        """
        if not target_monitors:
            return region

        new_x, new_y = self._translate_point_to_monitors(region.x, region.y, target_monitors)

        return Region(
            x=new_x,
            y=new_y,
            width=region.width,
            height=region.height,
        )

    def load_config(self, config: QontinuiConfig, state_service: StateService) -> bool:
        """Load a QontinuiConfig and populate the StateService.

        Args:
            config: QontinuiConfig from qontinui-schemas
            state_service: StateService to populate with states

        Returns:
            True if all states loaded successfully, False if any errors occurred
        """
        try:
            # Phase 1: Load all images into registry
            self._load_images(config.images)

            # Phase 2: Load all states
            success = self._load_states(config.states, state_service)

            # Phase 3: Load transitions (TODO: implement when needed)
            # self._load_transitions(config.transitions, state_service)

            return success

        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            return False

    def _load_images(self, images: list[ImageAsset]) -> None:
        """Load all ImageAssets into the registry.

        Args:
            images: List of ImageAsset objects from config
        """
        logger.info(f"Loading {len(images)} images from config")

        for image_asset in images:
            try:
                # Decode base64 image data
                image_data = base64.b64decode(image_asset.data)
                pil_image = PILImage.open(BytesIO(image_data))

                # Convert to qontinui Image
                image = Image.from_pil(pil_image, name=image_asset.name)

                # Register in global registry
                registry.register_image(
                    image_asset.id,
                    image,
                    name=image_asset.name,
                )

                # Keep local lookup for state loading
                self._image_lookup[image_asset.id] = image

                logger.debug(
                    f"Loaded image '{image_asset.name}' (id={image_asset.id}, "
                    f"{image_asset.width}x{image_asset.height})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to load image '{image_asset.id}': {e}",
                    exc_info=True,
                )

        logger.info(f"Successfully loaded {len(self._image_lookup)} images")

    def _load_states(self, states: list[ConfigState], state_service: StateService) -> bool:
        """Load all states from config.

        Args:
            states: List of State objects from config
            state_service: StateService to populate

        Returns:
            True if all states loaded successfully
        """
        logger.info(f"Loading {len(states)} states from config")

        success_count = 0
        error_count = 0

        for config_state in states:
            try:
                if self._load_single_state(config_state, state_service):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to load state '{config_state.id}': {e}",
                    exc_info=True,
                )
                error_count += 1

        logger.info(f"State loading complete: {success_count} succeeded, {error_count} failed")

        return error_count == 0

    def _load_single_state(self, config_state: ConfigState, state_service: StateService) -> bool:
        """Load a single state from config.

        Args:
            config_state: State from config
            state_service: StateService for ID generation and storage

        Returns:
            True if state loaded successfully
        """
        # Generate integer ID for this state
        int_id = state_service.generate_id_for_string_id(config_state.id)
        self._state_id_mapping[config_state.id] = int_id

        # Create State using StateBuilder
        builder = StateBuilder(config_state.name)

        if config_state.description:
            builder.with_description(config_state.description)

        # Process stateImages (v2.x format)
        image_count = 0
        for config_state_image in config_state.state_images:
            runtime_state_image = self._convert_state_image(config_state_image)
            if runtime_state_image:
                builder.with_images(runtime_state_image)
                image_count += 1

        # Process stateRegions if present
        if hasattr(config_state, "state_regions") and config_state.state_regions:
            for config_region in config_state.state_regions:
                runtime_region = self._convert_state_region(config_region)
                if runtime_region:
                    builder.with_regions(runtime_region)

        # Process stateLocations if present
        if hasattr(config_state, "state_locations") and config_state.state_locations:
            for config_location in config_state.state_locations:
                runtime_location = self._convert_state_location(config_location)
                if runtime_location:
                    builder.with_locations(runtime_location)

        # Process stateStrings if present
        if hasattr(config_state, "state_strings") and config_state.state_strings:
            for config_string in config_state.state_strings:
                runtime_string = self._convert_state_string(config_string)
                if runtime_string:
                    builder.with_strings(runtime_string)

        # Set is_initial flag
        if config_state.is_initial:
            builder.set_is_initial(True)

        # Set blocking flag if present
        if hasattr(config_state, "blocking") and config_state.blocking:
            builder.set_blocking(True)

        # Set canHide states
        if hasattr(config_state, "can_hide") and config_state.can_hide:
            builder.can_hide_states(*config_state.can_hide)

        # Build the State object
        state = builder.build()
        state.id = int_id

        # Add state to service
        state_service.add_state(state)

        logger.debug(
            f"Loaded state '{config_state.name}': id={int_id}, images={image_count}, "
            f"initial={config_state.is_initial}"
        )

        return True

    def _convert_state_image(
        self, config_state_image: ConfigStateImage
    ) -> RuntimeStateImage | None:
        """Convert a config StateImage to a runtime StateImage.

        This handles the key conversion from the config format (patterns with
        image ID references, monitors list) to the runtime format (actual Image
        object with search configuration).

        Args:
            config_state_image: StateImage from config

        Returns:
            Runtime StateImage or None if conversion fails
        """
        # Get the first pattern's image (primary image)
        if not config_state_image.patterns:
            logger.warning(f"StateImage '{config_state_image.id}' has no patterns, skipping")
            return None

        primary_pattern = config_state_image.patterns[0]
        image_id = primary_pattern.image_id

        if not image_id:
            logger.warning(f"StateImage '{config_state_image.id}' pattern has no imageId, skipping")
            return None

        # Look up the image
        image = self._image_lookup.get(image_id)
        if not image:
            logger.error(f"StateImage '{config_state_image.id}': image '{image_id}' not found")
            return None

        # Create the runtime StateImage
        runtime_state_image = RuntimeStateImage(
            image=image,
            name=config_state_image.name or config_state_image.id,
        )

        # Apply similarity threshold from pattern
        if primary_pattern.similarity is not None:
            runtime_state_image.set_similarity(primary_pattern.similarity)

        # Apply shared flag
        if config_state_image.shared:
            runtime_state_image.set_shared(True)

        # Handle monitors and search regions together
        # Priority: When monitors are configured, translate pattern searchRegions to target monitors
        target_monitors = config_state_image.monitors or []

        # Get pattern's search_regions (plural - list of SearchRegion)
        pattern_search_regions = (
            primary_pattern.search_regions if hasattr(primary_pattern, "search_regions") else []
        )
        has_pattern_regions = bool(pattern_search_regions and len(pattern_search_regions) > 0)

        # Log what we're processing
        logger.info(
            f"StateImage '{config_state_image.id}': monitors={target_monitors}, "
            f"has_pattern_regions={has_pattern_regions}, "
            f"pattern_regions_count={len(pattern_search_regions) if pattern_search_regions else 0}"
        )

        if target_monitors and has_pattern_regions:
            # Translate pattern's searchRegions to target monitors
            translated_search_regions = SearchRegions()
            for config_region in pattern_search_regions:
                original_region = self._convert_search_region(config_region)
                if original_region:
                    translated_region = self._translate_region_to_monitors(
                        original_region, target_monitors
                    )
                    translated_search_regions.add_region(translated_region)
                    logger.debug(
                        f"StateImage '{config_state_image.id}': translated searchRegion "
                        f"from ({original_region.x}, {original_region.y}) to "
                        f"({translated_region.x}, {translated_region.y}) for monitors {target_monitors}"
                    )
            if not translated_search_regions.is_empty():
                runtime_state_image.set_search_regions(translated_search_regions)
        elif target_monitors:
            # Monitors configured but no pattern searchRegions - use full monitor bounds
            logger.info(
                f"StateImage '{config_state_image.id}': using full monitor bounds for {target_monitors}"
            )
            monitor_search_regions = self._create_search_regions_for_monitors(target_monitors)
            if monitor_search_regions:
                runtime_state_image.set_search_regions(monitor_search_regions)
                # Log the actual regions being used
                for region in monitor_search_regions.regions:
                    logger.info(
                        f"  -> Search region: x={region.x}, y={region.y}, "
                        f"w={region.width}, h={region.height}"
                    )
        elif has_pattern_regions:
            # No monitors configured - use pattern searchRegions as-is (absolute coordinates)
            pattern_regions = SearchRegions()
            for config_region in pattern_search_regions:
                converted_region = self._convert_search_region(config_region)
                if converted_region:
                    pattern_regions.add_region(converted_region)
            if not pattern_regions.is_empty():
                runtime_state_image.set_search_regions(pattern_regions)

        # Apply RAG search configuration
        if config_state_image.search_mode:
            runtime_state_image.set_search_mode(config_state_image.search_mode.value)

        if config_state_image.rag_multi_pattern_mode:
            runtime_state_image.set_rag_multi_pattern_mode(
                config_state_image.rag_multi_pattern_mode.value
            )

        return runtime_state_image

    def _convert_search_region(self, search_region: SearchRegion) -> Region | None:
        """Convert a config SearchRegion to a runtime Region.

        Args:
            search_region: SearchRegion from config

        Returns:
            Runtime Region or None if conversion fails
        """
        try:
            return Region(
                x=search_region.x,
                y=search_region.y,
                width=search_region.width,
                height=search_region.height,
            )
        except Exception as e:
            logger.warning(f"Failed to convert search region: {e}")
            return None

    def _create_search_regions_for_monitors(
        self, monitor_indices: list[int]
    ) -> SearchRegions | None:
        """Create SearchRegions for specific monitors.

        This configures the search to only look on specific monitors,
        which is important for multi-monitor setups.

        Args:
            monitor_indices: List of monitor indices (0-based)

        Returns:
            SearchRegions configured for the specified monitors, or None
        """
        if not monitor_indices:
            return None

        try:
            from qontinui.monitor.monitor_manager import MonitorManager

            # Get monitor info to create regions
            monitor_manager = MonitorManager()
            search_regions = SearchRegions()

            for monitor_idx in monitor_indices:
                monitor_info = monitor_manager.get_monitor_info(monitor_idx)
                if monitor_info:
                    # Create a region from monitor bounds
                    region = Region(
                        x=monitor_info.x,
                        y=monitor_info.y,
                        width=monitor_info.width,
                        height=monitor_info.height,
                    )
                    search_regions.add_region(region)
                    logger.debug(
                        f"Added monitor {monitor_idx} as search region: "
                        f"{monitor_info.width}x{monitor_info.height} at "
                        f"({monitor_info.x}, {monitor_info.y})"
                    )
                else:
                    logger.warning(f"Monitor {monitor_idx} not found, skipping")

            if search_regions.is_empty():
                return None

            return search_regions

        except Exception as e:
            logger.warning(f"Failed to create search regions for monitors: {e}")
            return None

    def _convert_state_region(self, config_region: ConfigStateRegion) -> StateRegion | None:
        """Convert a config StateRegion to a runtime StateRegion.

        Translates region coordinates to the configured monitors.
        Regions require specific monitors (not "all monitors").

        Args:
            config_region: StateRegion from config

        Returns:
            Runtime StateRegion or None if conversion fails
        """
        try:
            # Get monitors from config, default to primary monitor
            monitors = [0]  # Default to primary monitor
            if hasattr(config_region, "monitors") and config_region.monitors:
                # Filter out -1 (all monitors) as regions require specific monitors
                monitors = [m for m in config_region.monitors if m >= 0]
                if not monitors:
                    monitors = [0]  # Fall back to primary if only -1 was specified
                    logger.warning(
                        f"StateRegion '{config_region.id}' had 'all monitors' (-1), "
                        f"defaulting to primary monitor. Regions require specific monitors."
                    )

            # Create region with original coordinates
            original_region = Region(
                x=config_region.x,
                y=config_region.y,
                width=config_region.width,
                height=config_region.height,
            )

            # Translate coordinates to target monitors
            translated_region = self._translate_region_to_monitors(original_region, monitors)

            if translated_region.x != original_region.x or translated_region.y != original_region.y:
                logger.debug(
                    f"StateRegion '{config_region.id}': translated from "
                    f"({original_region.x}, {original_region.y}) to "
                    f"({translated_region.x}, {translated_region.y}) for monitors {monitors}"
                )

            return StateRegion(
                region=translated_region,
                name=config_region.name or config_region.id,
                monitors=monitors,
            )
        except Exception as e:
            logger.warning(f"Failed to convert state region: {e}")
            return None

    def _convert_state_location(self, config_location: ConfigStateLocation) -> StateLocation | None:
        """Convert a config StateLocation to a runtime StateLocation.

        Translates location coordinates to the configured monitors.
        Locations require specific monitors (not "all monitors").

        Args:
            config_location: StateLocation from config

        Returns:
            Runtime StateLocation or None if conversion fails
        """
        try:
            from qontinui.model.element import Location

            # Get monitors from config, default to primary monitor
            monitors = [0]  # Default to primary monitor
            if hasattr(config_location, "monitors") and config_location.monitors:
                # Filter out -1 (all monitors) as locations require specific monitors
                monitors = [m for m in config_location.monitors if m >= 0]
                if not monitors:
                    monitors = [0]  # Fall back to primary if only -1 was specified
                    logger.warning(
                        f"StateLocation '{config_location.id}' had 'all monitors' (-1), "
                        f"defaulting to primary monitor. Locations require specific monitors."
                    )

            # Translate coordinates to target monitors
            original_x = config_location.x
            original_y = config_location.y
            translated_x, translated_y = self._translate_point_to_monitors(
                original_x, original_y, monitors
            )

            if translated_x != original_x or translated_y != original_y:
                logger.debug(
                    f"StateLocation '{config_location.id}': translated from "
                    f"({original_x}, {original_y}) to ({translated_x}, {translated_y}) "
                    f"for monitors {monitors}"
                )

            # Create Location object with translated coordinates
            location = Location(
                x=translated_x,
                y=translated_y,
            )

            return StateLocation(
                location=location,
                name=config_location.name or config_location.id,
                monitors=monitors,
            )
        except Exception as e:
            logger.warning(f"Failed to convert state location: {e}")
            return None

    def _convert_state_string(self, config_string: ConfigStateString) -> StateString | None:
        """Convert a config StateString to a runtime StateString.

        Args:
            config_string: StateString from config

        Returns:
            Runtime StateString or None if conversion fails
        """
        try:
            # Get the string value from config
            string_value = ""
            if hasattr(config_string, "value") and config_string.value:
                string_value = config_string.value

            return StateString(
                string=string_value,
                name=config_string.name or config_string.id,
            )
        except Exception as e:
            logger.warning(f"Failed to convert state string: {e}")
            return None


def load_config(config: QontinuiConfig, state_service: StateService) -> bool:
    """Load a QontinuiConfig and populate the StateService.

    Convenience function for loading configuration.

    Args:
        config: QontinuiConfig from qontinui-schemas
        state_service: StateService to populate with states

    Returns:
        True if all states loaded successfully, False if any errors occurred
    """
    adapter = ConfigAdapter()
    return adapter.load_config(config, state_service)


def load_config_from_dict(config_dict: dict[str, Any], state_service: StateService) -> bool:
    """Load a config dictionary and populate the StateService.

    This function accepts a raw dictionary (e.g., from JSON.loads) and
    validates it against qontinui-schemas before loading.

    Args:
        config_dict: Raw configuration dictionary
        state_service: StateService to populate with states

    Returns:
        True if all states loaded successfully, False if any errors occurred
    """
    try:
        from qontinui_schemas import QontinuiConfig

        config = QontinuiConfig.model_validate(config_dict)
        return load_config(config, state_service)

    except ImportError:
        logger.error("qontinui-schemas not installed. Install with: pip install qontinui-schemas")
        return False

    except Exception as e:
        logger.error(f"Failed to validate config: {e}", exc_info=True)
        return False
