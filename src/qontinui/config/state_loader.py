"""State Loader - loads states from configuration into StateService.

This module implements Phase 2 of the State/Transition Loading Implementation Plan.
It parses state definitions from JSON configuration and creates State objects that
are populated into the StateService.

The loader handles:
- Creating State objects from config definitions
- Looking up Image objects from the registry
- Creating StateImage objects and linking them to states
- Setting state properties (name, description, isInitial)
- Managing ID mapping between string IDs (from config) and integer IDs (library)
- Error handling for missing images or malformed config
"""

import logging
from typing import Any

from qontinui import registry
from qontinui.model.state.state import StateBuilder
from qontinui.model.state.state_image import StateImage
from qontinui.model.state.state_service import StateService

logger = logging.getLogger(__name__)


def load_states_from_config(config: dict[str, Any], state_service: StateService) -> bool:
    """Load all states from configuration and populate StateService.

    This function is the main entry point for Phase 2 of the state loading process.
    It extracts state definitions from the config JSON, creates State objects with
    their identifying images, and adds them to the StateService.

    The function uses the registry to look up Image objects that were previously
    loaded by the runner. For each state, it:
    1. Generates an integer ID from the string ID
    2. Creates a State object using StateBuilder
    3. Creates StateImage objects for each identifying image
    4. Sets state properties (isInitial flag)
    5. Adds the state to the StateService

    Args:
        config: Full configuration dictionary containing "states" array
        state_service: StateService to populate with loaded states

    Returns:
        True if all states loaded successfully, False if any errors occurred

    Example:
        >>> config = {
        ...     "states": [
        ...         {
        ...             "id": "state-start",
        ...             "name": "Start State",
        ...             "description": "Initial state",
        ...             "identifyingImages": ["image-id-1", "image-id-2"],
        ...             "position": {"x": 100, "y": 100},
        ...             "isInitial": true
        ...         }
        ...     ]
        ... }
        >>> state_service = StateService()
        >>> success = load_states_from_config(config, state_service)
    """
    if "states" not in config:
        logger.warning("No 'states' key found in configuration")
        return True  # Not an error - config may have no states

    states = config["states"]
    if not isinstance(states, list):
        logger.error("Configuration 'states' must be a list")
        return False

    logger.info(f"Loading {len(states)} states from configuration")

    success_count = 0
    error_count = 0

    for state_def in states:
        try:
            if _load_single_state(state_def, state_service):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error loading state: {e}", exc_info=True)
            error_count += 1

    logger.info(f"State loading complete: {success_count} succeeded, {error_count} failed")

    # Return True only if all states loaded successfully
    return error_count == 0


def _load_single_state(state_def: dict[str, Any], state_service: StateService) -> bool:
    """Load a single state from its definition.

    This helper function handles the loading of one state from the config.
    It performs validation, creates the State object, and populates it with
    StateImage objects.

    Args:
        state_def: State definition from config
        state_service: StateService for ID generation and storage

    Returns:
        True if state loaded successfully, False otherwise
    """
    state_id = state_def.get("id", "<unknown>")

    # Validate required fields
    if not _validate_state_definition(state_def, state_id):
        return False

    # Generate integer ID for this state
    int_id = state_service.generate_id_for_string_id(state_id)

    # Extract name and description
    name = state_def.get("name", state_id)
    description = state_def.get("description", "")

    # Create State using StateBuilder
    builder = StateBuilder(name).with_description(description)

    # Process images - support both legacy "identifyingImages" and new "stateImages" format
    image_count = 0

    # Try new stateImages format first (v2.x)
    state_images_list = state_def.get("stateImages", [])
    if state_images_list and isinstance(state_images_list, list):
        logger.debug(f"State '{state_id}': processing {len(state_images_list)} stateImages")
        for si in state_images_list:
            if not isinstance(si, dict):
                logger.warning(f"State '{state_id}': skipping non-dict stateImage: {si}")
                continue

            si_id = si.get("id")
            si_name = si.get("name", si_id)
            patterns = si.get("patterns", [])

            if not patterns:
                logger.warning(f"State '{state_id}': stateImage '{si_id}' has no patterns")
                continue

            # Use first pattern's image as the primary image for this StateImage
            for pattern in patterns:
                pattern_image_id = pattern.get("imageId") or pattern.get("image")
                if not pattern_image_id:
                    continue

                # Look up image in registry
                image = registry.get_image(pattern_image_id)
                if image is None and si_id is not None:
                    # Try the state image ID itself (might be registered under that)
                    image = registry.get_image(str(si_id))

                if image is None:
                    logger.debug(
                        f"State '{state_id}': image '{pattern_image_id}' not found in registry"
                    )
                    continue

                # Create StateImage and add to builder
                state_image = StateImage(image=image, name=si_name or pattern_image_id)
                builder.with_images(state_image)
                image_count += 1
                logger.debug(
                    f"State '{state_id}': added stateImage '{si_name}' from pattern image '{pattern_image_id}'"
                )
                break  # Only use first valid pattern image for this StateImage

    # Fall back to legacy identifyingImages format
    else:
        identifying_images = state_def.get("identifyingImages", [])
        if not isinstance(identifying_images, list):
            logger.error(
                f"State '{state_id}': identifyingImages must be a list, got {type(identifying_images)}"
            )
            return False

        for image_id in identifying_images:
            if not isinstance(image_id, str):
                logger.warning(f"State '{state_id}': skipping non-string image ID: {image_id}")
                continue

            # Look up image in registry
            image = registry.get_image(image_id)
            if image is None:
                logger.error(f"State '{state_id}': image '{image_id}' not found in registry")
                # Continue loading other images instead of failing completely
                continue

            # Create StateImage and add to builder
            state_image = StateImage(image=image, name=image_id)
            builder.with_images(state_image)
            image_count += 1

    if image_count == 0:
        state_images_count = len(state_def.get("stateImages", []))
        identifying_images_count = len(state_def.get("identifyingImages", []))
        if state_images_count > 0 or identifying_images_count > 0:
            logger.warning(
                f"State '{state_id}': no valid images found "
                f"(stateImages={state_images_count}, identifyingImages={identifying_images_count})"
            )

    # Build the State object
    state = builder.build()

    # Set the integer ID
    state.id = int_id

    # Store isInitial flag as state property
    # This boolean flag from the config indicates the starting state
    is_initial = state_def.get("isInitial", False)

    if is_initial:
        # Mark as initial state (used by navigation to determine starting points)
        state.is_initial = True
        logger.debug(f"State '{name}' marked as initial state")

    # Optional: parse position (currently not used by State class)
    position = state_def.get("position")
    if position and isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
        logger.debug(f"State '{name}': position defined at ({x}, {y}) (not stored)")

    # Add state to service
    state_service.add_state(state)

    logger.debug(
        f"Loaded state '{name}': id={int_id}, images={image_count}, " f"initial={is_initial}"
    )

    return True


def _validate_state_definition(state_def: dict[str, Any], state_id: str) -> bool:
    """Validate that a state definition has all required fields.

    Checks for the presence of required fields and validates their types.
    Logs detailed error messages for any validation failures.

    Args:
        state_def: State definition to validate
        state_id: ID of state (for error messages)

    Returns:
        True if valid, False otherwise
    """
    # 'id' is required for mapping
    if "id" not in state_def:
        logger.error("State definition missing required field 'id'")
        return False

    # 'name' is optional but recommended (defaults to id if not present)
    if "name" not in state_def:
        logger.debug(f"State '{state_id}': no 'name' field, using id as name")

    # Validate identifyingImages if present
    if "identifyingImages" in state_def:
        if not isinstance(state_def["identifyingImages"], list):
            logger.error(
                f"State '{state_id}': identifyingImages must be a list, "
                f"got {type(state_def['identifyingImages'])}"
            )
            return False

    return True


# Additional helper functions for diagnostics and debugging


def get_state_statistics(state_service: StateService) -> dict[str, Any]:
    """Get statistics about loaded states.

    Provides diagnostic information about the states that have been loaded,
    including counts, image distribution, and initial/final state detection.

    Args:
        state_service: StateService to analyze

    Returns:
        Dictionary with state statistics
    """
    total_states = len(state_service.get_all_states())
    states_with_images = 0
    total_images = 0
    initial_states = []
    max_images_per_state = 0

    for state in state_service.get_all_states():
        image_count = len(state.state_images)
        total_images += image_count

        if image_count > 0:
            states_with_images += 1
            max_images_per_state = max(max_images_per_state, image_count)

        # Detect initial states by probability
        if state.probability_exists == 100:
            initial_states.append(state.name)

    return {
        "total_states": total_states,
        "states_with_images": states_with_images,
        "states_without_images": total_states - states_with_images,
        "total_images": total_images,
        "avg_images_per_state": (total_images / total_states if total_states > 0 else 0),
        "max_images_per_state": max_images_per_state,
        "initial_states": initial_states,
    }


def validate_state_images(state_service: StateService) -> list[str]:
    """Validate that all state images are properly configured.

    Checks for common issues with state images such as:
    - States with no identifying images
    - State images with missing or invalid Image objects
    - Duplicate image names within a state

    Args:
        state_service: StateService to validate

    Returns:
        List of warning/error messages (empty if no issues found)
    """
    issues = []

    for state in state_service.get_all_states():
        # Check for states with no images
        if len(state.state_images) == 0:
            issues.append(f"State '{state.name}' has no identifying images")

        # Check for invalid images
        for state_image in state.state_images:
            if state_image.image is None:
                issues.append(
                    f"State '{state.name}': StateImage '{state_image.name}' has null image"
                )
            elif hasattr(state_image.image, "is_empty") and state_image.image.is_empty():
                issues.append(
                    f"State '{state.name}': StateImage '{state_image.name}' has empty image"
                )

        # Check for duplicate image names
        image_names = [img.name for img in state.state_images if img.name]
        if len(image_names) != len(set(image_names)):
            duplicates = [name for name in image_names if image_names.count(name) > 1]
            issues.append(f"State '{state.name}': duplicate image names: {set(duplicates)}")

    return issues


def get_state_summary(state_service: StateService) -> str:
    """Get a human-readable summary of loaded states.

    Creates a formatted summary string showing key information about
    all loaded states, useful for logging and debugging.

    Args:
        state_service: StateService to summarize

    Returns:
        Multi-line string with state summary
    """
    states = state_service.get_all_states()
    if not states:
        return "No states loaded"

    lines = [f"Loaded {len(states)} states:"]

    for state in sorted(states, key=lambda s: s.id or 0):
        string_id = state_service.get_string_id(state.id) if state.id else "N/A"
        image_count = len(state.state_images)
        is_initial = state.probability_exists == 100

        status = []
        if is_initial:
            status.append("INITIAL")

        status_str = f" [{', '.join(status)}]" if status else ""

        lines.append(
            f"  - {state.name} (id={state.id}, string_id={string_id}, "
            f"images={image_count}){status_str}"
        )

    return "\n".join(lines)
