"""Shared registry module for images and workflows.

This module provides global registries that enable sharing of images and workflows
between the qontinui runner and library components. This is a critical piece of
infrastructure for the config-based workflow system.

WHY THIS EXISTS:
================
The qontinui library needs access to images and workflows that are loaded from
configuration files by the runner. Rather than passing these objects through every
function call, we use a centralized registry pattern that allows:

1. The runner to register images/workflows after loading config
2. The library to retrieve them when executing actions
3. Decoupling between runner and library (no direct dependencies)
4. Clean separation of concerns (runner handles loading, library handles execution)

WHEN TO USE THIS:
=================
- During config loading: Runner calls register_image/register_workflow
- During action execution: Library calls get_image/get_workflow
- During testing: Test setup calls register_* and teardown calls clear_*
- NEVER use this for runtime state - only for configuration objects

ARCHITECTURE NOTE:
==================
This follows the same pattern as navigation_api.py - a simple module-level
registry that provides a clean interface between runner and library without
requiring complex dependency injection or singleton patterns.

THREAD SAFETY:
==============
This module is NOT thread-safe. It assumes:
1. Configuration is loaded once at startup (single-threaded)
2. Execution happens after loading is complete
3. No concurrent registration/retrieval operations

If thread safety is needed in the future, add threading.Lock around
the registry dictionaries.

Example Usage:
==============
# In runner (after loading config):
from qontinui import registry

image = Image.from_file("button.png")
registry.register_image("submit_button", image)

workflow = load_workflow_from_config(workflow_dict)
registry.register_workflow("login_flow", workflow)

# In library (during action execution):
from qontinui import registry

button_image = registry.get_image("submit_button")
if button_image:
    action.find(button_image)

workflow = registry.get_workflow("login_flow")
if workflow:
    execute_workflow(workflow)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model.element import Image

logger = logging.getLogger(__name__)

# Global registries - private to this module
_image_registry: dict[str, Image] = {}
_image_metadata_registry: dict[str, dict[str, str]] = (
    {}
)  # Stores {image_id: {file_path: str, name: str}}
_workflow_registry: dict[str, Any] = {}
_workflow_names: dict[str, str] = {}  # Stores {workflow_id: workflow_name}
_workflow_definitions: dict[str, dict[str, Any]] = (
    {}
)  # Stores full workflow definitions for graph workflows


def register_image(
    image_id: str, image: Image, file_path: str | None = None, name: str | None = None
) -> None:
    """Register an image for use by the library.

    This should be called by the runner after loading images from configuration.
    If an image with the same ID already exists, it will be replaced and a warning
    will be logged.

    Args:
        image_id: Unique identifier for the image (typically from config)
        image: Image object to register
        file_path: Optional file path where the image was loaded from
        name: Optional name for the image

    Example:
        >>> from qontinui.model.element import Image
        >>> from qontinui import registry
        >>> img = Image.from_file("button.png")
        >>> registry.register_image("submit_button", img, file_path="button.png", name="Submit Button")
    """
    if image_id in _image_registry:
        logger.warning(f"Image '{image_id}' already registered - replacing with new image")
    _image_registry[image_id] = image

    # Store metadata if provided
    if file_path is not None or name is not None:
        _image_metadata_registry[image_id] = {
            "file_path": file_path or "",
            "name": name or image.name or image_id,
        }

    logger.debug(f"Registered image: {image_id}")


def register_workflow(workflow_id: str, workflow: Any, workflow_name: str | None = None) -> None:
    """Register a workflow for use by the library.

    This should be called by the runner after loading workflows from configuration.
    If a workflow with the same ID already exists, it will be replaced and a warning
    will be logged.

    Args:
        workflow_id: Unique identifier for the workflow (typically from config)
        workflow: Workflow object to register (type depends on workflow implementation)
        workflow_name: Optional human-readable name for logging purposes

    Example:
        >>> from qontinui import registry
        >>> workflow = load_workflow_from_config(workflow_dict)
        >>> registry.register_workflow("login_flow", workflow, "Login Flow")
    """
    display_name = workflow_name or workflow_id
    if workflow_id in _workflow_registry:
        logger.warning(
            f"Workflow '{display_name}' already registered - replacing with new workflow"
        )
    _workflow_registry[workflow_id] = workflow

    # Store workflow name if provided
    if workflow_name:
        _workflow_names[workflow_id] = workflow_name

    logger.debug(f"Registered workflow: {display_name}")


def get_image(image_id: str) -> Image | None:
    """Retrieve a registered image by ID.

    This should be called by the library when it needs to access an image
    that was loaded from configuration.

    Args:
        image_id: Unique identifier of the image to retrieve

    Returns:
        Image object if found, None otherwise

    Example:
        >>> from qontinui import registry
        >>> button = registry.get_image("submit_button")
        >>> if button:
        ...     action.find(button)
    """
    image = _image_registry.get(image_id)
    if image is None:
        logger.warning(f"Image '{image_id}' not found in registry")
    return image


def get_image_metadata(image_id: str) -> dict[str, str] | None:
    """Retrieve metadata for a registered image by ID.

    Args:
        image_id: Unique identifier of the image

    Returns:
        Dictionary with 'file_path' and 'name' keys if found, None otherwise

    Example:
        >>> from qontinui import registry
        >>> metadata = registry.get_image_metadata("submit_button")
        >>> if metadata:
        ...     print(f"Image path: {metadata['file_path']}")
    """
    return _image_metadata_registry.get(image_id)


def get_workflow(workflow_id: str) -> Any | None:
    """Retrieve a registered workflow by ID.

    This should be called by the library when it needs to access a workflow
    that was loaded from configuration.

    Args:
        workflow_id: Unique identifier of the workflow to retrieve

    Returns:
        Workflow object if found, None otherwise

    Example:
        >>> from qontinui import registry
        >>> workflow = registry.get_workflow("login_flow")
        >>> if workflow:
        ...     execute_workflow(workflow)
    """
    workflow = _workflow_registry.get(workflow_id)
    if workflow is None:
        logger.warning(f"Workflow '{workflow_id}' not found in registry")
    return workflow


def get_workflow_name(workflow_id: str) -> str | None:
    """Retrieve the name of a registered workflow by ID.

    Args:
        workflow_id: Unique identifier of the workflow

    Returns:
        Workflow name if found, None otherwise

    Example:
        >>> from qontinui import registry
        >>> name = registry.get_workflow_name("login_flow")
        >>> print(f"Workflow name: {name}")
    """
    return _workflow_names.get(workflow_id)


def clear_images() -> None:
    """Clear all registered images and their metadata.

    This is primarily useful for testing to ensure clean state between tests.
    Should NOT be called during normal operation.

    Example:
        >>> from qontinui import registry
        >>> # In test teardown:
        >>> registry.clear_images()
    """
    _image_registry.clear()
    _image_metadata_registry.clear()
    logger.debug("Cleared all registered images and metadata")


def register_workflow_definition(workflow_id: str, workflow_def: dict[str, Any]) -> None:
    """Register a full workflow definition (for graph workflows).

    This stores the complete workflow definition including connections, metadata, etc.
    Used for graph-based workflows that need more than just the actions array.

    Args:
        workflow_id: Unique identifier for the workflow
        workflow_def: Full workflow definition dictionary from config

    Example:
        >>> from qontinui import registry
        >>> workflow_def = {
        ...     "id": "my-workflow",
        ...     "format": "graph",
        ...     "actions": [...],
        ...     "connections": {...}
        ... }
        >>> registry.register_workflow_definition("my-workflow", workflow_def)
    """
    if workflow_id in _workflow_definitions:
        logger.warning(f"Workflow definition '{workflow_id}' already registered - replacing")
    _workflow_definitions[workflow_id] = workflow_def
    logger.debug(f"Registered workflow definition: {workflow_id}")


def get_workflow_definition(workflow_id: str) -> dict[str, Any] | None:
    """Retrieve a full workflow definition by ID.

    Args:
        workflow_id: Unique identifier of the workflow definition

    Returns:
        Workflow definition dictionary if found, None otherwise

    Example:
        >>> from qontinui import registry
        >>> workflow_def = registry.get_workflow_definition("my-workflow")
        >>> if workflow_def and workflow_def.get("format") == "graph":
        ...     execute_graph_workflow(workflow_def)
    """
    workflow_def = _workflow_definitions.get(workflow_id)
    if workflow_def is None:
        logger.debug(f"Workflow definition '{workflow_id}' not found in registry")
    return workflow_def


def clear_workflows() -> None:
    """Clear all registered workflows and workflow definitions.

    This is primarily useful for testing to ensure clean state between tests.
    Should NOT be called during normal operation.

    Example:
        >>> from qontinui import registry
        >>> # In test teardown:
        >>> registry.clear_workflows()
    """
    _workflow_registry.clear()
    _workflow_names.clear()
    _workflow_definitions.clear()
    logger.debug("Cleared all registered workflows and definitions")


def clear_all() -> None:
    """Clear both image and workflow registries.

    This is primarily useful for testing to ensure clean state between tests.
    Should NOT be called during normal operation.

    Example:
        >>> from qontinui import registry
        >>> # In test teardown:
        >>> registry.clear_all()
    """
    clear_images()
    clear_workflows()
    logger.debug("Cleared all registries")


def get_all_image_ids() -> list[str]:
    """Get all registered image IDs.

    Returns:
        List of all image IDs currently in the registry

    Example:
        >>> from qontinui import registry
        >>> image_ids = registry.get_all_image_ids()
        >>> print(f"Registered images: {', '.join(image_ids)}")
    """
    return list(_image_registry.keys())


def get_all_workflow_ids() -> list[str]:
    """Get all registered workflow IDs.

    Returns:
        List of all workflow IDs currently in the registry

    Example:
        >>> from qontinui import registry
        >>> workflow_ids = registry.get_all_workflow_ids()
        >>> print(f"Registered workflows: {', '.join(workflow_ids)}")
    """
    return list(_workflow_registry.keys())
