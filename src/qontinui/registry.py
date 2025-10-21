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
from typing import Any

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model.element import Image

# Global registries - private to this module
_image_registry: dict[str, "Image"] = {}
_workflow_registry: dict[str, Any] = {}


def register_image(image_id: str, image: "Image") -> None:
    """Register an image for use by the library.

    This should be called by the runner after loading images from configuration.
    If an image with the same ID already exists, it will be replaced and a warning
    will be logged.

    Args:
        image_id: Unique identifier for the image (typically from config)
        image: Image object to register

    Example:
        >>> from qontinui.model.element import Image
        >>> from qontinui import registry
        >>> img = Image.from_file("button.png")
        >>> registry.register_image("submit_button", img)
    """
    if image_id in _image_registry:
        logger.warning(
            f"Image '{image_id}' already registered - replacing with new image"
        )
    _image_registry[image_id] = image
    logger.debug(f"Registered image: {image_id}")


def register_workflow(workflow_id: str, workflow: Any) -> None:
    """Register a workflow for use by the library.

    This should be called by the runner after loading workflows from configuration.
    If a workflow with the same ID already exists, it will be replaced and a warning
    will be logged.

    Args:
        workflow_id: Unique identifier for the workflow (typically from config)
        workflow: Workflow object to register (type depends on workflow implementation)

    Example:
        >>> from qontinui import registry
        >>> workflow = load_workflow_from_config(workflow_dict)
        >>> registry.register_workflow("login_flow", workflow)
    """
    if workflow_id in _workflow_registry:
        logger.warning(
            f"Workflow '{workflow_id}' already registered - replacing with new workflow"
        )
    _workflow_registry[workflow_id] = workflow
    logger.debug(f"Registered workflow: {workflow_id}")


def get_image(image_id: str) -> "Image | None":
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


def clear_images() -> None:
    """Clear all registered images.

    This is primarily useful for testing to ensure clean state between tests.
    Should NOT be called during normal operation.

    Example:
        >>> from qontinui import registry
        >>> # In test teardown:
        >>> registry.clear_images()
    """
    _image_registry.clear()
    logger.debug("Cleared all registered images")


def clear_workflows() -> None:
    """Clear all registered workflows.

    This is primarily useful for testing to ensure clean state between tests.
    Should NOT be called during normal operation.

    Example:
        >>> from qontinui import registry
        >>> # In test teardown:
        >>> registry.clear_workflows()
    """
    _workflow_registry.clear()
    logger.debug("Cleared all registered workflows")


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
