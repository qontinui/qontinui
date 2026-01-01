"""
Event Analyzer for React.

Handles conditional rendering and event handler extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.static.models import (
    ConditionalRender,
    EventHandler,
    StateVariable,
)

from . import handlers as handler_module
from . import jsx as jsx_module

logger = logging.getLogger(__name__)


class EventAnalyzer:
    """Analyzer for extracting conditional rendering and event handlers."""

    def __init__(self) -> None:
        """Initialize the event analyzer."""
        self.conditional_renders: list[ConditionalRender] = []
        self.event_handlers: list[EventHandler] = []

    def extract_conditionals_for_component(
        self, component_parse: dict, component_name: str, file_path: Path
    ) -> list[ConditionalRender]:
        """
        Extract all conditional rendering patterns for a component.

        Args:
            component_parse: Parse result for the component
            component_name: Component name
            file_path: Source file path

        Returns:
            List of ConditionalRender objects
        """
        conditionals: list[ConditionalRender] = []

        # Extract different conditional patterns
        conditionals.extend(
            jsx_module.extract_logical_and(component_parse, component_name, file_path)
        )
        conditionals.extend(jsx_module.extract_ternary(component_parse, component_name, file_path))
        conditionals.extend(
            jsx_module.extract_early_returns(component_parse, component_name, file_path)
        )
        conditionals.extend(
            jsx_module.extract_switch_render(component_parse, component_name, file_path)
        )

        self.conditional_renders.extend(conditionals)
        return conditionals

    def extract_event_handlers_for_component(
        self,
        component_parse: dict,
        component_name: str,
        file_path: Path,
        state_vars: list[StateVariable],
    ) -> list[EventHandler]:
        """
        Extract event handlers for a component.

        Args:
            component_parse: Parse result for the component
            component_name: Component name
            file_path: Source file path
            state_vars: State variables in the component

        Returns:
            List of EventHandler objects
        """
        handlers = handler_module.extract_event_handlers(
            component_parse, component_name, file_path, state_vars
        )
        self.event_handlers.extend(handlers)
        return handlers

    def get_conditional_renders(self) -> list[ConditionalRender]:
        """Get all extracted conditional renders."""
        return self.conditional_renders

    def get_event_handlers(self) -> list[EventHandler]:
        """Get all extracted event handlers."""
        return self.event_handlers

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.conditional_renders = []
        self.event_handlers = []
