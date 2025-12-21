"""
State Analyzer for React.

Handles state variable extraction and visibility state analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.static.models import (
    ComponentDefinition,
    ConditionalRender,
    EventHandler,
    StateVariable,
    VisibilityState,
)

from . import hooks as hook_module

logger = logging.getLogger(__name__)


class StateAnalyzer:
    """Analyzer for extracting and analyzing React state variables."""

    def __init__(self):
        """Initialize the state analyzer."""
        self.state_variables: list[StateVariable] = []
        self.visibility_states: list[VisibilityState] = []

    def extract_state_for_component(
        self, component_parse: dict, component_name: str, file_path: Path
    ) -> list[StateVariable]:
        """
        Extract all state variables for a component.

        Args:
            component_parse: Parse result for the component
            component_name: Component name
            file_path: Source file path

        Returns:
            List of StateVariable objects
        """
        state_vars: list[StateVariable] = []

        # Extract from different hook types
        state_vars.extend(hook_module.extract_use_state(component_parse, component_name, file_path))
        state_vars.extend(
            hook_module.extract_use_reducer(component_parse, component_name, file_path)
        )
        state_vars.extend(
            hook_module.extract_use_context(component_parse, component_name, file_path)
        )
        state_vars.extend(
            hook_module.extract_custom_hooks(component_parse, component_name, file_path)
        )

        self.state_variables.extend(state_vars)
        return state_vars

    def extract_visibility_states(
        self,
        component: ComponentDefinition,
        state_vars: list[StateVariable],
        conditionals: list[ConditionalRender],
        handlers: list[EventHandler],
    ) -> list[VisibilityState]:
        """
        Extract visibility-based sub-states from a component.

        Analyzes conditional rendering patterns and state variables to detect
        different UI configurations of the same page (e.g., modal open/closed,
        sidebar expanded/collapsed, dropdown visible/hidden).

        Args:
            component: The component to analyze
            state_vars: State variables in this component
            conditionals: Conditional rendering patterns in this component
            handlers: Event handlers in this component

        Returns:
            List of VisibilityState objects representing sub-states
        """
        visibility_states: list[VisibilityState] = []

        # Build a map of state variable names to their IDs for quick lookup
        {var.name: var for var in state_vars}

        # Identify visibility-controlling state variables
        # These typically have names like: isOpen, showModal, menuExpanded, etc.
        visibility_vars = self._identify_visibility_variables(state_vars)

        if not visibility_vars:
            # No visibility-controlling variables found
            return visibility_states

        # For each visibility variable, create sub-states
        for var in visibility_vars:
            # Find conditionals that use this variable
            related_conditionals = [
                cond
                for cond in conditionals
                if var.id in cond.controlling_variables or var.name in cond.condition
            ]

            if not related_conditionals:
                continue

            # Find event handlers that toggle this variable
            toggle_handlers = self._find_toggle_handlers(var, handlers)

            # For boolean visibility variables, create two states: visible and hidden
            if var.initial_value is False or var.initial_value is True or self._is_boolean_var(var):
                # State 1: Variable is False (default/closed/hidden)
                default_state = VisibilityState(
                    id=f"{component.id}:{var.name}_false",
                    name=f"{component.name}_{var.name}_false",
                    parent_component=component.id,
                    parent_route=component.route_path,
                    controlling_variable=var.id,
                    variable_value=False,
                    rendered_components=[],  # Nothing extra rendered
                    hidden_components=self._extract_rendered_components(related_conditionals, True),
                    toggle_handlers=[h.id for h in toggle_handlers],
                    conditional_render_id=(
                        related_conditionals[0].id if related_conditionals else None
                    ),
                    file_path=component.file_path,
                    line_number=var.line_number,
                    metadata={
                        "variable_name": var.name,
                        "is_default": var.initial_value is False,
                    },
                )
                visibility_states.append(default_state)

                # State 2: Variable is True (visible/open/expanded)
                visible_state = VisibilityState(
                    id=f"{component.id}:{var.name}_true",
                    name=f"{component.name}_{var.name}_true",
                    parent_component=component.id,
                    parent_route=component.route_path,
                    controlling_variable=var.id,
                    variable_value=True,
                    rendered_components=self._extract_rendered_components(
                        related_conditionals, True
                    ),
                    hidden_components=self._extract_rendered_components(
                        related_conditionals, False
                    ),
                    toggle_handlers=[h.id for h in toggle_handlers],
                    conditional_render_id=(
                        related_conditionals[0].id if related_conditionals else None
                    ),
                    file_path=component.file_path,
                    line_number=var.line_number,
                    metadata={
                        "variable_name": var.name,
                        "is_default": var.initial_value is True,
                    },
                )
                visibility_states.append(visible_state)

        self.visibility_states.extend(visibility_states)
        return visibility_states

    def _identify_visibility_variables(
        self, state_vars: list[StateVariable]
    ) -> list[StateVariable]:
        """
        Identify state variables that likely control visibility.

        Common patterns:
        - is*, show*, *Open, *Visible, *Expanded, *Active, *Hidden, *Collapsed
        - Boolean variables used in conditional rendering

        Args:
            state_vars: List of state variables to filter

        Returns:
            List of state variables that appear to control visibility
        """
        visibility_vars = []

        # Common visibility-related prefixes and suffixes
        visibility_patterns = [
            "is",
            "show",
            "hide",
            "visible",
            "hidden",
            "open",
            "closed",
            "expanded",
            "collapsed",
            "active",
            "inactive",
            "enabled",
            "disabled",
            "toggle",
            "display",
            "render",
            "mounted",
        ]

        for var in state_vars:
            var_name_lower = var.name.lower()

            # Check if name contains visibility patterns
            for pattern in visibility_patterns:
                if (
                    var_name_lower.startswith(pattern)
                    or var_name_lower.endswith(pattern)
                    or pattern in var_name_lower
                ):
                    visibility_vars.append(var)
                    break
            else:
                # Also check if initial value is boolean (strong indicator)
                if var.initial_value is True or var.initial_value is False:
                    visibility_vars.append(var)

        return visibility_vars

    def _is_boolean_var(self, var: StateVariable) -> bool:
        """Check if a state variable is boolean-typed."""
        if var.value_type and "boolean" in var.value_type.lower():
            return True
        if var.initial_value is True or var.initial_value is False:
            return True
        return False

    def _find_toggle_handlers(
        self, var: StateVariable, handlers: list[EventHandler]
    ) -> list[EventHandler]:
        """
        Find event handlers that toggle or modify a state variable.

        Args:
            var: The state variable to find handlers for
            handlers: List of all event handlers

        Returns:
            List of handlers that modify this variable
        """
        toggle_handlers = []

        for handler in handlers:
            # Check if this handler modifies the state variable
            if var.id in handler.state_changes:
                toggle_handlers.append(handler)

        return toggle_handlers

    def _extract_rendered_components(
        self, conditionals: list[ConditionalRender], when_true: bool
    ) -> list[str]:
        """
        Extract component names that are rendered based on conditional state.

        Args:
            conditionals: List of conditional renders to analyze
            when_true: If True, extract components rendered when condition is true;
                      if False, extract components rendered when condition is false

        Returns:
            List of component/element names rendered in this state
        """
        components = []

        for cond in conditionals:
            if when_true:
                components.extend(cond.renders_when_true)
            else:
                components.extend(cond.renders_when_false)

        # Remove duplicates while preserving order
        seen = set()
        unique_components = []
        for comp in components:
            if comp and comp not in seen:
                seen.add(comp)
                unique_components.append(comp)

        return unique_components

    def get_state_variables(self) -> list[StateVariable]:
        """Get all extracted state variables."""
        return self.state_variables

    def get_visibility_states(self) -> list[VisibilityState]:
        """Get all extracted visibility states."""
        return self.visibility_states

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.state_variables = []
        self.visibility_states = []
