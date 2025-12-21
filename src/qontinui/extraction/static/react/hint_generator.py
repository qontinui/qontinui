"""
Hint Generator for React.

Generates runtime hints from static analysis results.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.static.models import (
    ComponentCategory,
    ComponentDefinition,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    StateHint,
    StateImageHint,
    StateVariable,
    TransitionHint,
    VisibilityState,
)

logger = logging.getLogger(__name__)


class HintGenerator:
    """Generator for creating runtime hints from static analysis."""

    def generate_hints(
        self,
        components: list[ComponentDefinition],
        state_variables: list[StateVariable],
        conditional_renders: list[ConditionalRender],
        event_handlers: list[EventHandler],
        routes: list[RouteDefinition],
        visibility_states: list[VisibilityState],
        navigation_links: list[dict],
    ) -> tuple[list[StateHint], list[StateImageHint], list[TransitionHint]]:
        """
        Generate hints for runtime state discovery.

        This method analyzes the extracted components, routes, conditional renders,
        event handlers, and visibility states to produce hints that guide runtime
        state discovery.

        Args:
            components: Extracted components
            state_variables: Extracted state variables
            conditional_renders: Extracted conditional renders
            event_handlers: Extracted event handlers
            routes: Extracted routes
            visibility_states: Extracted visibility states
            navigation_links: Extracted navigation links

        Returns:
            Tuple of (state_hints, state_image_hints, transition_hints)
        """
        state_hints: list[StateHint] = []
        state_image_hints: list[StateImageHint] = []
        transition_hints: list[TransitionHint] = []

        # Build lookup maps
        component_by_id = {c.id: c for c in components}
        {s.id: s for s in state_variables}
        {h.id: h for h in event_handlers}

        # 1. Generate StateHints from routes (each route is a potential state)
        for route in routes:
            state_hint = StateHint(
                id=f"state_hint_{route.id}",
                name=self._route_to_state_name(route.path),
                source_type="route",
                file_path=route.file_path,
                line_number=0,
                route_path=route.path,
                route_params=[p.name for p in route.params],
                metadata={"route_type": route.route_type.value},
            )
            state_hints.append(state_hint)

        # 2. Generate StateHints from visibility states (sub-states within pages)
        for vis_state in visibility_states:
            parent_hint_id = None
            # Find parent route state hint
            parent_comp = component_by_id.get(vis_state.parent_component)
            if parent_comp and parent_comp.route_path:
                parent_hint_id = f"state_hint_route_{parent_comp.route_path}"

            state_hint = StateHint(
                id=f"state_hint_{vis_state.id}",
                name=vis_state.name,
                source_type="conditional_render",
                file_path=vis_state.file_path,
                line_number=vis_state.line_number,
                parent_state_hint_id=parent_hint_id,
                controlling_variable=vis_state.controlling_variable,
                condition_value=vis_state.variable_value,
                metadata={
                    "rendered_components": vis_state.rendered_components,
                    "hidden_components": vis_state.hidden_components,
                },
            )
            state_hints.append(state_hint)

        # 3. Generate StateImageHints from interactive components
        for component in components:
            # Skip non-interactive widgets
            if component.category != ComponentCategory.WIDGET:
                continue

            # Find event handlers attached to this component
            component_handlers = [h for h in event_handlers if h.trigger_element == component.id]

            if not component_handlers:
                continue

            # Determine interaction type
            interaction_types = {h.event_type for h in component_handlers}
            primary_interaction = (
                "click" if "click" in interaction_types else list(interaction_types)[0]
            )

            # Check if conditionally rendered
            is_conditional = any(
                component.id in cr.renders_when_true or component.id in cr.renders_when_false
                for cr in conditional_renders
            )

            state_image_hint = StateImageHint(
                id=f"state_image_hint_{component.id}",
                name=component.name,
                component_id=component.id,
                file_path=component.file_path,
                line_number=component.line_number,
                element_type=self._infer_element_type(component.name),
                jsx_element_name=component.name,
                is_interactive=True,
                interaction_type=primary_interaction,
                conditionally_rendered=is_conditional,
                metadata={"handlers": [h.id for h in component_handlers]},
            )
            state_image_hints.append(state_image_hint)

        # 4. Generate TransitionHints from navigation links
        for nav_link in navigation_links:
            target_path = nav_link.get("target", "")
            if not target_path or target_path.startswith("http"):
                continue  # Skip external links

            # Find source state hint (based on file/component)
            from_state = None
            source_file = nav_link.get("file", "")
            for sh in state_hints:
                if sh.file_path and str(sh.file_path) == source_file:
                    from_state = sh.id
                    break

            # Find target state hint
            to_state = None
            for sh in state_hints:
                if sh.route_path == target_path:
                    to_state = sh.id
                    break

            transition_hint = TransitionHint(
                id=f"transition_hint_nav_{len(transition_hints)}",
                from_state_hint=from_state,
                to_state_hint=to_state,
                trigger_type="navigation",
                navigation_path=target_path,
                file_path=Path(source_file) if source_file else None,
                line_number=nav_link.get("line", 0),
                confidence=0.8 if to_state else 0.5,
            )
            transition_hints.append(transition_hint)

        # 5. Generate TransitionHints from event handlers that modify state
        for handler in event_handlers:
            # Check if handler navigates
            if handler.navigation:
                to_state = None
                for sh in state_hints:
                    if sh.route_path == handler.navigation:
                        to_state = sh.id
                        break

                transition_hint = TransitionHint(
                    id=f"transition_hint_handler_{handler.id}",
                    from_state_hint=None,  # Would need component context
                    to_state_hint=to_state,
                    trigger_type=handler.event_type,
                    event_handler_id=handler.id,
                    navigation_path=handler.navigation,
                    file_path=handler.file_path,
                    line_number=handler.line_number,
                    confidence=0.7,
                )
                transition_hints.append(transition_hint)

            # Check if handler changes visibility state
            for state_change in handler.state_changes:
                # Find visibility states controlled by this variable
                for vis_state in visibility_states:
                    if vis_state.controlling_variable == state_change:
                        transition_hint = TransitionHint(
                            id=f"transition_hint_vis_{handler.id}_{vis_state.id}",
                            from_state_hint=None,
                            to_state_hint=f"state_hint_{vis_state.id}",
                            trigger_type="state_change",
                            event_handler_id=handler.id,
                            file_path=handler.file_path,
                            line_number=handler.line_number,
                            confidence=0.6,
                            metadata={"state_variable": state_change},
                        )
                        transition_hints.append(transition_hint)

        return state_hints, state_image_hints, transition_hints

    def _route_to_state_name(self, route_path: str) -> str:
        """Convert a route path to a readable state name."""
        if route_path == "/" or route_path == "":
            return "HomePage"

        # Remove leading slash and convert to PascalCase
        parts = route_path.strip("/").split("/")
        name_parts = []
        for part in parts:
            if part.startswith(":"):
                # Dynamic segment
                name_parts.append(part[1:].title() + "Dynamic")
            elif part.startswith("[") and part.endswith("]"):
                # Next.js dynamic segment
                name_parts.append(part[1:-1].title() + "Dynamic")
            else:
                name_parts.append(part.title().replace("-", "").replace("_", ""))

        return "".join(name_parts) + "Page"

    def _infer_element_type(self, component_name: str) -> str:
        """Infer the element type from the component name."""
        name_lower = component_name.lower()

        if "button" in name_lower or "btn" in name_lower:
            return "button"
        elif "input" in name_lower or "field" in name_lower or "text" in name_lower:
            return "input"
        elif "icon" in name_lower:
            return "icon"
        elif "image" in name_lower or "img" in name_lower or "avatar" in name_lower:
            return "image"
        elif "link" in name_lower or "anchor" in name_lower:
            return "link"
        elif "modal" in name_lower or "dialog" in name_lower:
            return "modal"
        elif "menu" in name_lower or "dropdown" in name_lower:
            return "menu"
        elif "card" in name_lower:
            return "card"
        elif "list" in name_lower or "table" in name_lower:
            return "list"
        else:
            return "component"
