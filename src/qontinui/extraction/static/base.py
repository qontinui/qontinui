"""
Abstract base class for static code analysis.

This module defines the interface for static analyzers that extract UI structure
and state information from application source code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qontinui.extraction.config import FrameworkType, StaticConfig
    from qontinui.extraction.models.static import (
        APICallDefinition,
        ComponentDefinition,
        ConditionalRender,
        EventHandler,
        RouteDefinition,
        StateVariable,
        StaticAnalysisResult,
    )


class StaticAnalyzer(ABC):
    """
    Abstract base class for static code analysis.

    Static analyzers parse application source code to extract:
    - Component/widget definitions and their hierarchies
    - State variables that control UI visibility and behavior
    - Conditional rendering patterns (if/else, switches, etc.)
    - Event handlers and their effects on state
    - Routing configuration
    - API calls that affect application state

    This information forms the "known structure" that can be correlated
    with runtime observations to build a complete state model.
    """

    @abstractmethod
    async def analyze(self, config: StaticConfig) -> StaticAnalysisResult:
        """
        Analyze source code and return a complete state model.

        This is the main entry point for static analysis. It coordinates
        all the other extraction methods to build a comprehensive model
        of the application's UI structure.

        Args:
            config: Configuration specifying what to analyze (source paths,
                   framework type, analysis depth, etc.)

        Returns:
            StaticAnalysisResult containing all extracted information including
            components, state variables, conditional renders, event handlers,
            routes, and API calls.

        Raises:
            AnalysisError: If the source code cannot be parsed or analyzed.
        """
        pass

    @abstractmethod
    def get_components(self) -> list[ComponentDefinition]:
        """
        Extract component/widget definitions from the source code.

        Components are the building blocks of the UI. This method identifies:
        - Component names and types (e.g., React components, Vue components)
        - Component hierarchies (parent-child relationships)
        - Props/parameters that control component behavior
        - Component lifecycle methods
        - Component file locations

        Returns:
            List of ComponentDefinition objects representing all components
            found in the analyzed source code.
        """
        pass

    @abstractmethod
    def get_state_variables(self) -> list[StateVariable]:
        """
        Extract state variables that control UI visibility and behavior.

        State variables are the data that determines what the user sees.
        This method identifies:
        - State variable names and types
        - Initial/default values
        - Scope (component-local, global, context, store, etc.)
        - State management patterns (useState, Redux, Vuex, etc.)
        - Variables that control conditional rendering

        Returns:
            List of StateVariable objects representing state that affects
            the UI.
        """
        pass

    @abstractmethod
    def get_conditional_renders(self) -> list[ConditionalRender]:
        """
        Extract conditional rendering patterns from the source code.

        Conditional rendering determines which UI elements appear based on
        state. This method identifies:
        - If/else statements that render different UI
        - Ternary operators used for conditional rendering
        - Switch/case statements for multiple UI variants
        - Logical AND/OR operators for conditional rendering
        - The state variables and conditions that control each branch

        Returns:
            List of ConditionalRender objects representing each conditional
            rendering pattern found in the code.
        """
        pass

    @abstractmethod
    def get_event_handlers(self) -> list[EventHandler]:
        """
        Extract event handlers and their effects on state.

        Event handlers are functions that respond to user interactions.
        This method identifies:
        - Event handler functions (onClick, onChange, etc.)
        - The UI elements they're attached to
        - State mutations they perform (setState, dispatch, etc.)
        - Side effects (API calls, navigation, etc.)
        - Event types (click, hover, focus, etc.)

        Returns:
            List of EventHandler objects representing all event handlers
            and their state effects.
        """
        pass

    @abstractmethod
    def get_routes(self) -> list[RouteDefinition]:
        """
        Extract routing configuration from the application.

        Routes define navigation between different views/pages. This method
        identifies:
        - Route paths and patterns
        - Components/views associated with each route
        - Route parameters and query strings
        - Navigation guards and middleware
        - Nested/child routes

        Returns:
            List of RouteDefinition objects representing the application's
            routing structure.
        """
        pass

    @abstractmethod
    def get_api_calls(self) -> list[APICallDefinition]:
        """
        Extract API/backend calls that affect application state.

        API calls often trigger state changes that affect the UI. This method
        identifies:
        - API endpoints and HTTP methods
        - Request parameters and bodies
        - Response handling and state updates
        - Error handling
        - Loading states triggered by API calls

        Returns:
            List of APICallDefinition objects representing API calls that
            affect UI state.
        """
        pass

    @classmethod
    @abstractmethod
    def supports_framework(cls, framework: FrameworkType) -> bool:
        """
        Check if this analyzer supports the given framework.

        Different analyzers are designed for different frameworks (React, Vue,
        Angular, Flutter, etc.). This method allows the framework to select
        the appropriate analyzer for a given codebase.

        Args:
            framework: The framework type to check support for.

        Returns:
            True if this analyzer can analyze code from the given framework,
            False otherwise.
        """
        pass
