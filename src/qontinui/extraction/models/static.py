"""
Static analysis models.

These models represent the results of static code analysis - extracting
component structure, state variables, conditional rendering, routes, and
event handlers from source code.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ComponentType(Enum):
    """Type of component definition."""

    FUNCTION = "function"  # Function component (React, Vue Composition API)
    CLASS = "class"  # Class component (React, Angular)
    SERVER = "server"  # Server component (React Server Components)
    SFC = "sfc"  # Single File Component (Vue, Svelte)
    WIDGET = "widget"  # Widget (Flutter)


class StateSourceType(Enum):
    """Source/mechanism for state management."""

    HOOK = "hook"  # useState, useReducer, etc.
    CONTEXT = "context"  # React Context, Vue provide/inject
    STORE = "store"  # Redux, Vuex, Pinia, etc.
    SIGNAL = "signal"  # Solid signals, Angular signals
    REF = "ref"  # Vue ref, React useRef
    PROP = "prop"  # Props passed from parent


class StateScope(Enum):
    """Scope of state variable."""

    LOCAL = "local"  # Component-local state
    CONTEXT = "context"  # Shared via context/provide
    GLOBAL = "global"  # Global store
    ROUTE = "route"  # URL/route parameters


class ConditionalPattern(Enum):
    """Pattern used for conditional rendering."""

    AND = "and"  # condition && <Element>
    TERNARY = "ternary"  # condition ? <A> : <B>
    EARLY_RETURN = "early_return"  # if (!condition) return null
    SWITCH = "switch"  # switch/case statements
    V_IF = "v_if"  # Vue v-if
    V_SHOW = "v_show"  # Vue v-show
    IF_BLOCK = "if_block"  # Svelte {#if}
    NG_IF = "ng_if"  # Angular *ngIf


class RouteType(Enum):
    """Type of route definition."""

    PAGE = "page"  # Regular page route
    LAYOUT = "layout"  # Layout component
    API = "api"  # API route
    MIDDLEWARE = "middleware"  # Route middleware


class APICallType(Enum):
    """Type of API call mechanism."""

    FETCH = "fetch"  # fetch() API
    AXIOS = "axios"  # Axios library
    REACT_QUERY = "react_query"  # React Query/TanStack Query
    SERVER_ACTION = "server_action"  # Next.js Server Actions
    TAURI_INVOKE = "tauri_invoke"  # Tauri invoke


@dataclass
class ComponentDefinition:
    """A component definition found in source code."""

    id: str
    name: str
    file_path: Path
    line_number: int
    component_type: ComponentType
    framework: str  # react, vue, svelte, etc.

    # Hierarchy
    parent_component: str | None = None  # Parent component ID
    child_components: list[str] = field(default_factory=list)  # Child component IDs

    # Props and state
    props: dict[str, str] = field(default_factory=dict)  # prop_name -> type
    state_variables_used: list[str] = field(default_factory=list)  # State variable IDs

    # Routing
    route_path: str | None = None  # If this is a route component

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateVariable:
    """A state variable found in source code."""

    id: str
    name: str
    file_path: Path
    line_number: int

    # Type information
    value_type: str | None = None  # TypeScript type or inferred type
    possible_values: list[Any] = field(default_factory=list)  # Known possible values
    initial_value: Any = None

    # Source
    source_type: StateSourceType = StateSourceType.HOOK
    source_name: str = ""  # e.g., "useState", "contextName", "storeName"
    scope: StateScope = StateScope.LOCAL

    # Impact
    controls_visibility: list[str] = field(default_factory=list)  # Conditional render IDs
    affected_components: list[str] = field(default_factory=list)  # Component IDs

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalRender:
    """A conditional rendering pattern found in source code."""

    id: str
    file_path: Path
    line_number: int
    condition: str  # Source code of the condition
    condition_ast: dict[str, Any] = field(default_factory=dict)  # Parsed AST

    # Dependencies
    controlling_variables: list[str] = field(default_factory=list)  # State variable IDs

    # Rendered elements
    renders_when_true: list[str] = field(default_factory=list)  # Component/element IDs
    renders_when_false: list[str] = field(default_factory=list)  # Component/element IDs

    # Pattern
    pattern_type: ConditionalPattern = ConditionalPattern.TERNARY

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteParam:
    """A route parameter definition."""

    name: str
    type: str = "string"  # string, number, slug, etc.
    optional: bool = False
    catch_all: bool = False  # [...slug] style


@dataclass
class SearchParam:
    """A search/query parameter definition."""

    name: str
    type: str = "string"
    optional: bool = True
    default_value: Any = None


@dataclass
class RouteDefinition:
    """A route definition found in source code."""

    id: str
    path: str  # Route path pattern
    file_path: Path
    route_type: RouteType = RouteType.PAGE

    # Parameters
    params: list[RouteParam] = field(default_factory=list)
    search_params: list[SearchParam] = field(default_factory=list)

    # Component
    component: str | None = None  # Component ID
    parent_layout: str | None = None  # Parent layout route ID

    # Special states
    has_loading_state: bool = False
    has_error_boundary: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandler:
    """An event handler found in source code."""

    id: str
    file_path: Path
    line_number: int
    handler_name: str  # Function name
    event_type: str  # click, submit, change, etc.

    # Trigger
    trigger_element: str | None = None  # Component/element ID
    trigger_selector: str | None = None  # CSS selector if available

    # Effects
    state_changes: list[str] = field(default_factory=list)  # State variable IDs modified
    navigation: str | None = None  # Route navigated to
    api_calls: list[str] = field(default_factory=list)  # API call IDs

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class APICallDefinition:
    """An API call found in source code."""

    id: str
    file_path: Path
    line_number: int
    method: str  # GET, POST, etc.
    endpoint: str  # API endpoint or pattern

    # Context
    triggered_by: list[str] = field(default_factory=list)  # Event handler IDs
    affects_state: list[str] = field(default_factory=list)  # State variable IDs

    # Type
    call_type: APICallType = APICallType.FETCH

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticAnalysisResult:
    """Complete result of static code analysis."""

    components: list[ComponentDefinition] = field(default_factory=list)
    state_variables: list[StateVariable] = field(default_factory=list)
    conditional_renders: list[ConditionalRender] = field(default_factory=list)
    routes: list[RouteDefinition] = field(default_factory=list)
    event_handlers: list[EventHandler] = field(default_factory=list)
    api_calls: list[APICallDefinition] = field(default_factory=list)

    # Lookups
    def get_component(self, component_id: str) -> ComponentDefinition | None:
        """Get component by ID."""
        for comp in self.components:
            if comp.id == component_id:
                return comp
        return None

    def get_state_variable(self, var_id: str) -> StateVariable | None:
        """Get state variable by ID."""
        for var in self.state_variables:
            if var.id == var_id:
                return var
        return None

    def get_route(self, route_id: str) -> RouteDefinition | None:
        """Get route by ID."""
        for route in self.routes:
            if route.id == route_id:
                return route
        return None

    def get_event_handler(self, handler_id: str) -> EventHandler | None:
        """Get event handler by ID."""
        for handler in self.event_handlers:
            if handler.id == handler_id:
                return handler
        return None

    def get_api_call(self, call_id: str) -> APICallDefinition | None:
        """Get API call by ID."""
        for call in self.api_calls:
            if call.id == call_id:
                return call
        return None
