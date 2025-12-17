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


class ComponentCategory(Enum):
    """Category of component in GUI state machine modeling."""

    STATE = "state"  # Page-level component (navigable screen/view)
    WIDGET = "widget"  # UI element within a state (button, input, card, etc.)


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

    # Classification for GUI state machine modeling
    category: ComponentCategory = ComponentCategory.WIDGET  # Default to widget

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
    controls_visibility: list[str] = field(
        default_factory=list
    )  # Conditional render IDs
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
    state_changes: list[str] = field(
        default_factory=list
    )  # State variable IDs modified
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
class VisibilityState:
    """A visibility-based sub-state within a component/page.

    Represents different UI configurations of the same page based on
    conditional rendering and visibility toggles (e.g., modals, sidebars, dropdowns).
    """

    id: str
    name: str  # e.g., "page_sidebar_open", "page_modal_visible"
    parent_component: str  # Component ID this sub-state belongs to
    parent_route: str | None = None  # Route path if applicable

    # Controlling state variable
    controlling_variable: str | None = None  # StateVariable ID
    variable_value: Any = None  # Expected value (True/False for booleans)

    # What gets rendered in this state
    rendered_components: list[str] = field(
        default_factory=list
    )  # Component/element names
    hidden_components: list[str] = field(
        default_factory=list
    )  # Components hidden in this state

    # Transitions to/from this state
    toggle_handlers: list[str] = field(
        default_factory=list
    )  # EventHandler IDs that toggle this state

    # Conditional render that defines this state
    conditional_render_id: str | None = None

    # Metadata
    file_path: Path | None = None
    line_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateImageHint:
    """A hint for a potential StateImage from static analysis.

    StateImages are visual patterns that identify states. Static analysis can identify
    potential UI elements that may become StateImages during runtime extraction.

    In qontinui, a StateImage is a visual pattern with:
    - Multiple patterns (variations like normal/hover/clicked)
    - Search regions (where to look for this pattern on screen)
    - Pixel data (coordinates, hashes, stability scores)

    Static analysis CANNOT produce actual StateImages (which require screenshots).
    Instead, it produces hints about UI elements that may become StateImages.
    """

    id: str
    name: str  # e.g., "LoginButton", "SubmitForm"

    # Source information
    component_id: str  # The component this element belongs to
    file_path: Path
    line_number: int

    # Element type hint
    element_type: str  # "button", "input", "icon", "image", "text", etc.
    jsx_element_name: str  # The JSX element name (e.g., "Button", "Icon", "img")

    # Interaction hints
    is_interactive: bool = False  # Has onClick, onChange, etc.
    interaction_type: str | None = None  # "click", "input", "hover", etc.

    # Visibility hints
    conditionally_rendered: bool = False  # Is this conditionally rendered?
    controlling_variable: str | None = None  # StateVariable ID if conditional

    # Text content hints (for OCR/text-based StateImages)
    text_content: str | None = None  # Static text content if available
    has_dynamic_text: bool = False  # Contains variables/expressions

    # Layout hints (relative, not absolute - actual positions require runtime)
    parent_element: str | None = None  # Parent element hint
    css_selector_hint: str | None = None  # CSS selector if determinable

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateHint:
    """A hint for a potential State from static analysis.

    In qontinui, a State is a visual screen configuration identified by StateImages.
    States are NOT routes or components - they are specific visual configurations.

    Static analysis identifies:
    - Routes/pages (each route is potentially a different state)
    - Conditional renders (modal open vs closed = different states)
    - Loading states, error boundaries, etc.

    StateHints guide runtime extraction to discover actual states with StateImages.
    """

    id: str
    name: str  # Descriptive name like "LoginPage", "Dashboard_ModalOpen"

    # Source
    source_type: str  # "route", "conditional_render", "loading_state", "error_boundary"
    file_path: Path | None = None
    line_number: int = 0

    # Route info (if source_type == "route")
    route_path: str | None = None
    route_params: list[str] = field(default_factory=list)

    # Parent relationship (for sub-states)
    parent_state_hint_id: str | None = (
        None  # e.g., modal state's parent is the page state
    )

    # Conditional info (if source_type == "conditional_render")
    controlling_variable: str | None = None
    condition_value: Any = None  # Expected value for this state variant

    # UI element hints that may become StateImages
    state_image_hints: list[str] = field(default_factory=list)  # StateImageHint IDs

    # Transition hints
    outgoing_transition_hints: list[str] = field(
        default_factory=list
    )  # Element IDs that trigger transitions
    incoming_from: list[str] = field(
        default_factory=list
    )  # StateHint IDs that can navigate here

    # Navigation hint
    navigation_target: str | None = None  # For states reachable via navigation

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionHint:
    """A hint for a potential transition from static analysis.

    Transitions connect states and are triggered by user actions.
    Static analysis identifies potential transitions from:
    - Link/navigation elements
    - Event handlers that change state
    - Form submissions
    """

    # Required fields first
    id: str
    trigger_type: str  # "click", "submit", "navigation", "state_change"

    # Optional fields with defaults
    from_state_hint: str | None = None  # StateHint ID
    to_state_hint: str | None = None  # StateHint ID
    trigger_element_hint: str | None = None  # StateImageHint ID

    # Handler info
    event_handler_id: str | None = None  # EventHandler ID
    navigation_path: str | None = None  # For Link-based navigation

    # Source
    file_path: Path | None = None
    line_number: int = 0

    # Confidence (how certain is this transition from static analysis)
    confidence: float = 0.5

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticAnalysisResult:
    """Complete result of static code analysis.

    NOTE: Static analysis produces HINTS, not final states/StateImages.
    Actual State and StateImage objects require runtime extraction with screenshots.

    The hints guide runtime extraction to:
    1. Navigate to potential states (via route hints)
    2. Trigger state variations (via conditional render hints)
    3. Identify UI elements as StateImages (via component/element hints)
    """

    # Raw extracted data
    components: list[ComponentDefinition] = field(default_factory=list)
    state_variables: list[StateVariable] = field(default_factory=list)
    conditional_renders: list[ConditionalRender] = field(default_factory=list)
    routes: list[RouteDefinition] = field(default_factory=list)
    event_handlers: list[EventHandler] = field(default_factory=list)
    api_calls: list[APICallDefinition] = field(default_factory=list)
    visibility_states: list[VisibilityState] = field(default_factory=list)

    # Hints for runtime extraction (the main output for state discovery)
    state_hints: list[StateHint] = field(default_factory=list)
    state_image_hints: list[StateImageHint] = field(default_factory=list)
    transition_hints: list[TransitionHint] = field(default_factory=list)

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

    # Component classification helpers
    def get_page_components(self) -> list[ComponentDefinition]:
        """Get all page-level components (states in GUI state machine)."""
        return [c for c in self.components if c.category == ComponentCategory.STATE]

    def get_widget_components(self) -> list[ComponentDefinition]:
        """Get all UI widget components (UI elements within states)."""
        return [c for c in self.components if c.category == ComponentCategory.WIDGET]

    def count_page_components(self) -> int:
        """Count page-level components (states)."""
        return sum(1 for c in self.components if c.category == ComponentCategory.STATE)

    def count_widget_components(self) -> int:
        """Count UI widget components."""
        return sum(1 for c in self.components if c.category == ComponentCategory.WIDGET)
