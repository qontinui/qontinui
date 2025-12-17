"""
Event handler extraction utilities.

This module provides functions to extract event handlers and trace their effects:
- Extract onClick, onChange, onSubmit, etc. handlers
- Trace which state setters are called
- Identify API calls and navigation
"""

from pathlib import Path

from qontinui.extraction.models import EventHandler, StateVariable


def extract_event_handlers(
    parse_result: dict,
    component_name: str,
    file_path: Path,
    state_variables: list[StateVariable],
) -> list[EventHandler]:
    """
    Extract onClick, onChange, onSubmit handlers.

    Detects:
    - Inline arrow functions: onClick={() => ...}
    - Function references: onClick={handleClick}
    - Method calls: onClick={this.handleClick}

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path
        state_variables: List of state variables to trace mutations

    Returns:
        List of EventHandler objects
    """
    handlers: list[EventHandler] = []

    # Extract handlers from JSX attributes
    jsx_attributes = parse_result.get("jsx_attributes", [])

    for attr in jsx_attributes:
        attr_name = attr.get("name", "")

        # Check if this is an event handler (starts with 'on')
        if not attr_name.startswith("on"):
            continue

        # Determine event type from attribute name
        event_type = _event_type_from_name(attr_name)

        value = attr.get("value", {})
        handler_info = _extract_handler_info(value, parse_result)

        if handler_info:
            # Trace state mutations in the handler
            state_mutations = trace_state_changes(handler_info, state_variables)

            # Extract API calls
            api_calls = _extract_api_calls(handler_info)

            # Extract navigation calls
            navigations = _extract_navigation_calls(handler_info)

            handlers.append(
                EventHandler(
                    id=f"{file_path}:{component_name}:{handler_info.get('name', 'inline')}",
                    file_path=file_path,
                    line_number=attr.get("line", 0),
                    handler_name=handler_info.get("name", "inline"),
                    event_type=event_type,
                    trigger_element=attr.get("element", ""),
                    state_changes=state_mutations,
                    navigation=navigations[0] if navigations else None,
                    api_calls=api_calls,
                    metadata={
                        "component": component_name,
                        "attribute": attr_name,
                        "inline": handler_info.get("inline", False),
                    },
                )
            )

    # Also extract handlers defined as function declarations/expressions
    function_declarations = parse_result.get("function_declarations", [])
    for func in function_declarations:
        func_name = func.get("name", "")

        # Check if function name suggests it's a handler
        if _is_handler_name(func_name):
            event_type = _event_type_from_name(func_name)

            state_mutations = trace_state_changes(func, state_variables)
            api_calls = _extract_api_calls(func)
            navigations = _extract_navigation_calls(func)

            handlers.append(
                EventHandler(
                    id=f"{file_path}:{component_name}:{func_name}",
                    file_path=file_path,
                    line_number=func.get("line", 0),
                    handler_name=func_name,
                    event_type=event_type,
                    state_changes=state_mutations,
                    navigation=navigations[0] if navigations else None,
                    api_calls=api_calls,
                    metadata={
                        "component": component_name,
                        "type": "function_declaration",
                    },
                )
            )

    return handlers


def trace_state_changes(
    handler: dict, state_variables: list[StateVariable]
) -> list[str]:
    """
    Trace which state setters are called in a handler.

    Args:
        handler: AST node representing the handler function
        state_variables: List of state variables with their setters

    Returns:
        List of state setter names called in the handler
    """
    setters_called: list[str] = []

    # Build a set of all setter names from metadata
    setter_names = {
        var.metadata.get("setter_name")
        for var in state_variables
        if var.metadata.get("setter_name")
    }

    # Find all call expressions in the handler
    call_expressions = handler.get("call_expressions", [])

    for call in call_expressions:
        callee = call.get("callee", {})
        callee_name = _get_callee_name(callee)

        if callee_name in setter_names:
            # Find the state variable ID for this setter
            for var in state_variables:
                if var.metadata.get("setter_name") == callee_name:
                    setters_called.append(var.id)
                    break

    return list(set(setters_called))  # Remove duplicates


def _event_type_from_name(name: str) -> str:
    """
    Determine event type from attribute/function name.

    Args:
        name: Attribute name (e.g., 'onClick') or function name (e.g., 'handleClick')

    Returns:
        Event type (e.g., 'click', 'change', 'submit')
    """
    # Map common event handlers to event types
    event_map = {
        "onClick": "click",
        "onChange": "change",
        "onSubmit": "submit",
        "onInput": "input",
        "onFocus": "focus",
        "onBlur": "blur",
        "onKeyDown": "keydown",
        "onKeyUp": "keyup",
        "onKeyPress": "keypress",
        "onMouseEnter": "mouseenter",
        "onMouseLeave": "mouseleave",
        "onMouseOver": "mouseover",
        "onMouseOut": "mouseout",
        "onMouseDown": "mousedown",
        "onMouseUp": "mouseup",
        "onDoubleClick": "dblclick",
        "onDrag": "drag",
        "onDrop": "drop",
        "onScroll": "scroll",
        "onWheel": "wheel",
        "onTouchStart": "touchstart",
        "onTouchEnd": "touchend",
        "onTouchMove": "touchmove",
    }

    # Check if it's a JSX attribute
    if name in event_map:
        return event_map[name]

    # Try to extract from function name (e.g., handleClick -> click)
    lower_name = name.lower()
    if "click" in lower_name:
        return "click"
    elif "change" in lower_name:
        return "change"
    elif "submit" in lower_name:
        return "submit"
    elif "input" in lower_name:
        return "input"
    elif "focus" in lower_name:
        return "focus"
    elif "blur" in lower_name:
        return "blur"
    elif "key" in lower_name:
        return "key"
    elif "mouse" in lower_name:
        return "mouse"
    elif "touch" in lower_name:
        return "touch"
    else:
        return "unknown"


def _is_handler_name(name: str) -> bool:
    """
    Check if a function name suggests it's an event handler.

    Args:
        name: Function name

    Returns:
        True if the name suggests an event handler
    """
    handler_prefixes = ["handle", "on"]
    handler_suffixes = ["Handler", "Callback"]

    lower_name = name.lower()

    # Check prefixes
    for prefix in handler_prefixes:
        if lower_name.startswith(prefix):
            return True

    # Check suffixes
    for suffix in handler_suffixes:
        if name.endswith(suffix):
            return True

    return False


def _extract_handler_info(value_node: dict, parse_result: dict) -> dict | None:
    """
    Extract information about a handler from a JSX attribute value.

    Args:
        value_node: JSX attribute value AST node
        parse_result: Full parse result for function lookup

    Returns:
        Dictionary with handler info or None
    """
    node_type = value_node.get("type", "")

    if node_type == "jsx_expression_container":
        expression = value_node.get("expression", {})
        return _extract_handler_info(expression, parse_result)

    elif node_type == "arrow_function_expression":
        # Inline arrow function
        return {
            "type": "arrow_function",
            "name": "inline",
            "inline": True,
            "call_expressions": _extract_calls_from_body(value_node.get("body", {})),
        }

    elif node_type == "function_expression":
        # Inline function expression
        return {
            "type": "function_expression",
            "name": "inline",
            "inline": True,
            "call_expressions": _extract_calls_from_body(value_node.get("body", {})),
        }

    elif node_type == "identifier":
        # Reference to a function
        func_name = value_node.get("name", "")

        # Try to find the function definition
        for func in parse_result.get("function_declarations", []):
            if func.get("name") == func_name:
                return {
                    "type": "function_reference",
                    "name": func_name,
                    "inline": False,
                    "call_expressions": func.get("call_expressions", []),
                    "line": func.get("line", 0),
                }

        # If not found, still return the reference
        return {
            "type": "function_reference",
            "name": func_name,
            "inline": False,
            "call_expressions": [],
        }

    elif node_type == "member_expression":
        # Method reference (e.g., this.handleClick)
        property_name = value_node.get("property", {}).get("name", "")
        return {
            "type": "member_expression",
            "name": property_name,
            "inline": False,
            "call_expressions": [],
        }

    return None


def _extract_calls_from_body(body_node: dict) -> list[dict]:
    """
    Extract call expressions from a function body.

    Args:
        body_node: AST node representing function body

    Returns:
        List of call expression nodes
    """
    calls: list[dict] = []

    def visit_node(node: dict):
        if not isinstance(node, dict):
            return

        node_type = node.get("type", "")

        if node_type == "call_expression":
            calls.append(node)

        # Recursively visit children
        for value in node.values():
            if isinstance(value, dict):
                visit_node(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        visit_node(item)

    visit_node(body_node)
    return calls


def _get_callee_name(callee_node: dict) -> str:
    """
    Get the name of a function being called.

    Args:
        callee_node: Callee AST node

    Returns:
        Function name
    """
    node_type = callee_node.get("type", "")

    if node_type == "identifier":
        name = callee_node.get("name", "")
        return str(name) if name is not None else ""
    elif node_type == "member_expression":
        # For member expressions, get the property name
        name = callee_node.get("property", {}).get("name", "")
        return str(name) if name is not None else ""
    else:
        return ""


def _extract_api_calls(handler: dict) -> list[str]:
    """
    Extract API call identifiers from a handler.

    Args:
        handler: Handler info dictionary

    Returns:
        List of API call names/patterns
    """
    api_calls: list[str] = []

    call_expressions = handler.get("call_expressions", [])

    for call in call_expressions:
        callee = call.get("callee", {})
        callee_name = _get_callee_name(callee)

        # Check for common API call patterns
        if callee_name in ["fetch", "axios", "get", "post", "put", "delete", "patch"]:
            api_calls.append(callee_name)
        elif callee_name.startswith("api") or callee_name.endswith("Api"):
            api_calls.append(callee_name)

    return list(set(api_calls))


def _extract_navigation_calls(handler: dict) -> list[str]:
    """
    Extract navigation/routing calls from a handler.

    Args:
        handler: Handler info dictionary

    Returns:
        List of navigation call names
    """
    navigations: list[str] = []

    call_expressions = handler.get("call_expressions", [])

    for call in call_expressions:
        callee = call.get("callee", {})
        callee_name = _get_callee_name(callee)

        # Check for common navigation patterns
        if callee_name in [
            "navigate",
            "push",
            "replace",
            "goBack",
            "goForward",
            "redirect",
        ]:
            navigations.append(callee_name)

    return list(set(navigations))
