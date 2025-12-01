"""
React hooks extraction utilities.

This module provides functions to extract state variables from React hooks including:
- useState
- useReducer
- useContext
- Custom hooks (useQuery, useSelector, etc.)
"""

from pathlib import Path

from qontinui.extraction.models import StateScope, StateSourceType, StateVariable


def extract_use_state(
    parse_result: dict, component_name: str, file_path: Path
) -> list[StateVariable]:
    """
    Extract useState calls and their state variables.

    Patterns detected:
    - const [count, setCount] = useState(0)
    - const [state, setState] = useState({ ... })
    - const [isOpen, setIsOpen] = useState(false)

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component containing the useState
        file_path: Source file path

    Returns:
        List of StateVariable objects for each useState call
    """
    state_vars: list[StateVariable] = []

    # Find all variable declarations with useState
    declarations = parse_result.get("variable_declarations", [])

    for decl in declarations:
        # Check if this is a useState call
        if decl.get("initializer", {}).get("type") != "call_expression":
            continue

        callee = decl.get("initializer", {}).get("callee", {})
        if callee.get("name") != "useState":
            continue

        # Extract destructured names [state, setState]
        pattern = decl.get("pattern", {})
        if pattern.get("type") == "array_pattern":
            elements = pattern.get("elements", [])

            if len(elements) >= 2:
                state_name = elements[0].get("name", "")
                setter_name = elements[1].get("name", "")

                # Get initial value
                args = decl.get("initializer", {}).get("arguments", [])
                initial_value = None
                if args:
                    initial_value = _serialize_value(args[0])

                state_vars.append(
                    StateVariable(
                        id=f"{file_path}:{component_name}:{state_name}",
                        name=state_name,
                        file_path=file_path,
                        line_number=decl.get("line", 0),
                        initial_value=initial_value,
                        source_type=StateSourceType.HOOK,
                        source_name="useState",
                        scope=StateScope.LOCAL,
                        metadata={
                            "component": component_name,
                            "setter_name": setter_name,
                        },
                    )
                )

    return state_vars


def extract_use_reducer(
    parse_result: dict, component_name: str, file_path: Path
) -> list[StateVariable]:
    """
    Extract useReducer calls.

    Patterns detected:
    - const [state, dispatch] = useReducer(reducer, initialState)
    - const [state, dispatch] = useReducer(reducer, initialArg, init)

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of StateVariable objects for each useReducer call
    """
    state_vars: list[StateVariable] = []

    declarations = parse_result.get("variable_declarations", [])

    for decl in declarations:
        if decl.get("initializer", {}).get("type") != "call_expression":
            continue

        callee = decl.get("initializer", {}).get("callee", {})
        if callee.get("name") != "useReducer":
            continue

        pattern = decl.get("pattern", {})
        if pattern.get("type") == "array_pattern":
            elements = pattern.get("elements", [])

            if len(elements) >= 2:
                state_name = elements[0].get("name", "")
                dispatch_name = elements[1].get("name", "")

                # Get initial state (second argument)
                args = decl.get("initializer", {}).get("arguments", [])
                initial_value = None
                if len(args) >= 2:
                    initial_value = _serialize_value(args[1])

                state_vars.append(
                    StateVariable(
                        id=f"{file_path}:{component_name}:{state_name}",
                        name=state_name,
                        file_path=file_path,
                        line_number=decl.get("line", 0),
                        initial_value=initial_value,
                        source_type=StateSourceType.HOOK,
                        source_name="useReducer",
                        scope=StateScope.LOCAL,
                        metadata={
                            "component": component_name,
                            "dispatch_name": dispatch_name,
                        },
                    )
                )

    return state_vars


def extract_use_context(
    parse_result: dict, component_name: str, file_path: Path
) -> list[StateVariable]:
    """
    Extract useContext consumption.

    Patterns detected:
    - const context = useContext(MyContext)
    - const { user, setUser } = useContext(AuthContext)

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of StateVariable objects for each useContext call
    """
    state_vars: list[StateVariable] = []

    declarations = parse_result.get("variable_declarations", [])

    for decl in declarations:
        if decl.get("initializer", {}).get("type") != "call_expression":
            continue

        callee = decl.get("initializer", {}).get("callee", {})
        if callee.get("name") != "useContext":
            continue

        pattern = decl.get("pattern", {})

        # Get context name from argument
        args = decl.get("initializer", {}).get("arguments", [])
        context_name = args[0].get("name", "") if args else ""

        if pattern.get("type") == "identifier":
            # Simple pattern: const context = useContext(MyContext)
            state_name = pattern.get("name", "")
            state_vars.append(
                StateVariable(
                    id=f"{file_path}:{component_name}:{state_name}",
                    name=state_name,
                    file_path=file_path,
                    line_number=decl.get("line", 0),
                    source_type=StateSourceType.CONTEXT,
                    source_name=context_name,
                    scope=StateScope.CONTEXT,
                    metadata={
                        "component": component_name,
                        "context": context_name,
                    },
                )
            )
        elif pattern.get("type") == "object_pattern":
            # Destructured pattern: const { user, setUser } = useContext(AuthContext)
            properties = pattern.get("properties", [])
            for prop in properties:
                state_name = prop.get("key", {}).get("name", "")
                state_vars.append(
                    StateVariable(
                        id=f"{file_path}:{component_name}:{state_name}",
                        name=state_name,
                        file_path=file_path,
                        line_number=decl.get("line", 0),
                        source_type=StateSourceType.CONTEXT,
                        source_name=context_name,
                        scope=StateScope.CONTEXT,
                        metadata={
                            "component": component_name,
                            "context": context_name,
                        },
                    )
                )

    return state_vars


def extract_custom_hooks(
    parse_result: dict, component_name: str, file_path: Path
) -> list[StateVariable]:
    """
    Extract state from custom hooks (useQuery, useSelector, etc.).

    Patterns detected:
    - const { data, isLoading } = useQuery(...)
    - const user = useSelector(selectUser)
    - const [state, actions] = useCustomHook()

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of StateVariable objects for custom hook usage
    """
    state_vars: list[StateVariable] = []

    # Common custom hooks to detect
    custom_hooks = {
        "useQuery": "react-query",
        "useMutation": "react-query",
        "useSelector": "redux",
        "useDispatch": "redux",
        "useSearchParams": "react-router",
        "useParams": "react-router",
        "useNavigate": "react-router",
        "useLocation": "react-router",
        "useForm": "react-hook-form",
    }

    declarations = parse_result.get("variable_declarations", [])

    for decl in declarations:
        if decl.get("initializer", {}).get("type") != "call_expression":
            continue

        callee = decl.get("initializer", {}).get("callee", {})
        hook_name = callee.get("name", "")

        # Check if it's a known custom hook
        if hook_name not in custom_hooks and not hook_name.startswith("use"):
            continue

        pattern = decl.get("pattern", {})
        library = custom_hooks.get(hook_name, "custom")

        if pattern.get("type") == "identifier":
            # Simple pattern: const data = useQuery(...)
            state_name = pattern.get("name", "")
            state_vars.append(
                StateVariable(
                    id=f"{file_path}:{component_name}:{state_name}",
                    name=state_name,
                    file_path=file_path,
                    line_number=decl.get("line", 0),
                    source_type=StateSourceType.HOOK,
                    source_name=hook_name,
                    scope=StateScope.LOCAL,
                    metadata={
                        "component": component_name,
                        "library": library,
                    },
                )
            )
        elif pattern.get("type") == "object_pattern":
            # Destructured pattern: const { data, isLoading } = useQuery(...)
            properties = pattern.get("properties", [])
            for prop in properties:
                state_name = prop.get("key", {}).get("name", "")
                state_vars.append(
                    StateVariable(
                        id=f"{file_path}:{component_name}:{state_name}",
                        name=state_name,
                        file_path=file_path,
                        line_number=decl.get("line", 0),
                        source_type=StateSourceType.HOOK,
                        source_name=hook_name,
                        scope=StateScope.LOCAL,
                        metadata={
                            "component": component_name,
                            "library": library,
                        },
                    )
                )
        elif pattern.get("type") == "array_pattern":
            # Array pattern: const [state, actions] = useCustomHook()
            elements = pattern.get("elements", [])
            for i, elem in enumerate(elements):
                if elem:
                    state_name = elem.get("name", "")
                    state_vars.append(
                        StateVariable(
                            id=f"{file_path}:{component_name}:{state_name}:{i}",
                            name=state_name,
                            file_path=file_path,
                            line_number=decl.get("line", 0),
                            source_type=StateSourceType.HOOK,
                            source_name=hook_name,
                            scope=StateScope.LOCAL,
                            metadata={
                                "component": component_name,
                                "library": library,
                                "position": i,
                            },
                        )
                    )

    return state_vars


def _serialize_value(value_node: dict) -> str:
    """
    Serialize an AST value node to a string representation.

    Args:
        value_node: AST node representing a value

    Returns:
        String representation of the value
    """
    node_type = value_node.get("type", "")

    if node_type == "literal":
        return str(value_node.get("value", ""))
    elif node_type == "boolean":
        return str(value_node.get("value", "")).lower()
    elif node_type == "null":
        return "null"
    elif node_type == "undefined":
        return "undefined"
    elif node_type == "object_expression":
        # Serialize object as JSON-like string
        return "{...}"
    elif node_type == "array_expression":
        # Serialize array as JSON-like string
        return "[...]"
    elif node_type == "arrow_function":
        return "() => ..."
    elif node_type == "function_expression":
        return "function() {...}"
    else:
        # For complex expressions, just return the type
        return f"<{node_type}>"
