"""
Component extraction utilities.

This module provides functions to extract React components:
- Function components
- Class components
- Component hierarchies and relationships
"""

from pathlib import Path

from qontinui.extraction.models import ComponentCategory, ComponentDefinition, ComponentType


def extract_function_components(parse_result: dict, file_path: Path) -> list[ComponentDefinition]:
    """
    Extract function component definitions.

    Detects:
    - Function declarations: function MyComponent() { ... }
    - Arrow functions: const MyComponent = () => { ... }
    - Named exports: export function MyComponent() { ... }
    - Default exports: export default function MyComponent() { ... }

    Args:
        parse_result: Parsed AST from TypeScript parser
        file_path: Source file path

    Returns:
        List of ComponentDefinition objects
    """
    components: list[ComponentDefinition] = []

    # First try to use the pre-parsed components from the TypeScript parser
    # This is the preferred method as it has more accurate component detection
    parsed_components = parse_result.get("components", [])
    for comp in parsed_components:
        comp_type = comp.get("type", "")
        if comp_type in ("function", "arrow_function"):
            comp_name = comp.get("name", "")
            if comp_name:
                # Extract props from the component data
                props = [p.get("name", "") for p in comp.get("props", [])]

                components.append(
                    ComponentDefinition(
                        id=f"{file_path}:{comp_name}",
                        name=comp_name,
                        file_path=file_path,
                        line_number=comp.get("line", 0),
                        component_type=ComponentType.FUNCTION,
                        framework="react",
                        props={prop: "any" for prop in props if prop},
                        state_variables_used=[],
                        child_components=comp.get("children", []),
                        metadata={
                            "type": f"{comp_type}_component",
                            "returns_jsx": comp.get("returns_jsx", True),
                        },
                    )
                )

    # If we found components from the parser, return them
    if components:
        return components

    # Fallback: Extract from function declarations (legacy AST-based approach)
    function_declarations = parse_result.get("function_declarations", [])

    for func in function_declarations:
        func_name = func.get("name", "")

        # Check if this looks like a component (starts with uppercase)
        if func_name and func_name[0].isupper():
            # Check if it returns JSX
            if _returns_jsx(func):
                props = _extract_props(func)
                hooks = _extract_hooks_used(func)
                state_vars = _extract_state_variable_names(func)

                components.append(
                    ComponentDefinition(
                        id=f"{file_path}:{func_name}",
                        name=func_name,
                        file_path=file_path,
                        line_number=func.get("line", 0),
                        component_type=ComponentType.FUNCTION,
                        framework="react",
                        props={prop: "any" for prop in props},
                        state_variables_used=state_vars,
                        metadata={
                            "type": "function_component",
                            "is_default_export": func.get("is_default_export", False),
                            "hooks_used": hooks,
                        },
                    )
                )

    # Extract from variable declarations (arrow functions)
    variable_declarations = parse_result.get("variable_declarations", [])

    for var_decl in variable_declarations:
        var_name = var_decl.get("name", "")

        # Check if this looks like a component
        if var_name and var_name[0].isupper():
            initializer = var_decl.get("initializer", {})

            # Check if it's an arrow function or function expression
            if initializer.get("type") in [
                "arrow_function_expression",
                "function_expression",
            ]:
                if _returns_jsx(initializer):
                    props = _extract_props(initializer)
                    hooks = _extract_hooks_used(initializer)
                    state_vars = _extract_state_variable_names(initializer)

                    components.append(
                        ComponentDefinition(
                            id=f"{file_path}:{var_name}",
                            name=var_name,
                            file_path=file_path,
                            line_number=var_decl.get("line", 0),
                            component_type=ComponentType.FUNCTION,
                            framework="react",
                            props={prop: "any" for prop in props},
                            state_variables_used=state_vars,
                            metadata={
                                "type": "arrow_function_component",
                                "is_default_export": var_decl.get("is_default_export", False),
                                "hooks_used": hooks,
                            },
                        )
                    )

    return components


def extract_class_components(parse_result: dict, file_path: Path) -> list[ComponentDefinition]:
    """
    Extract class component definitions.

    Detects:
    - class MyComponent extends React.Component { ... }
    - class MyComponent extends Component { ... }

    Args:
        parse_result: Parsed AST from TypeScript parser
        file_path: Source file path

    Returns:
        List of ComponentDefinition objects
    """
    components: list[ComponentDefinition] = []

    # First try to use the pre-parsed components from the TypeScript parser
    parsed_components = parse_result.get("components", [])
    for comp in parsed_components:
        if comp.get("type") == "class":
            comp_name = comp.get("name", "")
            if comp_name:
                props = [p.get("name", "") for p in comp.get("props", [])]

                components.append(
                    ComponentDefinition(
                        id=f"{file_path}:{comp_name}",
                        name=comp_name,
                        file_path=file_path,
                        line_number=comp.get("line", 0),
                        component_type=ComponentType.CLASS,
                        framework="react",
                        props={prop: "any" for prop in props if prop},
                        state_variables_used=[],
                        child_components=comp.get("children", []),
                        metadata={
                            "type": "class_component",
                            "extends": comp.get("extends"),
                        },
                    )
                )

    # If we found components from the parser, return them
    if components:
        return components

    # Fallback: Extract from class declarations (legacy AST-based approach)
    class_declarations = parse_result.get("class_declarations", [])

    for cls in class_declarations:
        # Check if it extends React.Component or Component
        superclass = cls.get("superclass", {})
        superclass_name = _get_superclass_name(superclass)

        if superclass_name in [
            "React.Component",
            "Component",
            "React.PureComponent",
            "PureComponent",
        ]:
            class_name = cls.get("name", "")

            # Extract props from constructor or propTypes
            props = _extract_class_props(cls)

            # Extract state from constructor
            state_vars = _extract_class_state(cls)

            # Find lifecycle methods
            lifecycle_methods = _extract_lifecycle_methods(cls)

            components.append(
                ComponentDefinition(
                    id=f"{file_path}:{class_name}",
                    name=class_name,
                    file_path=file_path,
                    line_number=cls.get("line", 0),
                    component_type=ComponentType.CLASS,
                    framework="react",
                    props={prop: "any" for prop in props},
                    state_variables_used=state_vars,
                    metadata={
                        "type": "class_component",
                        "is_default_export": cls.get("is_default_export", False),
                        "lifecycle_methods": lifecycle_methods,
                        "is_pure": "Pure" in superclass_name,
                    },
                )
            )

    return components


def build_component_tree(components: list[ComponentDefinition], parse_result: dict) -> None:
    """
    Build parent-child relationships from JSX.

    This function mutates the components list by populating the
    children_components field based on which components are rendered
    inside others.

    Args:
        components: List of ComponentDefinition objects to update
        parse_result: Parsed AST with JSX information
    """
    # Build a map of component names for quick lookup
    component_map = {comp.name: comp for comp in components}

    # For each component, find which components it renders
    for component in components:
        # Get all JSX elements rendered by this component
        jsx_elements = parse_result.get("jsx_elements_by_component", {}).get(component.name, [])

        for jsx_elem in jsx_elements:
            element_name = _get_jsx_element_name(jsx_elem)

            # Check if this element is a component (starts with uppercase)
            if element_name and element_name[0].isupper():
                # Check if we have this component in our list
                if element_name in component_map:
                    if element_name not in component.child_components:
                        component.child_components.append(element_name)


def _returns_jsx(func_node: dict) -> bool:
    """
    Check if a function returns JSX.

    Args:
        func_node: Function AST node

    Returns:
        True if the function returns JSX
    """
    # Look for return statements with JSX
    return_statements = func_node.get("return_statements", [])

    for ret in return_statements:
        argument = ret.get("argument", {})
        if _is_jsx_node(argument):
            return True

    return False


def _is_jsx_node(node: dict) -> bool:
    """
    Check if an AST node represents JSX.

    Args:
        node: AST node

    Returns:
        True if the node is JSX
    """
    node_type = node.get("type", "")
    return node_type in [
        "jsx_element",
        "jsx_self_closing_element",
        "jsx_fragment",
    ]


def _extract_props(func_node: dict) -> list[str]:
    """
    Extract prop names from function parameters.

    Args:
        func_node: Function AST node

    Returns:
        List of prop names
    """
    props: list[str] = []

    params = func_node.get("parameters", [])

    if not params:
        return props

    # The first parameter is usually props
    first_param = params[0]
    param_type = first_param.get("type", "")

    if param_type == "identifier":
        # Simple parameter: function MyComponent(props) { ... }
        # We can't extract individual prop names without type info
        return ["props"]

    elif param_type == "object_pattern":
        # Destructured props: function MyComponent({ name, age }) { ... }
        properties = first_param.get("properties", [])

        for prop in properties:
            prop_name = prop.get("key", {}).get("name", "")
            if prop_name:
                props.append(prop_name)

    return props


def _extract_hooks_used(func_node: dict) -> list[str]:
    """
    Extract names of hooks used in a component.

    Args:
        func_node: Function AST node

    Returns:
        List of hook names
    """
    hooks: list[str] = []

    call_expressions = func_node.get("call_expressions", [])

    for call in call_expressions:
        callee = call.get("callee", {})
        callee_name = callee.get("name", "")

        # Check if it's a hook (starts with 'use')
        if callee_name.startswith("use"):
            hooks.append(callee_name)

    return list(set(hooks))  # Remove duplicates


def _extract_state_variable_names(func_node: dict) -> list[str]:
    """
    Extract state variable names from a function component.

    Args:
        func_node: Function AST node

    Returns:
        List of state variable names
    """
    state_vars: list[str] = []

    # Find useState calls
    call_expressions = func_node.get("call_expressions", [])

    for call in call_expressions:
        callee = call.get("callee", {})

        if callee.get("name") in ["useState", "useReducer"]:
            # Look for the variable declaration
            parent = call.get("parent", {})
            if parent.get("type") == "variable_declarator":
                pattern = parent.get("pattern", {})
                if pattern.get("type") == "array_pattern":
                    elements = pattern.get("elements", [])
                    if elements:
                        state_name = elements[0].get("name", "")
                        if state_name:
                            state_vars.append(state_name)

    return state_vars


def _get_superclass_name(superclass_node: dict) -> str:
    """
    Get the name of the superclass.

    Args:
        superclass_node: Superclass AST node

    Returns:
        Superclass name
    """
    node_type = superclass_node.get("type", "")

    if node_type == "identifier":
        name = superclass_node.get("name", "")
        return str(name) if name is not None else ""
    elif node_type == "member_expression":
        obj = superclass_node.get("object", {}).get("name", "")
        prop = superclass_node.get("property", {}).get("name", "")
        obj_str = str(obj) if obj is not None else ""
        prop_str = str(prop) if prop is not None else ""
        return f"{obj_str}.{prop_str}" if obj_str and prop_str else ""
    else:
        return ""


def _extract_class_props(class_node: dict) -> list[str]:
    """
    Extract prop names from a class component.

    Args:
        class_node: Class AST node

    Returns:
        List of prop names
    """
    # For class components, we'd need to look at:
    # 1. propTypes static property
    # 2. this.props.* usage in methods
    # 3. TypeScript interface/type definitions

    # Simplified: just return ["props"] for now
    # A full implementation would parse propTypes or TypeScript types
    return ["props"]


def _extract_class_state(class_node: dict) -> list[str]:
    """
    Extract state variable names from a class component.

    Args:
        class_node: Class AST node

    Returns:
        List of state variable names
    """
    state_vars: list[str] = []

    # Look for constructor with state initialization
    methods = class_node.get("methods", [])

    for method in methods:
        if method.get("name") == "constructor":
            # Look for this.state = { ... }
            body = method.get("body", {})
            assignments = body.get("assignments", [])

            for assignment in assignments:
                left = assignment.get("left", {})

                # Check if it's this.state
                if left.get("type") == "member_expression":
                    obj = left.get("object", {})
                    prop = left.get("property", {})

                    if obj.get("type") == "this" and prop.get("name") == "state":
                        # Extract property names from the state object
                        right = assignment.get("right", {})
                        if right.get("type") == "object_expression":
                            properties = right.get("properties", [])
                            for prop_node in properties:
                                key = prop_node.get("key", {})
                                state_vars.append(key.get("name", ""))

    return state_vars


def _extract_lifecycle_methods(class_node: dict) -> list[str]:
    """
    Extract lifecycle method names from a class component.

    Args:
        class_node: Class AST node

    Returns:
        List of lifecycle method names
    """
    lifecycle_methods = [
        "componentDidMount",
        "componentDidUpdate",
        "componentWillUnmount",
        "shouldComponentUpdate",
        "getDerivedStateFromProps",
        "getSnapshotBeforeUpdate",
        "componentDidCatch",
    ]

    methods = class_node.get("methods", [])
    found_methods = []

    for method in methods:
        method_name = method.get("name", "")
        if method_name in lifecycle_methods:
            found_methods.append(method_name)

    return found_methods


def _get_jsx_element_name(jsx_node: dict) -> str:
    """
    Get the name of a JSX element.

    Args:
        jsx_node: JSX element AST node

    Returns:
        Element name
    """
    node_type = jsx_node.get("type", "")

    if node_type == "jsx_element":
        opening = jsx_node.get("openingElement", {})
        name_node = opening.get("name", {})
        return _get_jsx_name(name_node)
    elif node_type == "jsx_self_closing_element":
        name_node = jsx_node.get("name", {})
        return _get_jsx_name(name_node)
    else:
        return ""


def _get_jsx_name(name_node: dict) -> str:
    """
    Extract the name from a JSX name node.

    Args:
        name_node: JSX name AST node

    Returns:
        Element name
    """
    node_type = name_node.get("type", "")

    if node_type == "jsx_identifier":
        name = name_node.get("name", "")
        return str(name) if name is not None else ""
    elif node_type == "jsx_member_expression":
        obj = _get_jsx_name(name_node.get("object", {}))
        prop = _get_jsx_name(name_node.get("property", {}))
        return f"{obj}.{prop}" if obj and prop else ""
    else:
        return ""


def classify_component(component: ComponentDefinition) -> ComponentCategory:
    """
    Classify a component as either a STATE (page-level) or WIDGET (UI element).

    Classification heuristics:
    1. Page-level components (STATE):
       - Located in app/*/page.tsx or pages/*.tsx (Next.js routes)
       - Have names like *Page, *Screen, *View
       - Are default exports from route files
       - Have an associated route_path

    2. UI components (WIDGET):
       - Located in components/ directories
       - Have names like Button, Card, Input, Nav*, *Icon, *Link
       - Are reusable UI elements
       - Don't have associated routes

    Args:
        component: ComponentDefinition to classify

    Returns:
        ComponentCategory.STATE or ComponentCategory.WIDGET
    """
    file_path_str = str(component.file_path)
    name = component.name

    # 1. Check if component has an associated route - strong indicator of STATE
    if component.route_path is not None:
        return ComponentCategory.STATE

    # 2. Check file path for Next.js route patterns
    # Next.js App Router: app/*/page.tsx
    if "/app/" in file_path_str and (
        file_path_str.endswith("/page.tsx")
        or file_path_str.endswith("/page.ts")
        or file_path_str.endswith("/page.jsx")
        or file_path_str.endswith("/page.js")
    ):
        return ComponentCategory.STATE

    # Next.js Pages Router: pages/*.tsx (but not _app, _document, _error)
    if "/pages/" in file_path_str and not any(
        special in file_path_str for special in ["/_app.", "/_document.", "/_error."]
    ):
        # Exclude components in pages/components or pages/api
        if "/components/" not in file_path_str and "/api/" not in file_path_str:
            return ComponentCategory.STATE

    # 3. Check component name patterns for page-level components
    page_suffixes = ["Page", "Screen", "View", "Route", "Layout"]
    if any(name.endswith(suffix) for suffix in page_suffixes):
        # Additional check: not in components directory
        if "/components/" not in file_path_str:
            return ComponentCategory.STATE

    # 4. Check if component is in a components directory - strong indicator of WIDGET
    if "/components/" in file_path_str or "/ui/" in file_path_str:
        return ComponentCategory.WIDGET

    # 5. Check component name patterns for UI widgets
    widget_prefixes = ["Button", "Input", "Card", "Modal", "Dialog", "Form", "Nav", "Menu"]
    widget_suffixes = ["Button", "Input", "Icon", "Link", "Card", "Modal", "Dialog", "Menu"]
    widget_keywords = ["Header", "Footer", "Sidebar", "Navbar", "Toggle", "Switch", "Slider"]

    if any(name.startswith(prefix) for prefix in widget_prefixes):
        return ComponentCategory.WIDGET
    if any(name.endswith(suffix) for suffix in widget_suffixes):
        return ComponentCategory.WIDGET
    if any(keyword in name for keyword in widget_keywords):
        return ComponentCategory.WIDGET

    # 6. Default heuristic: if it's in src/app or src/pages and not explicitly a widget, likely STATE
    if (
        "/app/" in file_path_str or "/pages/" in file_path_str
    ) and "/components/" not in file_path_str:
        return ComponentCategory.STATE

    # 7. Default to WIDGET (most components are reusable UI elements)
    return ComponentCategory.WIDGET


def classify_components(components: list[ComponentDefinition]) -> None:
    """
    Classify all components in place, updating their category field.

    This function mutates the components list by setting the category field
    on each ComponentDefinition.

    Args:
        components: List of ComponentDefinition objects to classify
    """
    for component in components:
        component.category = classify_component(component)
