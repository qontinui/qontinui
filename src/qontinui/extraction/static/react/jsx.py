"""
JSX conditional rendering extraction utilities.

This module provides functions to extract conditional rendering patterns from JSX:
- Logical AND: {condition && <Component />}
- Ternary: {condition ? <A /> : <B />}
- Early returns: if (condition) return <Component />
- Switch statements
"""

from pathlib import Path

from qontinui.extraction.static.models import ConditionalRender


def extract_logical_and(
    parse_result: dict, component_name: str, file_path: Path
) -> list[ConditionalRender]:
    """
    Extract {condition && <Component />} patterns.

    This is the most common conditional rendering pattern in React.

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of ConditionalRender objects for logical AND patterns
    """
    conditionals: list[ConditionalRender] = []

    # Find all logical expressions within JSX
    jsx_expressions = parse_result.get("jsx_expressions", [])

    for expr in jsx_expressions:
        if expr.get("type") == "logical_expression" and expr.get("operator") == "&&":
            condition = _serialize_expression(expr.get("left", {}))
            true_branch = _serialize_jsx_element(expr.get("right", {}))

            controlling_vars = _extract_variables_from_condition(expr.get("left", {}))

            conditionals.append(
                ConditionalRender(
                    condition=condition,
                    render_type="logical_and",
                    component=component_name,
                    file_path=file_path,
                    true_branch=true_branch,
                    false_branch="null",
                    controlling_variables=controlling_vars,
                    line_number=expr.get("line", 0),
                )
            )

    return conditionals


def extract_ternary(
    parse_result: dict, component_name: str, file_path: Path
) -> list[ConditionalRender]:
    """
    Extract {condition ? <A /> : <B />} patterns.

    Used when you need to render different components based on a condition.

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of ConditionalRender objects for ternary patterns
    """
    conditionals: list[ConditionalRender] = []

    jsx_expressions = parse_result.get("jsx_expressions", [])

    for expr in jsx_expressions:
        if expr.get("type") == "conditional_expression":
            condition = _serialize_expression(expr.get("test", {}))
            true_branch = _serialize_jsx_element(expr.get("consequent", {}))
            false_branch = _serialize_jsx_element(expr.get("alternate", {}))

            controlling_vars = _extract_variables_from_condition(expr.get("test", {}))

            conditionals.append(
                ConditionalRender(
                    condition=condition,
                    render_type="ternary",
                    component=component_name,
                    file_path=file_path,
                    true_branch=true_branch,
                    false_branch=false_branch,
                    controlling_variables=controlling_vars,
                    line_number=expr.get("line", 0),
                )
            )

    return conditionals


def extract_early_returns(
    parse_result: dict, component_name: str, file_path: Path
) -> list[ConditionalRender]:
    """
    Extract if (condition) return <Component /> patterns.

    Common pattern for guard clauses and loading states.

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of ConditionalRender objects for early return patterns
    """
    conditionals: list[ConditionalRender] = []

    # Find all if statements with return statements
    if_statements = parse_result.get("if_statements", [])

    for if_stmt in if_statements:
        # Check if the consequent is a return statement
        consequent = if_stmt.get("consequent", {})

        if consequent.get("type") == "return_statement":
            condition = _serialize_expression(if_stmt.get("test", {}))
            true_branch = _serialize_jsx_element(consequent.get("argument", {}))

            controlling_vars = _extract_variables_from_condition(
                if_stmt.get("test", {})
            )

            conditionals.append(
                ConditionalRender(
                    condition=condition,
                    render_type="early_return",
                    component=component_name,
                    file_path=file_path,
                    true_branch=true_branch,
                    false_branch="continue",
                    controlling_variables=controlling_vars,
                    line_number=if_stmt.get("line", 0),
                )
            )
        elif consequent.get("type") == "block_statement":
            # Check if the block contains only a return statement
            body = consequent.get("body", [])
            if len(body) == 1 and body[0].get("type") == "return_statement":
                condition = _serialize_expression(if_stmt.get("test", {}))
                true_branch = _serialize_jsx_element(body[0].get("argument", {}))

                controlling_vars = _extract_variables_from_condition(
                    if_stmt.get("test", {})
                )

                conditionals.append(
                    ConditionalRender(
                        condition=condition,
                        render_type="early_return",
                        component=component_name,
                        file_path=file_path,
                        true_branch=true_branch,
                        false_branch="continue",
                        controlling_variables=controlling_vars,
                        line_number=if_stmt.get("line", 0),
                    )
                )

    return conditionals


def extract_switch_render(
    parse_result: dict, component_name: str, file_path: Path
) -> list[ConditionalRender]:
    """
    Extract switch statement rendering patterns.

    Common for rendering different states (loading, error, success, etc.).

    Args:
        parse_result: Parsed AST from TypeScript parser
        component_name: Name of the component
        file_path: Source file path

    Returns:
        List of ConditionalRender objects for switch patterns
    """
    conditionals: list[ConditionalRender] = []

    switch_statements = parse_result.get("switch_statements", [])

    for switch_stmt in switch_statements:
        discriminant = _serialize_expression(switch_stmt.get("discriminant", {}))
        controlling_vars = _extract_variables_from_condition(
            switch_stmt.get("discriminant", {})
        )

        cases = switch_stmt.get("cases", [])

        for case in cases:
            # Check if this case returns JSX
            consequent = case.get("consequent", [])

            for stmt in consequent:
                if stmt.get("type") == "return_statement":
                    test = case.get("test", {})
                    case_value = _serialize_expression(test) if test else "default"

                    condition = f"{discriminant} === {case_value}"
                    true_branch = _serialize_jsx_element(stmt.get("argument", {}))

                    conditionals.append(
                        ConditionalRender(
                            condition=condition,
                            render_type="switch",
                            component=component_name,
                            file_path=file_path,
                            true_branch=true_branch,
                            false_branch="other cases",
                            controlling_variables=controlling_vars,
                            line_number=case.get("line", 0),
                            metadata={"discriminant": discriminant, "case": case_value},
                        )
                    )

    return conditionals


def _serialize_expression(expr_node: dict) -> str:
    """
    Serialize an expression AST node to a string.

    Args:
        expr_node: AST node representing an expression

    Returns:
        String representation of the expression
    """
    node_type = expr_node.get("type", "")

    if node_type == "identifier":
        return expr_node.get("name", "")
    elif node_type == "member_expression":
        obj = _serialize_expression(expr_node.get("object", {}))
        prop = _serialize_expression(expr_node.get("property", {}))
        return f"{obj}.{prop}"
    elif node_type == "literal":
        value = expr_node.get("value", "")
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)
    elif node_type == "unary_expression":
        operator = expr_node.get("operator", "")
        argument = _serialize_expression(expr_node.get("argument", {}))
        return f"{operator}{argument}"
    elif node_type == "binary_expression":
        left = _serialize_expression(expr_node.get("left", {}))
        operator = expr_node.get("operator", "")
        right = _serialize_expression(expr_node.get("right", {}))
        return f"{left} {operator} {right}"
    elif node_type == "logical_expression":
        left = _serialize_expression(expr_node.get("left", {}))
        operator = expr_node.get("operator", "")
        right = _serialize_expression(expr_node.get("right", {}))
        return f"{left} {operator} {right}"
    elif node_type == "call_expression":
        callee = _serialize_expression(expr_node.get("callee", {}))
        return f"{callee}(...)"
    else:
        return f"<{node_type}>"


def _serialize_jsx_element(jsx_node: dict) -> str:
    """
    Serialize a JSX element to a string.

    Args:
        jsx_node: AST node representing JSX

    Returns:
        String representation of the JSX element
    """
    node_type = jsx_node.get("type", "")

    if node_type == "jsx_element":
        opening = jsx_node.get("openingElement", {})
        name = _get_jsx_element_name(opening.get("name", {}))
        return f"<{name} />"
    elif node_type == "jsx_self_closing_element":
        name = _get_jsx_element_name(jsx_node.get("name", {}))
        return f"<{name} />"
    elif node_type == "jsx_fragment":
        return "<>...</>"
    elif node_type == "literal":
        return str(jsx_node.get("value", ""))
    elif node_type == "null":
        return "null"
    else:
        return f"<{node_type}>"


def _get_jsx_element_name(name_node: dict) -> str:
    """
    Extract the name from a JSX element name node.

    Args:
        name_node: JSX name AST node

    Returns:
        Element name as string
    """
    node_type = name_node.get("type", "")

    if node_type == "jsx_identifier":
        return name_node.get("name", "")
    elif node_type == "jsx_member_expression":
        obj = _get_jsx_element_name(name_node.get("object", {}))
        prop = _get_jsx_element_name(name_node.get("property", {}))
        return f"{obj}.{prop}"
    else:
        return ""


def _extract_variables_from_condition(condition_node: dict) -> list[str]:
    """
    Extract variable names referenced in a condition.

    Args:
        condition_node: AST node representing the condition

    Returns:
        List of variable names
    """
    variables: list[str] = []

    def visit_node(node: dict):
        node_type = node.get("type", "")

        if node_type == "identifier":
            name = node.get("name", "")
            # Filter out common non-state identifiers
            if name and name not in ["true", "false", "null", "undefined"]:
                variables.append(name)
        elif node_type == "member_expression":
            # Get the base object
            visit_node(node.get("object", {}))
        elif node_type in ["binary_expression", "logical_expression"]:
            visit_node(node.get("left", {}))
            visit_node(node.get("right", {}))
        elif node_type == "unary_expression":
            visit_node(node.get("argument", {}))
        elif node_type == "call_expression":
            visit_node(node.get("callee", {}))

    visit_node(condition_node)
    return list(set(variables))  # Remove duplicates
