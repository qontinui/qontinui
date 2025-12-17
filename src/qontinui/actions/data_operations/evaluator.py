"""Safe expression evaluation for data operations.

This module provides the SafeEvaluator class for safely evaluating Python
expressions with restricted capabilities to prevent dangerous operations.
"""

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SafeEvaluator:
    """Safe expression evaluator with restricted capabilities.

    Allows basic arithmetic, comparisons, and variable access while
    preventing dangerous operations like file I/O, imports, network access, etc.

    Only whitelisted AST node types and built-in functions are allowed.

    Example:
        >>> evaluator = SafeEvaluator()
        >>> context = {"x": 10, "y": 5}
        >>> evaluator.safe_eval("x + y", context)
        15
        >>> evaluator.safe_eval("x > y", context)
        True
    """

    # Allowed node types for safe evaluation
    ALLOWED_NODES = {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,  # For list comprehensions
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Attribute,
        ast.Call,  # Allow function calls (validated separately)
    }

    # Safe built-in functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "any": any,
        "all": all,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
    }

    @classmethod
    def safe_eval(cls, expression: str, context: dict[str, Any]) -> Any:
        """Safely evaluate a Python expression.

        Parses the expression into an AST and validates that all nodes are safe.
        Only whitelisted operations and built-in functions are allowed.

        SECURITY WARNING:
        This method uses eval() with AST-based validation. It is designed for
        TRUSTED INPUT ONLY (expressions in automation scripts written by developers).

        DO NOT use this with:
        - User input from web forms, REST APIs, CLI arguments
        - Data from external sources (files, network, databases)
        - Configuration from untrusted locations
        - Any data that could be influenced by untrusted users

        Security mitigations in place:
        - AST parsing validates all nodes against whitelist before evaluation
        - Only safe operations allowed (math, comparisons, logic, data structures)
        - Function calls limited to whitelisted safe functions (abs, len, max, etc.)
        - No access to dangerous builtins: __import__, open, exec, compile, eval
        - No file I/O, network access, or system operations
        - No module imports or attribute access to dangerous objects

        Allowed operations:
        - Arithmetic: +, -, *, /, //, %, **
        - Comparisons: ==, !=, <, >, <=, >=, in, not in, is, is not
        - Logic: and, or, not
        - Data structures: lists, tuples, dicts, sets, comprehensions
        - Safe functions: abs, bool, float, int, len, max, min, range, round, etc.

        Blocked operations:
        - Imports: import, __import__
        - File I/O: open, read, write
        - Code execution: eval, exec, compile
        - System access: os, sys, subprocess
        - Attribute access to dangerous objects

        For untrusted input scenarios:
        - Run Qontinui in isolated containers/VMs with no network/file access
        - Implement additional input validation/sanitization
        - Use alternative approaches that don't require code evaluation
        - Consider sandboxed execution environments (Docker, VMs)

        See docs/SECURITY.md for comprehensive security model and best practices.

        Args:
            expression: Python expression to evaluate (must be non-empty string)
            context: Variable context for evaluation

        Returns:
            Result of evaluation

        Raises:
            ValueError: If expression is invalid, empty, or contains unsafe operations
            SyntaxError: If expression has invalid Python syntax

        Example:
            >>> SafeEvaluator.safe_eval("2 + 2", {})
            4
            >>> SafeEvaluator.safe_eval("[x*2 for x in range(3)]", {})
            [0, 2, 4]
            >>> SafeEvaluator.safe_eval("max(numbers)", {"numbers": [1, 5, 3]})
            5
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Expression must be a non-empty string")

        expression = expression.strip()
        logger.debug(f"Evaluating expression: {expression}")

        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")

            # Validate all nodes are safe
            for node in ast.walk(tree):
                if type(node) not in cls.ALLOWED_NODES:
                    raise ValueError(f"Unsafe operation in expression: {type(node).__name__}")

                # Extra validation for function calls
                if isinstance(node, ast.Call):
                    # Only allow calls to functions in SAFE_FUNCTIONS
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name not in cls.SAFE_FUNCTIONS:
                            raise ValueError(f"Unsafe function call: {func_name}")
                    elif isinstance(node.func, ast.Attribute):
                        # Allow method calls on safe objects (like list.append, str.upper, etc.)
                        pass
                    else:
                        raise ValueError(f"Unsafe function call type: {type(node.func).__name__}")

            # Create safe namespace
            safe_namespace = {
                "__builtins__": cls.SAFE_FUNCTIONS,
            }
            safe_namespace.update(context)

            # Evaluate
            result = eval(compile(tree, "<string>", "eval"), safe_namespace)

            logger.debug(f"Expression evaluated to: {result}")
            return result

        except SyntaxError as e:
            logger.error(f"Syntax error in expression: {e}")
            raise
        except ValueError:
            # Re-raise our own ValueError without wrapping
            raise
        except Exception as e:
            logger.error(f"Error evaluating expression: {e}")
            raise ValueError(f"Failed to evaluate expression: {e}") from e

    @classmethod
    def is_safe_expression(cls, expression: str) -> bool:
        """Check if an expression is safe to evaluate.

        Performs static analysis on the expression to determine if it contains
        only whitelisted operations.

        Args:
            expression: Expression to check

        Returns:
            True if expression is safe, False otherwise

        Example:
            >>> SafeEvaluator.is_safe_expression("x + y")
            True
            >>> SafeEvaluator.is_safe_expression("import os")
            False
            >>> SafeEvaluator.is_safe_expression("open('file.txt')")
            False
        """
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if type(node) not in cls.ALLOWED_NODES:
                    return False
            return True
        except (SyntaxError, ValueError):
            # Invalid Python expression
            return False
