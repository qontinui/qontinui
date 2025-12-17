"""String operation executor for text manipulation and processing.

This module provides the StringExecutor class for handling string operations
including concatenation, substring extraction, replacement, splitting, case
conversion, pattern matching, and JSON parsing.

Example:
    >>> from qontinui.actions.data_operations import VariableContext
    >>> from qontinui.actions.data_operations.string_executor import StringExecutor
    >>>
    >>> context = VariableContext()
    >>> context.set("name", "John")
    >>> executor = StringExecutor(context)
    >>>
    >>> # Concatenate strings
    >>> result = executor.concat("Hello ", ["World", "!"])
    >>> print(result)  # "Hello World!"
    >>>
    >>> # Extract substring
    >>> result = executor.substring("Hello World", start=0, end=5)
    >>> print(result)  # "Hello"
    >>>
    >>> # Replace text
    >>> result = executor.replace("Hello World", search="World", replacement="Python")
    >>> print(result)  # "Hello Python"
"""

import json
import logging
import re
from typing import Any

from .context import VariableContext

logger = logging.getLogger(__name__)


class StringExecutor:
    """Executor for string manipulation operations.

    Provides methods for common string operations with support for variable
    resolution from VariableContext. All operations return string results
    (except SPLIT which returns a list).

    Supported operations:
    - CONCAT: Concatenate multiple strings
    - SUBSTRING: Extract substring by position
    - REPLACE: Replace occurrences of text
    - SPLIT: Split string into list
    - TRIM: Remove leading/trailing whitespace
    - UPPERCASE: Convert to uppercase
    - LOWERCASE: Convert to lowercase
    - MATCH: Regular expression matching
    - PARSE_JSON: Parse and validate JSON

    Attributes:
        variable_context: Context for resolving variable references
    """

    def __init__(self, variable_context: VariableContext) -> None:
        """Initialize the string executor.

        Args:
            variable_context: Context for variable resolution

        Example:
            >>> context = VariableContext()
            >>> executor = StringExecutor(context)
        """
        self.variable_context = variable_context
        logger.debug("Initialized StringExecutor")

    def execute(
        self,
        operation: str,
        input_value: str | dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> str | list[str]:
        """Execute a string operation with the given parameters.

        This is the main entry point for executing string operations. It handles
        variable resolution, operation dispatch, and result formatting.

        Args:
            operation: Operation type (CONCAT, SUBSTRING, REPLACE, etc.)
            input_value: Input string or variable reference dict
            parameters: Operation-specific parameters

        Returns:
            Result string or list (for SPLIT operation)

        Raises:
            ValueError: If operation is invalid or required parameters are missing
            TypeError: If input cannot be converted to string

        Example:
            >>> executor.execute("UPPERCASE", "hello")
            'HELLO'
            >>> executor.execute("CONCAT", "Hello", {"strings": [" ", "World"]})
            'Hello World'
        """
        # Resolve input to string
        input_str = self._resolve_input(input_value)

        # Dispatch to appropriate operation handler
        operation = operation.upper()
        logger.debug(f"Executing string operation: {operation}")

        if operation == "CONCAT":
            return self.concat(input_str, parameters)
        elif operation == "SUBSTRING":
            return self.substring(input_str, parameters)
        elif operation == "REPLACE":
            return self.replace(input_str, parameters)
        elif operation == "SPLIT":
            return self.split(input_str, parameters)
        elif operation == "TRIM":
            return self.trim(input_str)
        elif operation == "UPPERCASE":
            return self.uppercase(input_str)
        elif operation == "LOWERCASE":
            return self.lowercase(input_str)
        elif operation == "MATCH":
            return self.match(input_str, parameters)
        elif operation == "PARSE_JSON":
            return self.parse_json(input_str)
        else:
            raise ValueError(f"Unknown string operation: {operation}")

    def concat(self, input_str: str, parameters: dict[str, Any] | None = None) -> str:
        """Concatenate input string with additional strings.

        Args:
            input_str: Base string to concatenate to
            parameters: Dict with 'strings' key containing list of strings to append

        Returns:
            Concatenated string

        Example:
            >>> executor.concat("Hello", {"strings": [" ", "World", "!"]})
            'Hello World!'
            >>> executor.concat("Hello", None)  # No additional strings
            'Hello'
        """
        if not parameters or "strings" not in parameters:
            logger.debug("CONCAT: No additional strings provided")
            return input_str

        additional_strings = parameters["strings"]
        if not isinstance(additional_strings, list):
            raise TypeError("CONCAT 'strings' parameter must be a list")

        result = input_str + "".join(str(s) for s in additional_strings)
        logger.debug(f"CONCAT: Combined {len(additional_strings) + 1} strings")
        return result

    def substring(
        self, input_str: str, parameters: dict[str, Any] | None = None
    ) -> str:
        """Extract substring from input string.

        Args:
            input_str: String to extract from
            parameters: Dict with 'start' (int) and optional 'end' (int) keys

        Returns:
            Extracted substring

        Example:
            >>> executor.substring("Hello World", {"start": 0, "end": 5})
            'Hello'
            >>> executor.substring("Hello World", {"start": 6})
            'World'
        """
        if not parameters:
            return input_str

        start = parameters.get("start", 0)
        end = parameters.get("end")

        if not isinstance(start, int):
            raise TypeError(f"SUBSTRING 'start' must be an integer, got {type(start)}")
        if end is not None and not isinstance(end, int):
            raise TypeError(f"SUBSTRING 'end' must be an integer, got {type(end)}")

        result = input_str[start:end] if end is not None else input_str[start:]
        logger.debug(f"SUBSTRING: Extracted from position {start} to {end or 'end'}")
        return result

    def replace(self, input_str: str, parameters: dict[str, Any] | None = None) -> str:
        """Replace occurrences of search string with replacement.

        Args:
            input_str: String to perform replacement on
            parameters: Dict with 'search' (str) and 'replacement' (str) keys

        Returns:
            String with replacements applied

        Raises:
            ValueError: If 'search' parameter is missing

        Example:
            >>> executor.replace("Hello World", {"search": "World", "replacement": "Python"})
            'Hello Python'
            >>> executor.replace("foo foo", {"search": "foo", "replacement": "bar"})
            'bar bar'
        """
        if not parameters or "search" not in parameters:
            raise ValueError("REPLACE operation requires 'search' parameter")

        search = str(parameters["search"])
        replacement = str(parameters.get("replacement", ""))

        result = input_str.replace(search, replacement)
        logger.debug(f"REPLACE: Replaced '{search}' with '{replacement}'")
        return result

    def split(
        self, input_str: str, parameters: dict[str, Any] | None = None
    ) -> list[str]:
        """Split string into a list of substrings.

        Args:
            input_str: String to split
            parameters: Dict with optional 'delimiter' (str) key

        Returns:
            List of string parts

        Example:
            >>> executor.split("a,b,c", {"delimiter": ","})
            ['a', 'b', 'c']
            >>> executor.split("hello world")  # Splits on whitespace
            ['hello', 'world']
        """
        delimiter = " "  # Default delimiter
        if parameters and "delimiter" in parameters:
            delimiter = str(parameters["delimiter"])

        result = input_str.split(delimiter)
        logger.debug(
            f"SPLIT: Split into {len(result)} parts using delimiter '{delimiter}'"
        )
        return result

    def trim(self, input_str: str) -> str:
        """Remove leading and trailing whitespace.

        Args:
            input_str: String to trim

        Returns:
            Trimmed string

        Example:
            >>> executor.trim("  hello  ")
            'hello'
            >>> executor.trim("\\n\\ttext\\r\\n")
            'text'
        """
        result = input_str.strip()
        logger.debug("TRIM: Removed leading/trailing whitespace")
        return result

    def uppercase(self, input_str: str) -> str:
        """Convert string to uppercase.

        Args:
            input_str: String to convert

        Returns:
            Uppercase string

        Example:
            >>> executor.uppercase("hello")
            'HELLO'
        """
        result = input_str.upper()
        logger.debug("UPPERCASE: Converted to uppercase")
        return result

    def lowercase(self, input_str: str) -> str:
        """Convert string to lowercase.

        Args:
            input_str: String to convert

        Returns:
            Lowercase string

        Example:
            >>> executor.lowercase("HELLO")
            'hello'
        """
        result = input_str.lower()
        logger.debug("LOWERCASE: Converted to lowercase")
        return result

    def match(self, input_str: str, parameters: dict[str, Any] | None = None) -> str:
        """Match string against regular expression pattern.

        Returns a JSON string containing match information including captured
        groups and named groups.

        Args:
            input_str: String to match against
            parameters: Dict with 'pattern' (str) key containing regex pattern

        Returns:
            JSON string with match results:
            - {"matched": true, "groups": [...], "group_dict": {...}} if matched
            - {"matched": false} if no match

        Raises:
            ValueError: If 'pattern' parameter is missing
            re.error: If pattern is invalid regex

        Example:
            >>> result = executor.match("test123", {"pattern": r"(\\w+)(\\d+)"})
            >>> json.loads(result)
            {'matched': True, 'groups': ['test', '123'], 'group_dict': {}}
        """
        if not parameters or "pattern" not in parameters:
            raise ValueError("MATCH operation requires 'pattern' parameter")

        pattern = str(parameters["pattern"])

        try:
            match_obj = re.search(pattern, input_str)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        if match_obj:
            result = json.dumps(
                {
                    "matched": True,
                    "groups": list(match_obj.groups()),
                    "group_dict": match_obj.groupdict(),
                }
            )
            logger.debug(
                f"MATCH: Pattern matched with {len(match_obj.groups())} groups"
            )
        else:
            result = json.dumps({"matched": False})
            logger.debug("MATCH: Pattern did not match")

        return result

    def parse_json(self, input_str: str) -> str:
        """Parse and validate JSON string.

        Parses the input JSON and returns it in normalized form. This validates
        JSON syntax and reformats it.

        Args:
            input_str: JSON string to parse

        Returns:
            Normalized JSON string

        Raises:
            ValueError: If input is not valid JSON

        Example:
            >>> executor.parse_json('{"name": "John", "age": 30}')
            '{"name": "John", "age": 30}'
            >>> executor.parse_json('[1, 2, 3]')
            '[1, 2, 3]'
        """
        try:
            parsed = json.loads(input_str)
            result = json.dumps(parsed)
            logger.debug("PARSE_JSON: Successfully parsed and normalized JSON")
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    def _resolve_input(self, input_value: str | dict[str, Any]) -> str:
        """Resolve input value to a string.

        If input is a dict with 'variableName', resolves it from context.
        Otherwise, converts to string.

        Args:
            input_value: Input value or variable reference

        Returns:
            Resolved string value

        Raises:
            ValueError: If variable reference is invalid or not found
            TypeError: If value cannot be converted to string
        """
        if isinstance(input_value, str):
            return input_value

        if isinstance(input_value, dict):
            # Variable reference
            var_name = input_value.get("variableName")
            if not var_name:
                raise ValueError("Variable reference requires 'variableName' field")

            value = self.variable_context.get(var_name)
            if value is None:
                raise ValueError(f"Variable '{var_name}' not found in context")

            logger.debug(f"Resolved variable '{var_name}' to value")
            return str(value)

        # Try to convert to string
        try:
            return str(input_value)
        except Exception as e:
            raise TypeError(f"Cannot convert input to string: {e}") from e
