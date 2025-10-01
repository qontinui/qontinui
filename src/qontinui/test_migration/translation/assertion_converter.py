"""
Assertion converter for mapping JUnit assertions to pytest assertions.
"""

import re
from typing import cast


class AssertionConverter:
    """
    Converts JUnit assertions to pytest assertions.

    Handles common assertion patterns including:
    - Basic assertions (assertTrue, assertFalse, assertEquals, etc.)
    - Custom assertion methods and error messages
    - Complex assertion patterns with multiple parameters
    """

    def __init__(self):
        """Initialize the assertion converter with mapping dictionaries."""
        self._basic_assertion_mappings = {
            "assertTrue": self._convert_assert_true,
            "assertFalse": self._convert_assert_false,
            "assertEquals": self._convert_assert_equals,
            "assertNotEquals": self._convert_assert_not_equals,
            "assertNull": self._convert_assert_null,
            "assertNotNull": self._convert_assert_not_null,
            "assertSame": self._convert_assert_same,
            "assertNotSame": self._convert_assert_not_same,
            "assertThat": self._convert_assert_that,
            "assertThrows": self._convert_assert_throws,
            "assertDoesNotThrow": self._convert_assert_does_not_throw,
            "assertArrayEquals": self._convert_assert_array_equals,
            "assertIterableEquals": self._convert_assert_iterable_equals,
            "assertLinesMatch": self._convert_assert_lines_match,
            "assertTimeout": self._convert_assert_timeout,
            "assertTimeoutPreemptively": self._convert_assert_timeout_preemptively,
            "fail": self._convert_fail,
        }

        # Hamcrest matchers commonly used with assertThat
        self._hamcrest_matchers = {
            "is": "is",
            "equalTo": "==",
            "not": "not",
            "nullValue": "is None",
            "notNullValue": "is not None",
            "instanceOf": "isinstance",
            "hasSize": "len",
            "empty": "len({}) == 0",
            "contains": "in",
            "containsString": "in",
            "startsWith": "startswith",
            "endsWith": "endswith",
            "greaterThan": ">",
            "lessThan": "<",
            "greaterThanOrEqualTo": ">=",
            "lessThanOrEqualTo": "<=",
        }

    def convert_assertion(self, assertion_line: str) -> str:
        """
        Convert a JUnit assertion to pytest assertion.

        Args:
            assertion_line: Java assertion code line

        Returns:
            Python assertion code
        """
        assertion_line = assertion_line.strip()

        # Remove semicolon if present
        if assertion_line.endswith(";"):
            assertion_line = assertion_line[:-1]

        # Find the assertion method
        for junit_method, converter_func in self._basic_assertion_mappings.items():
            if junit_method in assertion_line:
                return cast(str, converter_func(assertion_line))

        # If no specific assertion found, return as-is
        return assertion_line

    def convert_multiple_assertions(self, assertion_lines: list[str]) -> list[str]:
        """
        Convert multiple assertion lines.

        Args:
            assertion_lines: List of Java assertion lines

        Returns:
            List of Python assertion lines
        """
        return [self.convert_assertion(line) for line in assertion_lines]

    def _extract_assertion_params(
        self, assertion_line: str, method_name: str
    ) -> tuple[list[str], str | None]:
        """
        Extract parameters and optional message from assertion.

        Args:
            assertion_line: The assertion line
            method_name: The assertion method name

        Returns:
            Tuple of (parameters, optional_message)
        """
        # Find the method call pattern - need to handle nested parentheses properly
        start_idx = assertion_line.find(method_name + "(")
        if start_idx == -1:
            return [], None

        # Find the matching closing parenthesis
        open_paren_idx = start_idx + len(method_name)
        paren_count = 0
        end_idx = -1

        for i in range(open_paren_idx, len(assertion_line)):
            if assertion_line[i] == "(":
                paren_count += 1
            elif assertion_line[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return [], None

        # Extract parameters string
        params_str = assertion_line[open_paren_idx + 1 : end_idx]

        if not params_str.strip():
            return [], None

        # Split parameters, handling nested parentheses and quotes
        param_parts = self._split_parameters(params_str)

        # Check if last parameter is a string message
        if len(param_parts) > 1 and self._is_string_literal(param_parts[-1]):
            message = param_parts[-1]
            params = param_parts[:-1]
        else:
            params = param_parts
            message = None

        return params, message

    def _split_parameters(self, params_str: str) -> list[str]:
        """Split parameter string handling nested structures."""
        params = []
        current_param = ""
        paren_count = 0
        quote_char = None

        for char in params_str:
            if quote_char:
                current_param += char
                if char == quote_char and (not current_param.endswith("\\" + quote_char)):
                    quote_char = None
            elif char in ['"', "'"]:
                quote_char = char
                current_param += char
            elif char == "(":
                paren_count += 1
                current_param += char
            elif char == ")":
                paren_count -= 1
                current_param += char
            elif char == "," and paren_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        return params

    def _is_string_literal(self, param: str) -> bool:
        """Check if parameter is a string literal."""
        param = param.strip()
        return (param.startswith('"') and param.endswith('"')) or (
            param.startswith("'") and param.endswith("'")
        )

    def _convert_assert_true(self, assertion_line: str) -> str:
        """Convert assertTrue to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertTrue")

        if not params:
            return assertion_line

        condition = params[0]

        if message:
            return f"assert {condition}, {message}"
        else:
            return f"assert {condition}"

    def _convert_assert_false(self, assertion_line: str) -> str:
        """Convert assertFalse to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertFalse")

        if not params:
            return assertion_line

        condition = params[0]

        if message:
            return f"assert not {condition}, {message}"
        else:
            return f"assert not {condition}"

    def _convert_assert_equals(self, assertion_line: str) -> str:
        """Convert assertEquals to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertEquals")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert {actual} == {expected}, {message}"
        else:
            return f"assert {actual} == {expected}"

    def _convert_assert_not_equals(self, assertion_line: str) -> str:
        """Convert assertNotEquals to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertNotEquals")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert {actual} != {expected}, {message}"
        else:
            return f"assert {actual} != {expected}"

    def _convert_assert_null(self, assertion_line: str) -> str:
        """Convert assertNull to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertNull")

        if not params:
            return assertion_line

        value = params[0]

        if message:
            return f"assert {value} is None, {message}"
        else:
            return f"assert {value} is None"

    def _convert_assert_not_null(self, assertion_line: str) -> str:
        """Convert assertNotNull to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertNotNull")

        if not params:
            return assertion_line

        value = params[0]

        if message:
            return f"assert {value} is not None, {message}"
        else:
            return f"assert {value} is not None"

    def _convert_assert_same(self, assertion_line: str) -> str:
        """Convert assertSame to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertSame")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert {actual} is {expected}, {message}"
        else:
            return f"assert {actual} is {expected}"

    def _convert_assert_not_same(self, assertion_line: str) -> str:
        """Convert assertNotSame to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertNotSame")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert {actual} is not {expected}, {message}"
        else:
            return f"assert {actual} is not {expected}"

    def _convert_assert_that(self, assertion_line: str) -> str:
        """Convert assertThat (Hamcrest) to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertThat")

        if len(params) < 2:
            return assertion_line

        actual = params[0]
        matcher = params[1]

        # Convert Hamcrest matcher to Python assertion
        python_assertion = self._convert_hamcrest_matcher(actual, matcher)

        if message:
            return f"assert {python_assertion}, {message}"
        else:
            return f"assert {python_assertion}"

    def _convert_hamcrest_matcher(self, actual: str, matcher: str) -> str:
        """Convert Hamcrest matcher to Python expression."""
        # Handle common Hamcrest patterns

        # is(equalTo(value)) -> actual == value
        if "is(equalTo(" in matcher:
            value_match = re.search(r"is\(equalTo\(([^)]+)\)\)", matcher)
            if value_match:
                expected = value_match.group(1)
                return f"{actual} == {expected}"

        # is(value) -> actual == value
        if matcher.startswith("is(") and matcher.endswith(")"):
            expected = matcher[3:-1]
            return f"{actual} == {expected}"

        # equalTo(value) -> actual == value
        if matcher.startswith("equalTo(") and matcher.endswith(")"):
            expected = matcher[8:-1]
            return f"{actual} == {expected}"

        # not(matcher) -> not (converted matcher)
        if matcher.startswith("not(") and matcher.endswith(")"):
            inner_matcher = matcher[4:-1]
            inner_assertion = self._convert_hamcrest_matcher(actual, inner_matcher)
            return f"not ({inner_assertion})"

        # nullValue() -> actual is None
        if matcher == "nullValue()":
            return f"{actual} is None"

        # notNullValue() -> actual is not None
        if matcher == "notNullValue()":
            return f"{actual} is not None"

        # hasSize(n) -> len(actual) == n
        if matcher.startswith("hasSize(") and matcher.endswith(")"):
            size = matcher[8:-1]
            return f"len({actual}) == {size}"

        # empty() -> len(actual) == 0
        if matcher == "empty()":
            return f"len({actual}) == 0"

        # containsString(str) -> str in actual
        if matcher.startswith("containsString(") and matcher.endswith(")"):
            substring = matcher[15:-1]
            return f"{substring} in {actual}"

        # startsWith(str) -> actual.startswith(str)
        if matcher.startswith("startsWith(") and matcher.endswith(")"):
            prefix = matcher[11:-1]
            return f"{actual}.startswith({prefix})"

        # endsWith(str) -> actual.endswith(str)
        if matcher.startswith("endsWith(") and matcher.endswith(")"):
            suffix = matcher[9:-1]
            return f"{actual}.endswith({suffix})"

        # greaterThan(value) -> actual > value
        if matcher.startswith("greaterThan(") and matcher.endswith(")"):
            value = matcher[12:-1]
            return f"{actual} > {value}"

        # lessThan(value) -> actual < value
        if matcher.startswith("lessThan(") and matcher.endswith(")"):
            value = matcher[9:-1]
            return f"{actual} < {value}"

        # instanceOf(Class.class) -> isinstance(actual, Class)
        if matcher.startswith("instanceOf(") and matcher.endswith(")"):
            class_ref = matcher[11:-1]
            if class_ref.endswith(".class"):
                class_name = class_ref[:-6]
                return f"isinstance({actual}, {class_name})"

        # Default: return as-is with actual
        return f"{actual} == {matcher}"

    def _convert_assert_throws(self, assertion_line: str) -> str:
        """Convert assertThrows to pytest.raises."""
        params, message = self._extract_assertion_params(assertion_line, "assertThrows")

        if len(params) < 2:
            return assertion_line

        exception_class = params[0]
        executable = params[1]

        # Remove .class suffix if present
        if exception_class.endswith(".class"):
            exception_class = exception_class[:-6]

        # Convert lambda or method reference to pytest.raises
        if "lambda" in executable or "->" in executable:
            # Handle lambda expressions
            lambda_body = self._extract_lambda_body(executable)
            return f"with pytest.raises({exception_class}):\n    {lambda_body}"
        else:
            # Handle method references
            return f"with pytest.raises({exception_class}):\n    {executable}"

    def _convert_assert_does_not_throw(self, assertion_line: str) -> str:
        """Convert assertDoesNotThrow to simple execution."""
        params, message = self._extract_assertion_params(assertion_line, "assertDoesNotThrow")

        if not params:
            return assertion_line

        executable = params[0]

        # Simply execute the code without expecting an exception
        if "lambda" in executable or "->" in executable:
            lambda_body = self._extract_lambda_body(executable)
            return lambda_body
        else:
            return executable

    def _convert_assert_array_equals(self, assertion_line: str) -> str:
        """Convert assertArrayEquals to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertArrayEquals")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert list({actual}) == list({expected}), {message}"
        else:
            return f"assert list({actual}) == list({expected})"

    def _convert_assert_iterable_equals(self, assertion_line: str) -> str:
        """Convert assertIterableEquals to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertIterableEquals")

        if len(params) < 2:
            return assertion_line

        expected = params[0]
        actual = params[1]

        if message:
            return f"assert list({actual}) == list({expected}), {message}"
        else:
            return f"assert list({actual}) == list({expected})"

    def _convert_assert_lines_match(self, assertion_line: str) -> str:
        """Convert assertLinesMatch to pytest assertion."""
        params, message = self._extract_assertion_params(assertion_line, "assertLinesMatch")

        if len(params) < 2:
            return assertion_line

        expected_lines = params[0]
        actual_lines = params[1]

        # This is a complex assertion that might need custom implementation
        if message:
            return f"assert {actual_lines} == {expected_lines}, {message}  # TODO: Implement line matching logic"
        else:
            return (
                f"assert {actual_lines} == {expected_lines}  # TODO: Implement line matching logic"
            )

    def _convert_assert_timeout(self, assertion_line: str) -> str:
        """Convert assertTimeout to pytest timeout."""
        params, message = self._extract_assertion_params(assertion_line, "assertTimeout")

        if len(params) < 2:
            return assertion_line

        timeout = params[0]
        executable = params[1]

        # Convert to pytest timeout (requires pytest-timeout plugin)
        if "lambda" in executable or "->" in executable:
            lambda_body = self._extract_lambda_body(executable)
            return f"# @pytest.mark.timeout({timeout})\n{lambda_body}"
        else:
            return f"# @pytest.mark.timeout({timeout})\n{executable}"

    def _convert_assert_timeout_preemptively(self, assertion_line: str) -> str:
        """Convert assertTimeoutPreemptively to pytest timeout."""
        # Similar to assertTimeout but with preemptive termination
        return self._convert_assert_timeout(
            assertion_line.replace("assertTimeoutPreemptively", "assertTimeout")
        )

    def _convert_fail(self, assertion_line: str) -> str:
        """Convert fail() to pytest.fail()."""
        params, _ = self._extract_assertion_params(assertion_line, "fail")

        if params:
            message = params[0]
            return f"pytest.fail({message})"
        else:
            return "pytest.fail()"

    def _extract_lambda_body(self, lambda_expr: str) -> str:
        """Extract the body from a lambda expression."""
        # Handle Java lambda: () -> expression
        if "->" in lambda_expr:
            parts = lambda_expr.split("->", 1)
            if len(parts) == 2:
                return parts[1].strip()

        # Handle method reference: Class::method
        if "::" in lambda_expr:
            return lambda_expr.replace("::", ".")

        return lambda_expr

    def extract_custom_assertion_methods(self, test_code: str) -> list[str]:
        """
        Extract custom assertion method names from test code.

        Args:
            test_code: Complete test class code

        Returns:
            List of custom assertion method names
        """
        custom_assertions = []

        # Look for methods that start with 'assert' but aren't standard JUnit
        method_pattern = r"(?:private|protected|public)?\s*(?:static)?\s*void\s+(assert\w+)\s*\("
        matches = re.findall(method_pattern, test_code)

        for method_name in matches:
            if method_name not in self._basic_assertion_mappings:
                custom_assertions.append(method_name)

        return custom_assertions

    def convert_custom_assertion(self, assertion_line: str, method_name: str) -> str:
        """
        Convert a custom assertion method call.

        Args:
            assertion_line: The assertion line containing the custom method
            method_name: The custom assertion method name

        Returns:
            Converted Python assertion or method call
        """
        # For custom assertions, we typically convert them to regular method calls
        # The actual assertion logic would be in the method implementation

        # Extract parameters
        params, message = self._extract_assertion_params(assertion_line, method_name)

        # Convert method name to snake_case
        python_method_name = self._camel_to_snake(method_name)

        if params:
            params_str = ", ".join(params)
            return f"self.{python_method_name}({params_str})"
        else:
            return f"self.{python_method_name}()"

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
