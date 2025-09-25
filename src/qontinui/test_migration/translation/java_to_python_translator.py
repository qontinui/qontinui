"""
Core Java-to-Python syntax translator for test migration.
"""

import re

from ..core.interfaces import TestTranslator
from ..core.models import TestFile, TestMethod


class JavaToPythonTranslator(TestTranslator):
    """
    Translates Java test code to Python equivalents.

    Handles basic syntax conversion including:
    - Method signatures (Java to Python)
    - Class structure translation
    - Basic type conversions
    - Import statement mapping
    """

    def __init__(self):
        """Initialize the translator with mapping dictionaries."""
        self._java_to_python_types = {
            "String": "str",
            "Integer": "int",
            "Boolean": "bool",
            "Double": "float",
            "Float": "float",
            "Long": "int",
            "List": "List",
            "Map": "Dict",
            "Set": "Set",
            "void": "None",
            "Object": "Any",
        }

        self._java_to_python_imports = {
            "java.util.List": "from typing import List",
            "java.util.Map": "from typing import Dict",
            "java.util.Set": "from typing import Set",
            "java.util.ArrayList": "from typing import List",
            "java.util.HashMap": "from typing import Dict",
            "java.util.HashSet": "from typing import Set",
            "org.junit.Test": "import pytest",
            "org.junit.Before": "import pytest",
            "org.junit.After": "import pytest",
            "org.junit.BeforeClass": "import pytest",
            "org.junit.AfterClass": "import pytest",
            "org.mockito.Mock": "from unittest.mock import Mock",
            "org.mockito.Mockito": "from unittest.mock import Mock, patch",
            "org.springframework.test.context.junit4.SpringJUnit4ClassRunner": "",
            "org.springframework.boot.test.context.SpringBootTest": "",
        }

    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a complete Java test file to Python.

        Args:
            test_file: The Java test file to translate

        Returns:
            Python test code as a string
        """
        python_code = []

        # Add imports
        python_imports = self._generate_python_imports(test_file)
        python_code.extend(python_imports)
        python_code.append("")

        # Add class definition
        class_definition = self._translate_class_definition(test_file.class_name)
        python_code.append(class_definition)
        python_code.append("")

        # Add setup methods
        for setup_method in test_file.setup_methods:
            method_code = self._translate_setup_method(setup_method)
            python_code.extend(method_code)
            python_code.append("")

        # Add test methods
        for test_method in test_file.test_methods:
            method_code = self._translate_test_method_full(test_method)
            python_code.extend(method_code)
            python_code.append("")

        # Add teardown methods
        for teardown_method in test_file.teardown_methods:
            method_code = self._translate_teardown_method(teardown_method)
            python_code.extend(method_code)
            python_code.append("")

        return "\n".join(python_code)

    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single Java test method to Python.

        Args:
            method_code: Java method code as string

        Returns:
            Python method code as string
        """
        # Parse method signature
        method_signature = self._extract_method_signature(method_code)
        if not method_signature:
            return method_code  # Return as-is if can't parse

        # Translate signature
        python_signature = self._translate_method_signature(method_signature)

        # Translate method body
        method_body = self._extract_method_body(method_code)
        python_body = self._translate_method_body(method_body)

        # Combine signature and body
        python_method = [python_signature]
        python_method.extend([f"    {line}" for line in python_body])

        return "\n".join(python_method)

    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate JUnit assertions to pytest assertions.

        Args:
            assertion_code: Java assertion code

        Returns:
            Python assertion code
        """
        # This will be implemented in the AssertionConverter
        # For now, return basic translation
        return self._basic_assertion_translation(assertion_code)

    def _generate_python_imports(self, test_file: TestFile) -> list[str]:
        """Generate Python import statements from Java dependencies."""
        imports = set()

        # Add standard test imports
        imports.add("import pytest")
        imports.add("from typing import Any, Dict, List, Optional")

        # Map Java imports to Python equivalents
        for dependency in test_file.dependencies:
            if dependency.java_import in self._java_to_python_imports:
                python_import = self._java_to_python_imports[dependency.java_import]
                if python_import:  # Skip empty mappings
                    imports.add(python_import)

        # Add mock imports if needed
        if any(mock.mock_type in ["brobot_mock", "spring_mock"] for mock in test_file.mock_usage):
            imports.add("from unittest.mock import Mock, patch, MagicMock")

        return sorted(imports)

    def _translate_class_definition(self, class_name: str) -> str:
        """Translate Java class definition to Python."""
        # Remove 'Test' suffix if present and convert to snake_case for class name
        python_class_name = class_name
        if class_name.endswith("Test"):
            python_class_name = class_name[:-4] + "Test"

        return f"class {python_class_name}:"

    def _translate_setup_method(self, method: TestMethod) -> list[str]:
        """Translate Java @Before/@BeforeClass methods to pytest fixtures."""
        lines = []

        if "@BeforeClass" in method.annotations or "@BeforeAll" in method.annotations:
            lines.append("@pytest.fixture(scope='class', autouse=True)")
        else:
            lines.append("@pytest.fixture(autouse=True)")

        # Convert method name to snake_case
        python_method_name = self._camel_to_snake(method.name)
        lines.append(f"def {python_method_name}(self):")

        # Translate method body
        body_lines = self._translate_method_body(method.body)
        if not body_lines:
            body_lines = ["pass"]

        lines.extend([f"    {line}" for line in body_lines])

        return lines

    def _translate_teardown_method(self, method: TestMethod) -> list[str]:
        """Translate Java @After/@AfterClass methods to pytest fixtures."""
        lines = []

        if "@AfterClass" in method.annotations or "@AfterAll" in method.annotations:
            lines.append("@pytest.fixture(scope='class', autouse=True)")
        else:
            lines.append("@pytest.fixture(autouse=True)")

        # Convert method name to snake_case
        python_method_name = self._camel_to_snake(method.name)
        lines.append(f"def {python_method_name}(self):")
        lines.append("    yield")

        # Translate method body (teardown code goes after yield)
        body_lines = self._translate_method_body(method.body)
        if body_lines:
            lines.extend([f"    {line}" for line in body_lines])

        return lines

    def _translate_test_method_full(self, method: TestMethod) -> list[str]:
        """Translate a complete test method including annotations."""
        lines = []

        # Add pytest decorators
        if "@Test" in method.annotations:
            # Check for timeout, expected exceptions, etc.
            test_annotation = next(
                (ann for ann in method.annotations if ann.startswith("@Test")), "@Test"
            )
            if "timeout" in test_annotation:
                timeout_match = re.search(r"timeout\s*=\s*(\d+)", test_annotation)
                if timeout_match:
                    timeout_ms = int(timeout_match.group(1))
                    timeout_s = timeout_ms / 1000
                    lines.append(f"@pytest.mark.timeout({timeout_s})")

            if "expected" in test_annotation:
                expected_match = re.search(r"expected\s*=\s*(\w+\.class)", test_annotation)
                if expected_match:
                    exception_class = expected_match.group(1).replace(".class", "")
                    lines.append(f"@pytest.mark.raises({exception_class})")

        # Convert method name to snake_case and ensure it starts with 'test_'
        python_method_name = self._camel_to_snake(method.name)
        if not python_method_name.startswith("test_"):
            python_method_name = f"test_{python_method_name}"

        lines.append(f"def {python_method_name}(self):")

        # Translate method body
        body_lines = self._translate_method_body(method.body)
        if not body_lines:
            body_lines = ["pass"]

        lines.extend([f"    {line}" for line in body_lines])

        return lines

    def _extract_method_signature(self, method_code: str) -> str | None:
        """Extract method signature from Java method code."""
        lines = method_code.strip().split("\n")
        for line in lines:
            line = line.strip()
            if ("public" in line or "private" in line or "protected" in line) and "(" in line:
                return line
        return None

    def _extract_method_body(self, method_code: str) -> str:
        """Extract method body from Java method code."""
        # Find the opening brace and extract everything until the matching closing brace
        brace_count = 0
        in_body = False
        body_lines = []

        for line in method_code.split("\n"):
            if "{" in line and not in_body:
                in_body = True
                # Add any code after the opening brace
                after_brace = line.split("{", 1)[1].strip()
                if after_brace and after_brace != "}":
                    body_lines.append(after_brace)
                brace_count += line.count("{") - line.count("}")
            elif in_body:
                brace_count += line.count("{") - line.count("}")
                if brace_count > 0:
                    body_lines.append(line.strip())
                else:
                    # Reached the end of the method
                    # Add any code before the closing brace
                    before_brace = line.split("}")[0].strip()
                    if before_brace:
                        body_lines.append(before_brace)
                    break

        return "\n".join(body_lines)

    def _translate_method_signature(self, signature: str) -> str:
        """Translate Java method signature to Python."""
        # Extract method name and parameters
        method_match = re.search(r"(\w+)\s*\(([^)]*)\)", signature)
        if not method_match:
            return signature

        method_name = method_match.group(1)
        parameters = method_match.group(2).strip()

        # Convert method name to snake_case
        python_method_name = self._camel_to_snake(method_name)

        # Translate parameters
        python_params = ["self"]
        if parameters:
            for param in parameters.split(","):
                param = param.strip()
                if param:
                    python_param = self._translate_parameter(param)
                    python_params.append(python_param)

        return f"def {python_method_name}({', '.join(python_params)}):"

    def _translate_parameter(self, java_param: str) -> str:
        """Translate a Java parameter to Python."""
        # Handle annotations like @Mock, @Autowired
        param = re.sub(r"@\w+\s+", "", java_param)

        # Split type and name
        parts = param.split()
        if len(parts) >= 2:
            java_type = parts[-2]
            param_name = parts[-1]

            # Convert type
            python_type = self._java_to_python_types.get(java_type, java_type)

            # Convert parameter name to snake_case
            python_param_name = self._camel_to_snake(param_name)

            return f"{python_param_name}: {python_type}"
        else:
            # Just a parameter name, no type
            return self._camel_to_snake(param.strip())

    def _translate_method_body(self, body: str) -> list[str]:
        """Translate Java method body to Python."""
        if not body.strip():
            return []

        lines = []
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Translate common Java patterns
            python_line = self._translate_java_line(line)
            lines.append(python_line)

        return lines

    def _translate_java_line(self, line: str) -> str:
        """Translate a single Java line to Python."""
        # Remove semicolons
        line = line.rstrip(";")

        # Translate variable declarations
        line = self._translate_variable_declarations(line)

        # Translate method calls
        line = self._translate_method_calls(line)

        # Translate basic assertions (more detailed translation in AssertionConverter)
        line = self._basic_assertion_translation(line)

        return line

    def _translate_variable_declarations(self, line: str) -> str:
        """Translate Java variable declarations to Python."""
        # Pattern: Type variableName = value;
        var_pattern = r"(\w+)\s+(\w+)\s*=\s*(.+)"
        match = re.match(var_pattern, line)

        if match:
            match.group(1)
            var_name = match.group(2)
            value = match.group(3)

            # Convert variable name to snake_case
            python_var_name = self._camel_to_snake(var_name)

            # Translate the value
            python_value = self._translate_value(value)

            return f"{python_var_name} = {python_value}"

        return line

    def _translate_method_calls(self, line: str) -> str:
        """Translate Java method calls to Python."""
        # Convert camelCase method names to snake_case
        method_pattern = r"(\w+)\.(\w+)\("

        def replace_method_call(match):
            object_name = match.group(1)
            method_name = match.group(2)
            python_object = self._camel_to_snake(object_name)
            python_method = self._camel_to_snake(method_name)
            return f"{python_object}.{python_method}("

        return re.sub(method_pattern, replace_method_call, line)

    def _translate_value(self, value: str) -> str:
        """Translate Java values to Python equivalents."""
        value = value.strip()

        # Boolean values
        if value == "true":
            return "True"
        elif value == "false":
            return "False"
        elif value == "null":
            return "None"

        # String literals - no change needed
        # Numeric literals - no change needed

        # Constructor calls
        if "new " in value:
            return self._translate_constructor_call(value)

        return value

    def _translate_constructor_call(self, constructor: str) -> str:
        """Translate Java constructor calls to Python."""
        # Pattern: new ClassName(args)
        new_pattern = r"new\s+(\w+)\s*\(([^)]*)\)"
        match = re.search(new_pattern, constructor)

        if match:
            class_name = match.group(1)
            args = match.group(2)

            # Map common Java classes to Python equivalents
            class_mapping = {
                "ArrayList": "list",
                "HashMap": "dict",
                "HashSet": "set",
                "StringBuilder": "str",
            }

            python_class = class_mapping.get(class_name, class_name)

            if python_class in ["list", "dict", "set"]:
                if not args.strip():
                    return f"{python_class}()"
                else:
                    return (
                        f"{python_class}([{args}])"
                        if python_class == "list"
                        else f"{python_class}()"
                    )

            return f"{python_class}({args})"

        return constructor

    def _basic_assertion_translation(self, line: str) -> str:
        """Basic assertion translation (detailed version in AssertionConverter)."""
        # Simple mappings for common assertions
        assertion_mappings = {
            "assertTrue": "assert",
            "assertFalse": "assert not",
            "assertEquals": "assert",
            "assertNull": "assert",
            "assertNotNull": "assert",
        }

        for java_assert, python_assert in assertion_mappings.items():
            if java_assert in line:
                # This is a simplified translation - AssertionConverter will handle details
                return line.replace(java_assert, python_assert)

        return line

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters (except the first character)
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
