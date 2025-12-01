"""
Python test file generator for creating pytest-compatible test files from migrated Java tests.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.interfaces import TestTranslator
    from ..core.models import Dependency, TestFile, TestMethod, TestType
else:
    try:
        from ..core.interfaces import TestTranslator
        from ..core.models import Dependency, TestFile, TestMethod, TestType
    except ImportError:
        # For standalone testing
        from core.interfaces import TestTranslator
        from core.models import Dependency, TestFile, TestMethod, TestType


class PythonTestGenerator(TestTranslator):
    """
    Generates pytest-compatible Python test files from migrated Java test data.

    This class handles:
    - Creating proper Python test file structure
    - Generating import statements for Python dependencies
    - Converting test methods with pytest decorators
    - Preserving test logic and assertions
    """

    def __init__(self) -> None:
        """Initialize the Python test generator."""
        self.dependency_mappings = self._initialize_dependency_mappings()
        self.pytest_fixtures = self._initialize_pytest_fixtures()

    def _initialize_dependency_mappings(self) -> dict[str, str]:
        """Initialize common Java to Python dependency mappings."""
        return {
            # JUnit mappings
            "org.junit.jupiter.api.Test": "pytest",
            "org.junit.jupiter.api.BeforeEach": "pytest",
            "org.junit.jupiter.api.AfterEach": "pytest",
            "org.junit.jupiter.api.BeforeAll": "pytest",
            "org.junit.jupiter.api.AfterAll": "pytest",
            "org.junit.jupiter.api.Assertions": "pytest",
            "org.junit.Assert": "pytest",
            # Mockito mappings
            "org.mockito.Mock": "unittest.mock",
            "org.mockito.MockitoAnnotations": "unittest.mock",
            "org.mockito.Mockito": "unittest.mock",
            # Spring Test mappings
            "org.springframework.boot.test.context.SpringBootTest": "pytest",
            "org.springframework.test.context.junit.jupiter.SpringJUnitConfig": "pytest",
            "org.springframework.boot.test.mock.mockito.MockBean": "unittest.mock",
            # Common Java libraries
            "java.util.List": "typing.List",
            "java.util.Map": "typing.Dict",
            "java.util.Optional": "typing.Optional",
            "java.util.Arrays": "# Arrays functionality built into Python lists",
        }

    def _initialize_pytest_fixtures(self) -> dict[str, str]:
        """Initialize pytest fixture mappings for common test patterns."""
        return {
            "setUp": "setup_method",
            "tearDown": "teardown_method",
            "setUpClass": "setup_class",
            "tearDownClass": "teardown_class",
        }

    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a complete test file to Python pytest format.

        Args:
            test_file: The TestFile object containing Java test information

        Returns:
            Complete Python test file content as a string
        """
        return self.generate_python_test_file(test_file)

    def generate_python_test_file(self, test_file: TestFile) -> str:
        """
        Generate a complete Python test file from Java test information.

        Args:
            test_file: The TestFile object containing Java test information

        Returns:
            Complete Python test file content as a string
        """
        lines = []

        # Add file header comment
        lines.append('"""')
        lines.append(f"Migrated test file from Java: {test_file.path.name}")
        lines.append(f"Original package: {test_file.package}")
        lines.append(f"Test type: {test_file.test_type.value}")
        lines.append('"""')
        lines.append("")

        # Generate imports
        imports = self._generate_imports(test_file)
        lines.extend(imports)
        lines.append("")

        # Generate test class
        class_content = self._generate_test_class(test_file)
        lines.extend(class_content)

        return "\n".join(lines)

    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single test method to Python.

        Args:
            method_code: Java test method code

        Returns:
            Python test method code
        """
        # This is a simplified implementation - in practice, this would need
        # more sophisticated parsing and translation
        python_code = method_code

        # Basic Java to Python syntax conversions
        python_code = re.sub(r"@Test\s*", "", python_code)
        python_code = re.sub(r"public\s+void\s+", "def ", python_code)
        python_code = re.sub(r"\{\s*$", ":", python_code, flags=re.MULTILINE)
        python_code = re.sub(r"^\s*\}\s*$", "", python_code, flags=re.MULTILINE)

        return python_code

    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate JUnit assertions to pytest assertions.

        Args:
            assertion_code: JUnit assertion code

        Returns:
            Pytest assertion code
        """
        # Basic assertion translations
        translations = {
            r"assertEquals\(([^,]+),\s*([^)]+)\)": r"assert \1 == \2",
            r"assertTrue\(([^)]+)\)": r"assert \1",
            r"assertFalse\(([^)]+)\)": r"assert not \1",
            r"assertNull\(([^)]+)\)": r"assert \1 is None",
            r"assertNotNull\(([^)]+)\)": r"assert \1 is not None",
            r"assertThat\(([^)]+)\)\.isEqualTo\(([^)]+)\)": r"assert \1 == \2",
        }

        result = assertion_code
        for pattern, replacement in translations.items():
            result = re.sub(pattern, replacement, result)

        return result

    def _generate_imports(self, test_file: TestFile) -> list[str]:
        """Generate Python import statements based on test dependencies."""
        imports = []
        imported_modules = set()

        # Always include pytest
        imports.append("import pytest")
        imported_modules.add("pytest")

        # Add unittest.mock if mocks are used
        if test_file.mock_usage:
            imports.append("from unittest.mock import Mock, patch, MagicMock")
            imported_modules.add("unittest.mock")

        # Add typing imports for common patterns
        typing_imports = set()

        # Process dependencies
        for dependency in test_file.dependencies:
            python_equivalent = self._get_python_equivalent(dependency)
            if python_equivalent and python_equivalent not in imported_modules:
                if python_equivalent.startswith("typing."):
                    typing_imports.add(python_equivalent.split(".", 1)[1])
                elif not python_equivalent.startswith("#"):
                    imports.append(f"import {python_equivalent}")
                    imported_modules.add(python_equivalent)

        # Add typing imports if any
        if typing_imports:
            typing_list = ", ".join(sorted(typing_imports))
            imports.insert(-1, f"from typing import {typing_list}")

        # Add Qontinui imports based on test type
        qontinui_imports = self._generate_qontinui_imports(test_file)
        imports.extend(qontinui_imports)

        return imports

    def _generate_qontinui_imports(self, test_file: TestFile) -> list[str]:
        """Generate Qontinui-specific imports based on test content."""
        imports = []

        # Basic Qontinui imports for all tests
        imports.append("from qontinui.core import QontinuiCore")

        # Add mock imports if needed
        if test_file.mock_usage:
            imports.append("from qontinui.test_migration.mocks import QontinuiMockGenerator")

        # Add integration test imports for integration tests
        if test_file.test_type == TestType.INTEGRATION:
            imports.append("from qontinui.config import ConfigurationManager")
            imports.append("from qontinui.startup import QontinuiStartup")

        return imports

    def _generate_test_class(self, test_file: TestFile) -> list[str]:
        """Generate the Python test class structure."""
        lines = []

        # Class definition
        class_name = self._convert_class_name(test_file.class_name)
        lines.append(f"class {class_name}:")
        lines.append(f'    """Migrated from {test_file.class_name}."""')
        lines.append("")

        # Generate setup methods
        if test_file.setup_methods:
            for setup_method in test_file.setup_methods:
                method_lines = self._generate_setup_method(setup_method)
                lines.extend([f"    {line}" for line in method_lines])
                lines.append("")

        # Generate test methods
        for test_method in test_file.test_methods:
            method_lines = self._generate_test_method(test_method)
            lines.extend([f"    {line}" for line in method_lines])
            lines.append("")

        # Generate teardown methods
        if test_file.teardown_methods:
            for teardown_method in test_file.teardown_methods:
                method_lines = self._generate_teardown_method(teardown_method)
                lines.extend([f"    {line}" for line in method_lines])
                lines.append("")

        return lines

    def _generate_test_method(self, test_method: TestMethod) -> list[str]:
        """Generate a single test method."""
        lines = []

        # Method signature
        method_name = self._convert_method_name(test_method.name)
        lines.append(f"def {method_name}(self):")

        # Method docstring
        lines.append(f'    """Migrated from {test_method.name}."""')

        # Mock setup if needed
        if test_method.mock_usage:
            mock_lines = self._generate_mock_setup(test_method.mock_usage)
            lines.extend([f"    {line}" for line in mock_lines])
            lines.append("")

        # Method body
        if test_method.body:
            body_lines = self._convert_method_body(test_method.body)
            lines.extend([f"    {line}" for line in body_lines])
        else:
            lines.append("    # TODO: Implement test logic")
            lines.append("    pass")

        return lines

    def _generate_setup_method(self, setup_method: TestMethod) -> list[str]:
        """Generate setup method with appropriate pytest fixture."""
        lines = []

        # Determine fixture type
        fixture_name = self.pytest_fixtures.get(setup_method.name, "setup_method")

        if fixture_name in ["setup_class", "teardown_class"]:
            lines.append("@classmethod")
            lines.append(f"def {fixture_name}(cls):")
        else:
            lines.append(f"def {fixture_name}(self):")

        lines.append(f'    """Migrated from {setup_method.name}."""')

        if setup_method.body:
            body_lines = self._convert_method_body(setup_method.body)
            lines.extend([f"    {line}" for line in body_lines])
        else:
            lines.append("    pass")

        return lines

    def _generate_teardown_method(self, teardown_method: TestMethod) -> list[str]:
        """Generate teardown method with appropriate pytest fixture."""
        return self._generate_setup_method(teardown_method)  # Same logic

    def _generate_mock_setup(self, mock_usages: list[Any]) -> list[str]:
        """Generate mock setup code for test methods."""
        lines = []

        for mock_usage in mock_usages:
            if mock_usage.mock_type == "brobot_mock":
                lines.append("# Brobot mock setup - TODO: Implement Qontinui equivalent")
                lines.append(f"# Original mock: {mock_usage.mock_class}")
            elif mock_usage.mock_type == "spring_mock":
                lines.append("# Spring mock setup")
                lines.append(f"mock_{mock_usage.mock_class.lower()} = Mock()")
            else:
                lines.append(f"# Generic mock setup for {mock_usage.mock_class}")
                lines.append(f"mock_{mock_usage.mock_class.lower()} = Mock()")

        return lines

    def _convert_class_name(self, java_class_name: str) -> str:
        """Convert Java class name to Python convention."""
        # Remove 'Test' suffix if present and ensure it starts with 'Test'
        name = java_class_name
        if name.endswith("Test"):
            name = name[:-4]
        return f"Test{name}"

    def _convert_method_name(self, java_method_name: str) -> str:
        """Convert Java method name to Python test method convention."""
        # Ensure method starts with 'test_'
        if not java_method_name.startswith("test"):
            return f"test_{java_method_name}"
        return java_method_name.replace("Test", "_test")

    def _convert_method_body(self, java_body: str) -> list[str]:
        """Convert Java method body to Python."""
        lines = java_body.split("\n")
        python_lines = []

        for line in lines:
            # Basic conversions
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Remove semicolons first
            line = re.sub(r";\s*$", "", line)

            # Convert assertions
            line = self.translate_assertions(line)

            # Basic syntax conversions
            line = re.sub(r"String\s+(\w+)\s*=", r"\1 =", line)  # String variable declaration
            line = re.sub(r"int\s+(\w+)\s*=", r"\1 =", line)  # int variable declaration
            line = re.sub(r"boolean\s+(\w+)\s*=", r"\1 =", line)  # boolean variable declaration
            line = re.sub(r"double\s+(\w+)\s*=", r"\1 =", line)  # double variable declaration
            line = re.sub(r"float\s+(\w+)\s*=", r"\1 =", line)  # float variable declaration

            if line:
                python_lines.append(line)

        return python_lines if python_lines else ["pass"]

    def _get_python_equivalent(self, dependency: Dependency) -> str | None:
        """Get Python equivalent for a Java dependency."""
        if dependency.python_equivalent:
            return dependency.python_equivalent

        return cast(str | None, self.dependency_mappings.get(dependency.java_import))

    def generate_test_file_path(self, test_file: TestFile, target_directory: Path) -> Path:
        """
        Generate the target path for a migrated test file.

        Args:
            test_file: The original test file information
            target_directory: Target directory for migrated tests

        Returns:
            Path where the migrated test should be saved
        """
        # Convert Java package structure to Python module structure
        package_path = test_file.package.replace(".", "/") if test_file.package else ""

        # Convert class name to Python file name (pytest convention)
        python_class_name = self._convert_class_name(test_file.class_name)
        # Convert TestCalculator -> test_calculator.py
        file_name_base = (
            python_class_name[4:] if python_class_name.startswith("Test") else python_class_name
        )
        file_name = f"test_{file_name_base.lower()}.py"

        # Combine paths
        if package_path:
            return target_directory / package_path / file_name
        else:
            return target_directory / file_name

    def validate_generated_file(self, file_content: str) -> list[str]:
        """
        Validate the generated Python test file for common issues.

        Args:
            file_content: The generated Python file content

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for basic Python syntax issues
        try:
            compile(file_content, "<string>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Check for required imports
        if "import pytest" not in file_content:
            errors.append("Missing pytest import")

        # Check for test methods
        if "def test_" not in file_content:
            errors.append("No test methods found")

        # Check for class definition
        if "class Test" not in file_content:
            errors.append("No test class found")

        return errors
