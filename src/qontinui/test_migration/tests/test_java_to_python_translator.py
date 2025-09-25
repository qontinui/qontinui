"""
Unit tests for JavaToPythonTranslator.
"""

from pathlib import Path

from qontinui.test_migration.core.models import (
    Dependency,
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.translation.java_to_python_translator import JavaToPythonTranslator


class TestJavaToPythonTranslator:
    """Test cases for JavaToPythonTranslator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.translator = JavaToPythonTranslator()

    def test_camel_to_snake_conversion(self):
        """Test camelCase to snake_case conversion."""
        assert self.translator._camel_to_snake("testMethod") == "test_method"
        assert self.translator._camel_to_snake("getUserName") == "get_user_name"
        assert self.translator._camel_to_snake("XMLParser") == "xml_parser"
        assert self.translator._camel_to_snake("simpleTest") == "simple_test"
        assert self.translator._camel_to_snake("test") == "test"

    def test_java_type_translation(self):
        """Test Java type to Python type translation."""
        type_mappings = self.translator._java_to_python_types

        assert type_mappings["String"] == "str"
        assert type_mappings["Integer"] == "int"
        assert type_mappings["Boolean"] == "bool"
        assert type_mappings["List"] == "List"
        assert type_mappings["Map"] == "Dict"
        assert type_mappings["void"] == "None"

    def test_method_signature_translation(self):
        """Test Java method signature translation."""
        java_signature = "public void testSomething(String param1, Integer param2)"
        python_signature = self.translator._translate_method_signature(java_signature)

        expected = "def test_something(self, param1: str, param2: int):"
        assert python_signature == expected

    def test_parameter_translation(self):
        """Test Java parameter translation."""
        # Simple parameter
        result = self.translator._translate_parameter("String userName")
        assert result == "user_name: str"

        # Parameter with annotation
        result = self.translator._translate_parameter("@Mock UserService userService")
        assert result == "user_service: UserService"

    def test_variable_declaration_translation(self):
        """Test Java variable declaration translation."""
        java_line = 'String userName = "test"'
        python_line = self.translator._translate_variable_declarations(java_line)

        assert python_line == 'user_name = "test"'

    def test_boolean_value_translation(self):
        """Test Java boolean value translation."""
        assert self.translator._translate_value("true") == "True"
        assert self.translator._translate_value("false") == "False"
        assert self.translator._translate_value("null") == "None"

    def test_constructor_call_translation(self):
        """Test Java constructor call translation."""
        # ArrayList
        result = self.translator._translate_constructor_call("new ArrayList()")
        assert result == "list()"

        # HashMap
        result = self.translator._translate_constructor_call("new HashMap()")
        assert result == "dict()"

        # Custom class
        result = self.translator._translate_constructor_call("new UserService(param)")
        assert result == "UserService(param)"

    def test_method_call_translation(self):
        """Test Java method call translation."""
        java_line = "userService.getUserName()"
        python_line = self.translator._translate_method_calls(java_line)

        assert python_line == "user_service.get_user_name()"

    def test_basic_assertion_translation(self):
        """Test basic assertion translation."""
        # assertTrue
        result = self.translator._basic_assertion_translation("assertTrue(condition)")
        assert result == "assert(condition)"

        # assertFalse
        result = self.translator._basic_assertion_translation("assertFalse(condition)")
        assert result == "assert not(condition)"

        # assertEquals
        result = self.translator._basic_assertion_translation("assertEquals(expected, actual)")
        assert result == "assert(expected, actual)"

    def test_java_line_translation(self):
        """Test complete Java line translation."""
        # Variable declaration with method call
        java_line = "String result = userService.getUserName();"
        python_line = self.translator._translate_java_line(java_line)

        expected = "result = user_service.get_user_name()"
        assert python_line == expected

    def test_method_body_translation(self):
        """Test Java method body translation."""
        java_body = """
        String userName = "test";
        Boolean isValid = userService.validateUser(userName);
        assertTrue(isValid);
        """

        python_lines = self.translator._translate_method_body(java_body)

        expected_lines = [
            'user_name = "test"',
            "is_valid = user_service.validate_user(user_name)",
            "assert(is_valid)",
        ]

        assert python_lines == expected_lines

    def test_setup_method_translation(self):
        """Test @Before method translation."""
        setup_method = TestMethod(
            name="setUp", annotations=["@Before"], body="userService = new UserService();"
        )

        python_lines = self.translator._translate_setup_method(setup_method)

        assert "@pytest.fixture(autouse=True)" in python_lines
        assert "def set_up(self):" in python_lines
        assert "    user_service = UserService()" in python_lines

    def test_teardown_method_translation(self):
        """Test @After method translation."""
        teardown_method = TestMethod(
            name="tearDown", annotations=["@After"], body="userService.cleanup();"
        )

        python_lines = self.translator._translate_teardown_method(teardown_method)

        assert "@pytest.fixture(autouse=True)" in python_lines
        assert "def tear_down(self):" in python_lines
        assert "    yield" in python_lines
        assert "    user_service.cleanup()" in python_lines

    def test_test_method_translation(self):
        """Test @Test method translation."""
        test_method = TestMethod(
            name="testUserValidation",
            annotations=["@Test"],
            body='String userName = "test";\nBoolean result = userService.validateUser(userName);\nassertTrue(result);',
        )

        python_lines = self.translator._translate_test_method_full(test_method)

        assert "def test_user_validation(self):" in python_lines
        assert '    user_name = "test"' in python_lines
        assert "    result = user_service.validate_user(user_name)" in python_lines
        assert "    assert(result)" in python_lines

    def test_test_method_with_timeout(self):
        """Test @Test method with timeout annotation."""
        test_method = TestMethod(
            name="testWithTimeout", annotations=["@Test(timeout = 5000)"], body="// test code"
        )

        python_lines = self.translator._translate_test_method_full(test_method)

        assert "@pytest.mark.timeout(5.0)" in python_lines
        assert "def test_with_timeout(self):" in python_lines

    def test_python_imports_generation(self):
        """Test Python import generation from Java dependencies."""
        test_file = TestFile(
            path=Path("TestClass.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            dependencies=[
                Dependency(java_import="java.util.List"),
                Dependency(java_import="org.junit.Test"),
                Dependency(java_import="org.mockito.Mock"),
            ],
            mock_usage=[MockUsage(mock_type="spring_mock", mock_class="UserService")],
        )

        imports = self.translator._generate_python_imports(test_file)

        assert "import pytest" in imports
        assert "from typing import Any, Dict, List, Optional" in imports
        assert "from typing import List" in imports
        assert "from unittest.mock import Mock, patch, MagicMock" in imports

    def test_class_definition_translation(self):
        """Test Java class definition translation."""
        result = self.translator._translate_class_definition("UserServiceTest")
        assert result == "class UserServiceTest:"

        result = self.translator._translate_class_definition("SimpleTest")
        assert result == "class SimpleTest:"

    def test_complete_test_file_translation(self):
        """Test complete test file translation."""
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            dependencies=[
                Dependency(java_import="org.junit.Test"),
                Dependency(java_import="org.junit.Before"),
            ],
            setup_methods=[
                TestMethod(
                    name="setUp", annotations=["@Before"], body="userService = new UserService();"
                )
            ],
            test_methods=[
                TestMethod(
                    name="testValidUser",
                    annotations=["@Test"],
                    body='String user = "valid";\nBoolean result = userService.validate(user);\nassertTrue(result);',
                )
            ],
        )

        python_code = self.translator.translate_test_file(test_file)

        # Check that the generated code contains expected elements
        assert "import pytest" in python_code
        assert "class UserServiceTest:" in python_code
        assert "@pytest.fixture(autouse=True)" in python_code
        assert "def set_up(self):" in python_code
        assert "def test_valid_user(self):" in python_code
        assert "user_service = UserService()" in python_code
        assert "assert(result)" in python_code

    def test_extract_method_signature(self):
        """Test method signature extraction from Java code."""
        java_method = """
        @Test
        public void testSomething(String param) {
            // method body
        }
        """

        signature = self.translator._extract_method_signature(java_method)
        assert "public void testSomething(String param)" in signature

    def test_extract_method_body(self):
        """Test method body extraction from Java code."""
        java_method = """
        public void testMethod() {
            String test = "value";
            assertTrue(test != null);
        }
        """

        body = self.translator._extract_method_body(java_method)
        expected_lines = ['String test = "value"', "assertTrue(test != null)"]

        for expected_line in expected_lines:
            assert expected_line.strip() in body
