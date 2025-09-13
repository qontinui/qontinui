"""
Unit tests for PythonTestGenerator.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from qontinui.test_migration.core.models import (
    Dependency, 
    TestFile, 
    TestMethod, 
    TestType,
    MockUsage
)
from qontinui.test_migration.execution.python_test_generator import PythonTestGenerator


class TestPythonTestGenerator:
    """Test cases for PythonTestGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PythonTestGenerator()
    
    def test_initialization(self):
        """Test that generator initializes correctly."""
        assert self.generator is not None
        assert len(self.generator.dependency_mappings) > 0
        assert len(self.generator.pytest_fixtures) > 0
    
    def test_dependency_mappings_include_junit(self):
        """Test that JUnit dependencies are mapped correctly."""
        mappings = self.generator.dependency_mappings
        assert "org.junit.jupiter.api.Test" in mappings
        assert mappings["org.junit.jupiter.api.Test"] == "pytest"
        assert "org.junit.jupiter.api.Assertions" in mappings
    
    def test_dependency_mappings_include_mockito(self):
        """Test that Mockito dependencies are mapped correctly."""
        mappings = self.generator.dependency_mappings
        assert "org.mockito.Mock" in mappings
        assert mappings["org.mockito.Mock"] == "unittest.mock"
    
    def test_convert_class_name(self):
        """Test Java class name conversion to Python."""
        # Test class with Test suffix
        result = self.generator._convert_class_name("UserServiceTest")
        assert result == "TestUserService"
        
        # Test class without Test suffix
        result = self.generator._convert_class_name("UserService")
        assert result == "TestUserService"
        
        # Test class already starting with Test
        result = self.generator._convert_class_name("TestUserService")
        assert result == "TestTestUserService"
    
    def test_convert_method_name(self):
        """Test Java method name conversion to Python."""
        # Test method without test prefix
        result = self.generator._convert_method_name("shouldCreateUser")
        assert result == "test_shouldCreateUser"
        
        # Test method with test prefix
        result = self.generator._convert_method_name("testCreateUser")
        assert result == "testCreateUser"
        
        # Test method with Test in name
        result = self.generator._convert_method_name("createUserTest")
        assert result == "test_createUser_test"
    
    def test_translate_assertions_basic(self):
        """Test basic JUnit assertion translation."""
        # assertEquals
        result = self.generator.translate_assertions("assertEquals(expected, actual)")
        assert result == "assert expected == actual"
        
        # assertTrue
        result = self.generator.translate_assertions("assertTrue(condition)")
        assert result == "assert condition"
        
        # assertFalse
        result = self.generator.translate_assertions("assertFalse(condition)")
        assert result == "assert not condition"
        
        # assertNull
        result = self.generator.translate_assertions("assertNull(value)")
        assert result == "assert value is None"
        
        # assertNotNull
        result = self.generator.translate_assertions("assertNotNull(value)")
        assert result == "assert value is not None"
    
    def test_generate_imports_basic(self):
        """Test basic import generation."""
        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            dependencies=[],
            mock_usage=[]
        )
        
        imports = self.generator._generate_imports(test_file)
        
        # Should always include pytest
        assert "import pytest" in imports
        # Should include basic Qontinui imports
        assert any("qontinui.core" in imp for imp in imports)
    
    def test_generate_imports_with_mocks(self):
        """Test import generation with mock usage."""
        mock_usage = MockUsage(
            mock_type="spring_mock",
            mock_class="UserRepository"
        )
        
        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            dependencies=[],
            mock_usage=[mock_usage]
        )
        
        imports = self.generator._generate_imports(test_file)
        
        # Should include mock imports
        assert any("unittest.mock" in imp for imp in imports)
        assert any("QontinuiMockGenerator" in imp for imp in imports)
    
    def test_generate_imports_integration_test(self):
        """Test import generation for integration tests."""
        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.INTEGRATION,
            class_name="UserServiceIntegrationTest",
            dependencies=[],
            mock_usage=[]
        )
        
        imports = self.generator._generate_imports(test_file)
        
        # Should include integration test imports
        assert any("ConfigurationManager" in imp for imp in imports)
        assert any("QontinuiStartup" in imp for imp in imports)
    
    def test_generate_test_method_basic(self):
        """Test basic test method generation."""
        test_method = TestMethod(
            name="shouldCreateUser",
            body="User user = new User();\nassertEquals(\"test\", user.getName());"
        )
        
        result = self.generator._generate_test_method(test_method)
        
        # Should have proper method signature
        assert "def test_shouldCreateUser(self):" in result
        # Should have docstring
        assert '"""Migrated from shouldCreateUser."""' in result
        # Should convert assertions
        assert any("assert" in line for line in result)
    
    def test_generate_test_method_with_mocks(self):
        """Test test method generation with mocks."""
        mock_usage = MockUsage(
            mock_type="spring_mock",
            mock_class="UserRepository"
        )
        
        test_method = TestMethod(
            name="shouldCreateUser",
            body="// test body",
            mock_usage=[mock_usage]
        )
        
        result = self.generator._generate_test_method(test_method)
        
        # Should include mock setup
        assert any("mock_" in line for line in result)
    
    def test_generate_setup_method(self):
        """Test setup method generation."""
        setup_method = TestMethod(
            name="setUp",
            body="// setup code"
        )
        
        result = self.generator._generate_setup_method(setup_method)
        
        # Should convert to pytest fixture
        assert "def setup_method(self):" in result
        assert '"""Migrated from setUp."""' in result
    
    def test_generate_setup_method_class_level(self):
        """Test class-level setup method generation."""
        setup_method = TestMethod(
            name="setUpClass",
            body="// class setup code"
        )
        
        result = self.generator._generate_setup_method(setup_method)
        
        # Should be class method
        assert "@classmethod" in result
        assert "def setup_class(cls):" in result
    
    def test_translate_test_file_complete(self):
        """Test complete test file translation."""
        test_method = TestMethod(
            name="shouldCreateUser",
            body="assertEquals(\"expected\", actual);"
        )
        
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            package="com.example.service",
            dependencies=[
                Dependency(java_import="org.junit.jupiter.api.Test")
            ],
            test_methods=[test_method]
        )
        
        result = self.generator.translate_test_file(test_file)
        
        # Should have file header
        assert "Migrated test file from Java" in result
        assert "UserServiceTest.java" in result
        
        # Should have imports
        assert "import pytest" in result
        
        # Should have class definition
        assert "class TestUserService:" in result
        
        # Should have test method
        assert "def test_shouldCreateUser(self):" in result
        
        # Should have converted assertions
        assert "assert" in result
    
    def test_generate_test_file_path(self):
        """Test test file path generation."""
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            package="com.example.service"
        )
        
        target_dir = Path("/target/tests")
        result = self.generator.generate_test_file_path(test_file, target_dir)
        
        expected = Path("/target/tests/com/example/service/testuserservice.py")
        assert result == expected
    
    def test_generate_test_file_path_no_package(self):
        """Test test file path generation without package."""
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            package=""
        )
        
        target_dir = Path("/target/tests")
        result = self.generator.generate_test_file_path(test_file, target_dir)
        
        expected = Path("/target/tests/testuserservice.py")
        assert result == expected
    
    def test_validate_generated_file_valid(self):
        """Test validation of valid generated file."""
        valid_content = '''
import pytest

class TestUserService:
    """Test class."""
    
    def test_create_user(self):
        """Test method."""
        assert True
'''
        
        errors = self.generator.validate_generated_file(valid_content)
        assert len(errors) == 0
    
    def test_validate_generated_file_syntax_error(self):
        """Test validation catches syntax errors."""
        invalid_content = '''
import pytest

class TestUserService
    def test_create_user(self):
        assert True
'''
        
        errors = self.generator.validate_generated_file(invalid_content)
        assert len(errors) > 0
        assert any("Syntax error" in error for error in errors)
    
    def test_validate_generated_file_missing_pytest(self):
        """Test validation catches missing pytest import."""
        content_without_pytest = '''
class TestUserService:
    def test_create_user(self):
        assert True
'''
        
        errors = self.generator.validate_generated_file(content_without_pytest)
        assert any("Missing pytest import" in error for error in errors)
    
    def test_validate_generated_file_no_test_methods(self):
        """Test validation catches missing test methods."""
        content_without_tests = '''
import pytest

class TestUserService:
    def setup_method(self):
        pass
'''
        
        errors = self.generator.validate_generated_file(content_without_tests)
        assert any("No test methods found" in error for error in errors)
    
    def test_validate_generated_file_no_test_class(self):
        """Test validation catches missing test class."""
        content_without_class = '''
import pytest

def test_something():
    assert True
'''
        
        errors = self.generator.validate_generated_file(content_without_class)
        assert any("No test class found" in error for error in errors)
    
    def test_convert_method_body_basic(self):
        """Test basic method body conversion."""
        java_body = '''String name = "test";
int count = 5;
assertEquals(expected, actual);'''
        
        result = self.generator._convert_method_body(java_body)
        
        # Should convert types
        assert any("str " in line for line in result)
        assert any("int " in line for line in result)
        
        # Should convert assertions
        assert any("assert expected == actual" in line for line in result)
    
    def test_convert_method_body_empty(self):
        """Test empty method body conversion."""
        result = self.generator._convert_method_body("")
        assert result == ["pass"]
    
    def test_get_python_equivalent_with_mapping(self):
        """Test getting Python equivalent for mapped dependency."""
        dependency = Dependency(java_import="org.junit.jupiter.api.Test")
        result = self.generator._get_python_equivalent(dependency)
        assert result == "pytest"
    
    def test_get_python_equivalent_with_explicit(self):
        """Test getting Python equivalent with explicit mapping."""
        dependency = Dependency(
            java_import="com.example.Custom",
            python_equivalent="custom_module"
        )
        result = self.generator._get_python_equivalent(dependency)
        assert result == "custom_module"
    
    def test_get_python_equivalent_unknown(self):
        """Test getting Python equivalent for unknown dependency."""
        dependency = Dependency(java_import="com.unknown.Library")
        result = self.generator._get_python_equivalent(dependency)
        assert result is None