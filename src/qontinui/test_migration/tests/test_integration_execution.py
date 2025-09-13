"""
Integration tests for test generation and execution workflow.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from qontinui.test_migration.core.models import (
    Dependency, 
    TestFile, 
    TestMethod, 
    TestType
)
from qontinui.test_migration.execution.python_test_generator import PythonTestGenerator
from qontinui.test_migration.execution.pytest_runner import PytestRunner


class TestExecutionIntegration:
    """Integration tests for test generation and execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PythonTestGenerator()
        self.runner = PytestRunner()
    
    def test_generate_and_validate_test_file(self):
        """Test generating a test file and validating it."""
        # Create a test file model
        test_method = TestMethod(
            name="shouldCreateUser",
            body="assertEquals(expected, actual);"
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
        
        # Generate Python test file
        python_content = self.generator.translate_test_file(test_file)
        
        # Validate the generated file
        errors = self.generator.validate_generated_file(python_content)
        assert len(errors) == 0, f"Validation errors: {errors}"
        
        # Verify key components
        assert "import pytest" in python_content
        assert "class TestUserService:" in python_content
        assert "def test_shouldCreateUser(self):" in python_content
        assert "assert expected == actual" in python_content
    
    def test_generate_and_save_test_file(self):
        """Test generating and saving a test file to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file model
            test_method = TestMethod(
                name="testSimpleAssertion",
                body="assertTrue(true);"
            )
            
            test_file = TestFile(
                path=Path("SimpleTest.java"),
                test_type=TestType.UNIT,
                class_name="SimpleTest",
                test_methods=[test_method]
            )
            
            # Generate Python content
            python_content = self.generator.translate_test_file(test_file)
            
            # Save to file
            target_path = self.generator.generate_test_file_path(test_file, temp_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(python_content)
            
            # Verify file was created
            assert target_path.exists()
            
            # Verify content
            saved_content = target_path.read_text()
            assert "def test_testSimpleAssertion(self):" in saved_content
            assert "assert true" in saved_content
    
    @patch('subprocess.run')
    def test_generate_and_run_test_file(self, mock_subprocess):
        """Test generating a test file and running it with pytest."""
        # Mock successful pytest execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test_simple.py::test_testSimpleAssertion PASSED\n1 passed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create and generate test file
            test_method = TestMethod(
                name="testSimpleAssertion",
                body="assertTrue(true);"
            )
            
            test_file = TestFile(
                path=Path("SimpleTest.java"),
                test_type=TestType.UNIT,
                class_name="SimpleTest",
                test_methods=[test_method]
            )
            
            python_content = self.generator.translate_test_file(test_file)
            
            # Save test file
            target_path = temp_path / "test_simple.py"
            target_path.write_text(python_content)
            
            # Run the test
            result = self.runner.run_single_test(target_path)
            
            # Verify execution
            assert result.passed
            assert result.test_name == "test_simple.py"
            assert result.execution_time >= 0
            
            # Verify subprocess was called correctly
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "pytest" in call_args
            assert str(target_path) in call_args
    
    @patch('subprocess.run')
    def test_generate_and_run_test_suite(self, mock_subprocess):
        """Test generating multiple test files and running as a suite."""
        # Mock successful pytest execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
test_user.py::test_createUser PASSED
test_user.py::test_deleteUser PASSED
test_service.py::test_processData PASSED
3 passed
"""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple test files
            test_files = [
                TestFile(
                    path=Path("UserTest.java"),
                    test_type=TestType.UNIT,
                    class_name="UserTest",
                    test_methods=[
                        TestMethod(name="createUser", body="assertTrue(true);"),
                        TestMethod(name="deleteUser", body="assertFalse(false);")
                    ]
                ),
                TestFile(
                    path=Path("ServiceTest.java"),
                    test_type=TestType.UNIT,
                    class_name="ServiceTest",
                    test_methods=[
                        TestMethod(name="processData", body="assertEquals(1, 1);")
                    ]
                )
            ]
            
            # Generate and save test files
            for test_file in test_files:
                python_content = self.generator.translate_test_file(test_file)
                target_path = self.generator.generate_test_file_path(test_file, temp_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(python_content)
            
            # Run test suite
            results = self.runner.run_test_suite(temp_path)
            
            # Verify results
            assert results.total_tests == 3
            assert results.passed_tests == 3
            assert results.failed_tests == 0
            assert results.execution_time >= 0
    
    def test_generate_test_with_mocks(self):
        """Test generating a test file with mock usage."""
        from qontinui.test_migration.core.models import MockUsage
        
        mock_usage = MockUsage(
            mock_type="spring_mock",
            mock_class="UserRepository"
        )
        
        test_method = TestMethod(
            name="testWithMock",
            body="when(userRepository.findById(1)).thenReturn(user);",
            mock_usage=[mock_usage]
        )
        
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            mock_usage=[mock_usage],
            test_methods=[test_method]
        )
        
        # Generate Python content
        python_content = self.generator.translate_test_file(test_file)
        
        # Verify mock imports and setup
        assert "from unittest.mock import" in python_content
        assert "QontinuiMockGenerator" in python_content
        assert "mock_userrepository" in python_content.lower()
        
        # Validate the file
        errors = self.generator.validate_generated_file(python_content)
        assert len(errors) == 0
    
    def test_generate_integration_test(self):
        """Test generating an integration test file."""
        test_method = TestMethod(
            name="testIntegration",
            body="// Integration test logic"
        )
        
        test_file = TestFile(
            path=Path("UserServiceIntegrationTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="UserServiceIntegrationTest",
            test_methods=[test_method]
        )
        
        # Generate Python content
        python_content = self.generator.translate_test_file(test_file)
        
        # Verify integration test imports
        assert "ConfigurationManager" in python_content
        assert "QontinuiStartup" in python_content
        
        # Validate the file
        errors = self.generator.validate_generated_file(python_content)
        assert len(errors) == 0
    
    def test_runner_environment_validation(self):
        """Test that the runner can validate its environment."""
        errors = self.runner.validate_test_environment()
        
        # Should return a list (may have errors if pytest not installed properly)
        assert isinstance(errors, list)
        
        # Get environment info
        env_info = self.runner.get_test_environment_info()
        assert isinstance(env_info, dict)
    
    def test_runner_configuration(self):
        """Test runner configuration options."""
        config = {
            "verbose": True,
            "capture_output": False,
            "collect_coverage": True,
            "timeout": 120
        }
        
        self.runner.configure_test_environment(config)
        
        assert self.runner.test_config == config
        assert "-v" in self.runner.default_pytest_args
        assert "-s" in self.runner.default_pytest_args
        assert "--cov" in self.runner.default_pytest_args
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from Java test to Python execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Create Java test model
            test_method = TestMethod(
                name="shouldCalculateSum",
                body="int result = 2 + 3;\nassertEquals(5, result);"
            )
            
            test_file = TestFile(
                path=Path("CalculatorTest.java"),
                test_type=TestType.UNIT,
                class_name="CalculatorTest",
                package="com.example.math",
                test_methods=[test_method]
            )
            
            # Step 2: Generate Python test
            python_content = self.generator.translate_test_file(test_file)
            
            # Step 3: Validate generated content
            errors = self.generator.validate_generated_file(python_content)
            assert len(errors) == 0, f"Validation errors: {errors}"
            
            # Step 4: Save to file
            target_path = self.generator.generate_test_file_path(test_file, temp_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(python_content)
            
            # Step 5: Verify file structure
            assert target_path.exists()
            expected_path = temp_path / "com" / "example" / "math" / "testcalculator.py"
            assert target_path == expected_path
            
            # Step 6: Verify content can be parsed as Python
            saved_content = target_path.read_text()
            try:
                compile(saved_content, str(target_path), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Generated Python code has syntax errors: {e}")
            
            # The file is ready for execution (would need actual pytest run for full test)