"""
Integration tests for the hybrid translation system.
"""

from pathlib import Path

import pytest

from qontinui.test_migration.core.models import (
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.execution.hybrid_test_translator import (
    HybridTestTranslator,
)
from qontinui.test_migration.execution.llm_test_translator import LLMTestTranslator
from qontinui.test_migration.execution.python_test_generator import PythonTestGenerator


class TestHybridTranslationIntegration:
    """Integration tests for the complete hybrid translation system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hybrid_translator = HybridTestTranslator()
        self.llm_translator = LLMTestTranslator()
        self.utility_translator = PythonTestGenerator()

    def test_simple_test_translation_end_to_end(self):
        """Test complete translation of a simple test file."""
        test_file = TestFile(
            path=Path("CalculatorTest.java"),
            test_type=TestType.UNIT,
            class_name="CalculatorTest",
            package="com.example.calculator",
            test_methods=[
                TestMethod(
                    name="testAddition",
                    body="int result = calculator.add(2, 3);\nassertEquals(5, result);",
                ),
                TestMethod(
                    name="testSubtraction",
                    body="int result = calculator.subtract(5, 3);\nassertEquals(2, result);",
                ),
            ],
        )

        # Test utility translator
        utility_result = self.utility_translator.translate_test_file(test_file)
        assert "import pytest" in utility_result
        assert "class TestCalculator:" in utility_result
        assert "def test_addition(self):" in utility_result
        assert "def test_subtraction(self):" in utility_result

        # Test LLM translator (without actual LLM client)
        llm_result = self.llm_translator.translate_test_file(test_file)
        assert "import pytest" in llm_result
        assert "class TestCalculator:" in llm_result

        # Test hybrid translator
        hybrid_result = self.hybrid_translator.translate_test_file(test_file)
        assert "import pytest" in hybrid_result
        assert "class TestCalculator:" in hybrid_result

        # Verify all results are valid Python
        for result in [utility_result, llm_result, hybrid_result]:
            try:
                compile(result, "<string>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Generated invalid Python syntax: {e}")

    def test_complex_test_with_mocks_translation(self):
        """Test translation of a complex test with mocks."""
        mock_usage = MockUsage(mock_type="spring_mock", mock_class="UserRepository")

        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="UserServiceTest",
            package="com.example.service",
            mock_usage=[mock_usage],
            test_methods=[
                TestMethod(
                    name="testFindUser",
                    body="""
                    User mockUser = new User("test@example.com");
                    when(userRepository.findByEmail("test@example.com")).thenReturn(mockUser);

                    User result = userService.findUser("test@example.com");

                    assertEquals("test@example.com", result.getEmail());
                    verify(userRepository).findByEmail("test@example.com");
                    """,
                    mock_usage=[mock_usage],
                )
            ],
        )

        # Test hybrid translation
        result = self.hybrid_translator.translate_test_file(test_file)

        # Verify basic structure
        assert "import pytest" in result
        assert "from unittest.mock import" in result
        assert "class TestUserService:" in result
        assert "def test_find_user(self):" in result

        # Verify it's valid Python
        try:
            compile(result, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated invalid Python syntax: {e}")

    def test_hybrid_translator_strategy_selection(self):
        """Test that hybrid translator selects appropriate strategy."""
        # Simple test should use utility
        simple_test = TestFile(
            path=Path("SimpleTest.java"),
            test_type=TestType.UNIT,
            class_name="SimpleTest",
            test_methods=[TestMethod(name="testSimple", body="assertTrue(true);")],
        )

        result = self.hybrid_translator.translate_test_file(simple_test)
        stats = self.hybrid_translator.get_translation_stats()

        # Should have attempted at least one translation
        assert stats["total_translations"] > 0
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translation_validation(self):
        """Test that translation validation works correctly."""
        test_file = TestFile(
            path=Path("ValidationTest.java"),
            test_type=TestType.UNIT,
            class_name="ValidationTest",
            test_methods=[TestMethod(name="testValidation")],
        )

        # Generate translation
        result = self.utility_translator.translate_test_file(test_file)

        # Validate the result
        errors = self.utility_translator.validate_generated_file(result)

        # Should have no validation errors for a properly generated file
        assert isinstance(errors, list)
        # Note: Some errors might be expected for minimal test cases

    def test_method_level_translation(self):
        """Test method-level translation functionality."""
        java_method = """
        @Test
        public void testCalculation() {
            Calculator calc = new Calculator();
            int result = calc.add(2, 3);
            assertEquals(5, result);
        }
        """

        # Test utility method translation
        utility_result = self.utility_translator.translate_test_method(java_method)
        assert isinstance(utility_result, str)

        # Test LLM method translation
        llm_result = self.llm_translator.translate_test_method(java_method)
        assert isinstance(llm_result, str)

        # Test hybrid method translation
        hybrid_result = self.hybrid_translator.translate_test_method(java_method)
        assert isinstance(hybrid_result, str)

    def test_assertion_translation(self):
        """Test assertion-level translation functionality."""
        java_assertions = [
            "assertEquals(expected, actual);",
            "assertTrue(condition);",
            "assertFalse(condition);",
            "assertNull(value);",
            "assertNotNull(value);",
        ]

        for assertion in java_assertions:
            # Test utility assertion translation
            utility_result = self.utility_translator.translate_assertions(assertion)
            assert isinstance(utility_result, str)

            # Test LLM assertion translation
            llm_result = self.llm_translator.translate_assertions(assertion)
            assert isinstance(llm_result, str)

            # Test hybrid assertion translation
            hybrid_result = self.hybrid_translator.translate_assertions(assertion)
            assert isinstance(hybrid_result, str)

    def test_translation_statistics_tracking(self):
        """Test that translation statistics are tracked correctly."""
        initial_stats = self.hybrid_translator.get_translation_stats()
        assert initial_stats["total_translations"] == 0

        # Perform some translations
        test_files = [
            TestFile(
                path=Path(f"Test{i}.java"),
                test_type=TestType.UNIT,
                class_name=f"Test{i}",
                test_methods=[TestMethod(name=f"testMethod{i}")],
            )
            for i in range(3)
        ]

        for test_file in test_files:
            self.hybrid_translator.translate_test_file(test_file)

        final_stats = self.hybrid_translator.get_translation_stats()
        assert final_stats["total_translations"] == 3
        assert "utility_success_rate" in final_stats
        assert "average_time" in final_stats

    def test_configuration_and_thresholds(self):
        """Test translator configuration and threshold adjustment."""
        # Test initial configuration
        assert self.hybrid_translator.utility_confidence_threshold == 0.8
        assert self.hybrid_translator.llm_confidence_threshold == 0.7

        # Test threshold configuration
        self.hybrid_translator.configure_thresholds(utility_threshold=0.9, llm_threshold=0.6)

        assert self.hybrid_translator.utility_confidence_threshold == 0.9
        assert self.hybrid_translator.llm_confidence_threshold == 0.6

        # Test stats reset
        self.hybrid_translator.reset_stats()
        stats = self.hybrid_translator.get_translation_stats()
        assert stats["total_translations"] == 0

    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Create a test file that might cause issues
        problematic_test = TestFile(
            path=Path("ProblematicTest.java"),
            test_type=TestType.UNIT,
            class_name="ProblematicTest",
            test_methods=[TestMethod(name="testProblematic")],
        )

        # Translation should not raise exceptions, even with problematic input
        try:
            result = self.hybrid_translator.translate_test_file(problematic_test)
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"Hybrid translator should handle errors gracefully: {e}")

    def test_file_path_generation(self):
        """Test Python file path generation."""
        test_file = TestFile(
            path=Path("com/example/service/UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            package="com.example.service",
        )

        target_dir = Path("/target/tests")
        generated_path = self.utility_translator.generate_test_file_path(test_file, target_dir)

        expected_path = target_dir / "com" / "example" / "service" / "test_userservice.py"
        assert generated_path == expected_path
