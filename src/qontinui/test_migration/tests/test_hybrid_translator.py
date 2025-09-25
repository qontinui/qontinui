"""
Unit tests for HybridTestTranslator.
"""

from pathlib import Path
from unittest.mock import Mock, patch

from qontinui.test_migration.core.models import MockUsage, TestFile, TestMethod, TestType
from qontinui.test_migration.execution.hybrid_test_translator import (
    HybridTestTranslator,
    TranslationResult,
)


class TestTranslationResult:
    """Test cases for TranslationResult."""

    def test_initialization(self):
        """Test TranslationResult initialization."""
        result = TranslationResult(
            content="test content",
            method="utility",
            confidence=0.9,
            errors=["error1"],
            execution_time=1.5,
        )

        assert result.content == "test content"
        assert result.method == "utility"
        assert result.confidence == 0.9
        assert result.errors == ["error1"]
        assert result.execution_time == 1.5
        assert not result.success  # Has errors

    def test_success_property(self):
        """Test success property calculation."""
        # Success case
        result = TranslationResult("content", "utility", 0.9)
        assert result.success

        # Failure case
        result = TranslationResult("content", "utility", 0.9, errors=["error"])
        assert not result.success


class TestHybridTestTranslator:
    """Test cases for HybridTestTranslator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock()
        self.translator = HybridTestTranslator(
            llm_client=self.mock_llm_client,
            utility_confidence_threshold=0.8,
            llm_confidence_threshold=0.7,
        )

    def test_initialization(self):
        """Test that translator initializes correctly."""
        assert self.translator is not None
        assert self.translator.utility_confidence_threshold == 0.8
        assert self.translator.llm_confidence_threshold == 0.7
        assert self.translator.enable_llm_validation
        assert self.translator.utility_translator is not None
        assert self.translator.llm_translator is not None

    def test_initialization_without_llm(self):
        """Test initialization without LLM client."""
        translator = HybridTestTranslator()
        assert translator.llm_translator.llm_client is None
        assert not translator.enable_llm_validation

    @patch("time.time")
    def test_try_utility_translation_success(self, mock_time):
        """Test successful utility translation."""
        mock_time.side_effect = [0.0, 0.1]  # Start and end times

        test_file = TestFile(
            path=Path("SimpleTest.java"),
            test_type=TestType.UNIT,
            class_name="SimpleTest",
            test_methods=[TestMethod(name="testSimple", body="assertTrue(true);")],
        )

        result = self.translator._try_utility_translation(test_file)

        assert result.method == "utility"
        assert result.success
        assert result.confidence > 0
        assert result.execution_time == 0.1
        assert "import pytest" in result.content

    @patch("time.time")
    def test_try_utility_translation_failure(self, mock_time):
        """Test utility translation failure handling."""
        mock_time.side_effect = [0.0, 0.1]

        test_file = TestFile(path=Path("test.java"), test_type=TestType.UNIT, class_name="Test")

        # Mock utility translator to fail
        with patch.object(
            self.translator.utility_translator, "generate_python_test_file"
        ) as mock_gen:
            mock_gen.side_effect = Exception("Translation failed")

            result = self.translator._try_utility_translation(test_file)

            assert result.method == "utility"
            assert not result.success
            assert result.confidence == 0.0
            assert len(result.errors) > 0

    @patch("time.time")
    def test_try_llm_translation_success(self, mock_time):
        """Test successful LLM translation."""
        mock_time.side_effect = [0.0, 0.2]

        test_file = TestFile(
            path=Path("ExampleTest.java"), test_type=TestType.UNIT, class_name="ExampleTest"
        )

        result = self.translator._try_llm_translation(test_file)

        assert result.method == "llm"
        assert result.success
        assert result.confidence > 0
        assert result.execution_time == 0.2
        assert result.content is not None

    def test_calculate_utility_confidence_simple(self):
        """Test confidence calculation for simple test."""
        test_file = TestFile(
            path=Path("SimpleTest.java"),
            test_type=TestType.UNIT,
            class_name="SimpleTest",
            test_methods=[TestMethod(name="testSimple", body="assertTrue(true);")],
        )

        content = """
import pytest

class TestSimple:
    def test_simple(self):
        assert True
"""

        confidence = self.translator._calculate_utility_confidence(test_file, content, [])
        assert confidence > 0.5

    def test_calculate_utility_confidence_with_errors(self):
        """Test confidence calculation with errors."""
        test_file = TestFile(path=Path("test.java"), test_type=TestType.UNIT, class_name="Test")

        confidence = self.translator._calculate_utility_confidence(test_file, "", ["error"])
        assert confidence == 0.0

    def test_calculate_utility_confidence_complex(self):
        """Test confidence calculation for complex test."""
        # Create complex test with many methods and mocks
        methods = [
            TestMethod(name=f"testMethod{i}", body="// complex logic\n" * 20) for i in range(15)
        ]
        mocks = [MockUsage(mock_type="spring_mock", mock_class=f"Service{i}") for i in range(5)]

        test_file = TestFile(
            path=Path("ComplexTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="ComplexTest",
            test_methods=methods,
            mock_usage=mocks,
        )

        content = "import pytest\nclass TestComplex: pass"
        confidence = self.translator._calculate_utility_confidence(test_file, content, [])
        assert confidence < 0.8  # Should be lower for complex test

    def test_translate_test_file_utility_success(self):
        """Test successful utility translation path."""
        test_file = TestFile(
            path=Path("SimpleTest.java"),
            test_type=TestType.UNIT,
            class_name="SimpleTest",
            test_methods=[TestMethod(name="testSimple", body="assertTrue(true);")],
        )

        result = self.translator.translate_test_file(test_file)

        assert isinstance(result, str)
        assert "import pytest" in result
        assert "class TestSimple:" in result

    def test_translate_test_file_llm_fallback(self):
        """Test LLM fallback when utility confidence is low."""
        # Create a complex test that should trigger LLM fallback
        methods = [
            TestMethod(name=f"testMethod{i}", body="// complex logic\n" * 25) for i in range(12)
        ]
        mocks = [MockUsage(mock_type="spring_mock", mock_class=f"Service{i}") for i in range(6)]

        test_file = TestFile(
            path=Path("ComplexTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="ComplexTest",
            test_methods=methods,
            mock_usage=mocks,
        )

        # Lower the utility threshold to force LLM usage
        self.translator.utility_confidence_threshold = 0.9

        result = self.translator.translate_test_file(test_file)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_test_method_utility_success(self):
        """Test successful utility method translation."""
        java_method = """
        @Test
        public void testExample() {
            assertEquals(5, 5);
        }
        """

        result = self.translator.translate_test_method(java_method)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_test_method_llm_fallback(self):
        """Test LLM fallback for method translation."""
        java_method = "complex method that utility can't handle"

        # Mock utility translator to fail
        with patch.object(
            self.translator.utility_translator, "translate_test_method"
        ) as mock_utility:
            mock_utility.side_effect = Exception("Utility failed")

            result = self.translator.translate_test_method(java_method)

            assert isinstance(result, str)
            # Should contain either translated code or fallback
            assert len(result) > 0

    def test_translate_assertions_utility_success(self):
        """Test successful utility assertion translation."""
        java_assertion = "assertEquals(expected, actual)"

        result = self.translator.translate_assertions(java_assertion)

        assert isinstance(result, str)
        assert "assert" in result or "TODO" in result

    def test_translate_assertions_llm_fallback(self):
        """Test LLM fallback for assertion translation."""
        java_assertion = "assertEquals(expected, actual)"

        # Mock utility to return empty result
        with patch.object(
            self.translator.utility_translator, "translate_assertions"
        ) as mock_utility:
            mock_utility.return_value = ""

            result = self.translator.translate_assertions(java_assertion)

            assert isinstance(result, str)
            assert len(result) > 0

    def test_create_hybrid_translation_utility_better(self):
        """Test hybrid translation when utility result is better."""
        utility_result = TranslationResult(
            content="good utility result", method="utility", confidence=0.9, errors=[]
        )

        llm_result = TranslationResult(
            content="poor llm result", method="llm", confidence=0.5, errors=["some error"]
        )

        test_file = TestFile(path=Path("test.java"), test_type=TestType.UNIT, class_name="Test")

        result = self.translator._create_hybrid_translation(utility_result, llm_result, test_file)

        assert result.content == "good utility result"
        assert result.confidence == 0.9

    def test_create_hybrid_translation_llm_better(self):
        """Test hybrid translation when LLM result is better."""
        utility_result = TranslationResult(
            content="poor utility result", method="utility", confidence=0.3, errors=["error"]
        )

        llm_result = TranslationResult(
            content="good llm result", method="llm", confidence=0.8, errors=[]
        )

        test_file = TestFile(path=Path("test.java"), test_type=TestType.UNIT, class_name="Test")

        result = self.translator._create_hybrid_translation(utility_result, llm_result, test_file)

        assert result.content == "good llm result"
        assert result.confidence == 0.8

    def test_create_hybrid_translation_both_failed(self):
        """Test hybrid translation when both approaches fail."""
        utility_result = TranslationResult(
            content="", method="utility", confidence=0.0, errors=["utility error"]
        )

        llm_result = TranslationResult(
            content="", method="llm", confidence=0.0, errors=["llm error"]
        )

        test_file = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
            test_methods=[TestMethod(name="testMethod")],
        )

        result = self.translator._create_hybrid_translation(utility_result, llm_result, test_file)

        assert result.method == "hybrid_fallback"
        assert result.confidence == 0.3
        assert "Fallback translation" in result.content
        assert "TestExample" in result.content

    def test_create_fallback_translation(self):
        """Test fallback translation creation."""
        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
            test_methods=[
                TestMethod(name="testMethod1", body="some test logic"),
                TestMethod(name="testMethod2", body="more test logic"),
            ],
        )

        result = self.translator._create_fallback_translation(test_file)

        assert isinstance(result, str)
        assert "import pytest" in result
        assert "class TestExample:" in result
        assert "def test_method1(self):" in result
        assert "def test_method2(self):" in result
        assert "pytest.skip" in result

    def test_get_translation_stats_initial(self):
        """Test getting initial translation statistics."""
        stats = self.translator.get_translation_stats()

        assert isinstance(stats, dict)
        assert stats["total_translations"] == 0
        assert stats["utility_success"] == 0
        assert stats["llm_fallback"] == 0
        assert stats["hybrid_success"] == 0
        assert stats["total_time"] == 0.0

    def test_get_translation_stats_after_translations(self):
        """Test getting statistics after performing translations."""
        test_file = TestFile(
            path=Path("SimpleTest.java"),
            test_type=TestType.UNIT,
            class_name="SimpleTest",
            test_methods=[TestMethod(name="testSimple")],
        )

        # Perform a translation
        self.translator.translate_test_file(test_file)

        stats = self.translator.get_translation_stats()

        assert stats["total_translations"] > 0
        assert "utility_success_rate" in stats
        assert "llm_fallback_rate" in stats
        assert "hybrid_success_rate" in stats
        assert "average_time" in stats

    def test_configure_thresholds(self):
        """Test configuring confidence thresholds."""
        original_utility = self.translator.utility_confidence_threshold
        original_llm = self.translator.llm_confidence_threshold

        self.translator.configure_thresholds(utility_threshold=0.9, llm_threshold=0.6)

        assert self.translator.utility_confidence_threshold == 0.9
        assert self.translator.llm_confidence_threshold == 0.6
        assert self.translator.utility_confidence_threshold != original_utility
        assert self.translator.llm_confidence_threshold != original_llm

    def test_reset_stats(self):
        """Test resetting translation statistics."""
        # Perform a translation to generate stats
        test_file = TestFile(path=Path("test.java"), test_type=TestType.UNIT, class_name="Test")
        self.translator.translate_test_file(test_file)

        # Verify stats exist
        stats_before = self.translator.get_translation_stats()
        assert stats_before["total_translations"] > 0

        # Reset stats
        self.translator.reset_stats()

        # Verify stats are reset
        stats_after = self.translator.get_translation_stats()
        assert stats_after["total_translations"] == 0
        assert stats_after["utility_success"] == 0
        assert stats_after["llm_fallback"] == 0
        assert stats_after["hybrid_success"] == 0
        assert stats_after["total_time"] == 0.0

    def test_translation_with_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        test_file = TestFile(
            path=Path("ProblematicTest.java"), test_type=TestType.UNIT, class_name="ProblematicTest"
        )

        # Mock both translators to fail
        with patch.object(
            self.translator.utility_translator, "generate_python_test_file"
        ) as mock_utility:
            with patch.object(self.translator.llm_translator, "translate_test_file") as mock_llm:
                mock_utility.side_effect = Exception("Utility failed")
                mock_llm.side_effect = Exception("LLM failed")

                # Should not raise exception, should return fallback
                result = self.translator.translate_test_file(test_file)

                assert isinstance(result, str)
                assert len(result) > 0
                # Should contain fallback content
                assert "Fallback translation" in result or "import pytest" in result
