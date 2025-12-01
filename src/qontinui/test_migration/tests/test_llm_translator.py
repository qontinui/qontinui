"""
Unit tests for LLMTestTranslator.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from qontinui.test_migration.core.models import (
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.translation.llm_test_translator import (
    LLMTestTranslator,
    LLMTranslationResult,
    MockLLMClient,
)


class TestLLMTestTranslator:
    """Test cases for LLMTestTranslator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MockLLMClient()
        self.translator = LLMTestTranslator(llm_client=self.mock_client)

    def test_initialization_with_client(self):
        """Test translator initialization with LLM client."""
        assert self.translator.llm_client is not None
        assert self.translator.model_name == "gpt-4"
        assert self.translator.brobot_context is not None
        assert self.translator.qontinui_context is not None

    def test_initialization_without_client(self):
        """Test translator initialization without LLM client."""
        translator = LLMTestTranslator()
        assert translator.llm_client is None
        assert translator.model_name == "gpt-4"

    def test_translate_test_file_without_client_raises_error(self):
        """Test that translation without client raises error."""
        translator = LLMTestTranslator()

        test_file = TestFile(path=Path("Test.java"), test_type=TestType.UNIT, class_name="Test")

        with pytest.raises(ValueError, match="LLM client not configured"):
            translator.translate_test_file(test_file)

    def test_translate_simple_test_file(self):
        """Test translating a simple test file."""
        test_method = TestMethod(name="testExample", body="assertTrue(true);")

        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
            test_methods=[test_method],
        )

        result = self.translator.translate_test_file(test_file)

        assert isinstance(result, str)
        assert "import pytest" in result
        assert "class TestMockTranslation" in result
        assert "def test_mock_method" in result

    def test_translate_with_detailed_result(self):
        """Test translation with detailed result metadata."""
        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
        )

        result = self.translator.translate_with_detailed_result(test_file)

        assert isinstance(result, LLMTranslationResult)
        assert result.translated_code is not None
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.translation_notes, list)
        assert isinstance(result.identified_patterns, list)
        assert isinstance(result.suggested_improvements, list)

    def test_translate_test_method(self):
        """Test translating a single test method."""
        java_method = """
        @Test
        public void testCalculation() {
            int result = 2 + 2;
            assertEquals(4, result);
        }
        """

        result = self.translator.translate_test_method(java_method)

        assert isinstance(result, str)
        assert "def test_mock_method" in result

    def test_translate_assertions(self):
        """Test translating assertions."""
        java_assertion = "assertEquals(expected, actual);"

        result = self.translator.translate_assertions(java_assertion)

        assert isinstance(result, str)
        # Mock client returns simple assertion
        assert "assert" in result or "test_mock_method" in result

    def test_enhance_utility_translation(self):
        """Test enhancing utility translation with LLM."""
        utility_result = """
import pytest

class TestExample:
    def test_method(self):
        assert True
"""

        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
        )

        result = self.translator.enhance_utility_translation(utility_result, test_file)

        assert isinstance(result, LLMTranslationResult)
        assert result.translated_code is not None
        assert result.confidence_score > 0

    def test_validate_translation(self):
        """Test validating a Python translation."""
        python_code = """
import pytest

class TestExample:
    def test_method(self):
        assert True
"""

        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
        )

        result = self.translator.validate_translation(python_code, test_file)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "confidence" in result

    def test_build_translation_prompt(self):
        """Test building translation prompt."""
        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            package="com.example.service",
        )

        prompt = self.translator._build_translation_prompt(test_file)

        assert "UserServiceTest" in prompt
        assert "com.example.service" in prompt
        assert "Brobot" in prompt
        assert "Qontinui" in prompt
        assert "pytest" in prompt

    def test_extract_java_code_with_original_content(self):
        """Test extracting Java code when original content is available."""
        test_file = TestFile(path=Path("Test.java"), test_type=TestType.UNIT, class_name="Test")

        # Add original content
        test_file.original_content = "public class Test { @Test public void test() {} }"

        result = self.translator._extract_java_code(test_file)

        assert result == test_file.original_content

    def test_extract_java_code_reconstruction(self):
        """Test reconstructing Java code from TestFile data."""
        test_method = TestMethod(name="testExample", body="assertTrue(true);")

        test_file = TestFile(
            path=Path("ExampleTest.java"),
            test_type=TestType.UNIT,
            class_name="ExampleTest",
            package="com.example",
            test_methods=[test_method],
        )

        result = self.translator._extract_java_code(test_file)

        assert "package com.example;" in result
        assert "public class ExampleTest" in result
        assert "@Test" in result
        assert "testExample" in result
        assert "assertTrue(true);" in result

    def test_parse_json_llm_response(self):
        """Test parsing JSON LLM response."""
        json_response = json.dumps(
            {
                "translated_code": "import pytest\n\nclass TestExample:\n    def test_method(self):\n        assert True",
                "confidence": 0.95,
                "notes": ["Translation successful"],
                "patterns": ["JUnit test"],
                "suggestions": ["Add more assertions"],
            }
        )

        test_file = TestFile(path=Path("Test.java"), test_type=TestType.UNIT, class_name="Test")

        result = self.translator._parse_llm_response(json_response, test_file)

        assert isinstance(result, LLMTranslationResult)
        assert "import pytest" in result.translated_code
        assert result.confidence_score == 0.95
        assert "Translation successful" in result.translation_notes
        assert "JUnit test" in result.identified_patterns
        assert "Add more assertions" in result.suggested_improvements

    def test_parse_raw_code_response(self):
        """Test parsing raw code LLM response."""
        raw_response = """
import pytest

class TestExample:
    def test_method(self):
        assert True
"""

        test_file = TestFile(path=Path("Test.java"), test_type=TestType.UNIT, class_name="Test")

        result = self.translator._parse_llm_response(raw_response, test_file)

        assert isinstance(result, LLMTranslationResult)
        assert "import pytest" in result.translated_code
        assert result.confidence_score == 0.7  # Default for raw response
        assert "Raw LLM response" in result.translation_notes

    def test_extract_code_from_response_with_code_blocks(self):
        """Test extracting code from response with code blocks."""
        response = """
Here's the translated code:

```python
import pytest

class TestExample:
    def test_method(self):
        assert True
```

This translation preserves the test intent.
"""

        result = self.translator._extract_code_from_response(response)

        assert "import pytest" in result
        assert "class TestExample" in result
        assert "def test_method" in result
        assert "This translation" not in result  # Should exclude explanation

    def test_extract_code_from_response_without_code_blocks(self):
        """Test extracting code from response without code blocks."""
        response = "def test_method(self):\n    assert True"

        result = self.translator._extract_code_from_response(response)

        assert result == response.strip()

    def test_get_translation_stats(self):
        """Test getting translation statistics."""
        stats = self.translator.get_translation_stats()

        assert isinstance(stats, dict)
        assert "model_name" in stats
        assert "client_configured" in stats
        assert "brobot_context_loaded" in stats
        assert "qontinui_context_loaded" in stats
        assert stats["model_name"] == "gpt-4"
        assert stats["client_configured"] is True

    def test_mock_llm_client_complete(self):
        """Test MockLLMClient complete method."""
        client = MockLLMClient()

        # Test JSON response
        json_prompt = "Translate this code. Respond in JSON format:"
        result = client.complete(json_prompt)

        assert isinstance(result, str)
        data = json.loads(result)
        assert "translated_code" in data
        assert "confidence" in data

        # Test regular response
        regular_prompt = "Translate this method:"
        result = client.complete(regular_prompt)

        assert isinstance(result, str)
        assert "def test_mock_method" in result

    def test_translation_with_mocks(self):
        """Test translation of test file with mock usage."""
        mock_usage = MockUsage(mock_type="spring_mock", mock_class="UserRepository")

        test_method = TestMethod(
            name="testWithMock",
            body="when(userRepository.findById(1)).thenReturn(user);",
            mock_usage=[mock_usage],
        )

        test_file = TestFile(
            path=Path("UserServiceTest.java"),
            test_type=TestType.UNIT,
            class_name="UserServiceTest",
            mock_usage=[mock_usage],
            test_methods=[test_method],
        )

        result = self.translator.translate_with_detailed_result(test_file)

        assert isinstance(result, LLMTranslationResult)
        assert result.translated_code is not None
        # Mock client should return some code
        assert len(result.translated_code) > 0

    def test_translation_with_integration_test(self):
        """Test translation of integration test."""
        test_method = TestMethod(name="testIntegration", body="// Integration test logic")

        test_file = TestFile(
            path=Path("UserServiceIntegrationTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="UserServiceIntegrationTest",
            test_methods=[test_method],
        )

        result = self.translator.translate_with_detailed_result(test_file)

        assert isinstance(result, LLMTranslationResult)
        assert result.translated_code is not None

    @patch("qontinui.test_migration.translation.llm_test_translator.json.loads")
    def test_parse_invalid_json_response(self, mock_json_loads):
        """Test handling of invalid JSON response."""
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        invalid_response = "This is not valid JSON"
        test_file = TestFile(path=Path("Test.java"), test_type=TestType.UNIT, class_name="Test")

        result = self.translator._parse_llm_response(invalid_response, test_file)

        assert isinstance(result, LLMTranslationResult)
        assert result.translated_code == invalid_response
        assert result.confidence_score == 0.6
        assert "Parsed from unstructured response" in result.translation_notes
