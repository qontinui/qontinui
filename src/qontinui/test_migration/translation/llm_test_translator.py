"""
LLM-based test translator for complex Java to Python test conversions.
"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.interfaces import TestTranslator
    from ..core.models import TestFile
else:
    try:
        from ..core.interfaces import TestTranslator
        from ..core.models import TestFile
    except ImportError:
        # For standalone testing
        from core.interfaces import TestTranslator
        from core.models import TestFile


@dataclass
class LLMTranslationResult:
    """Result of LLM translation with metadata."""

    translated_code: str
    confidence_score: float
    translation_notes: list[str]
    identified_patterns: list[str]
    suggested_improvements: list[str]


class LLMTestTranslator(TestTranslator):
    """
    LLM-based test translator that uses AI to handle complex Java to Python conversions.

    This translator is designed to complement the utility-based translator by handling:
    - Complex business logic in tests
    - Novel patterns not covered by rules
    - Context-aware refactoring
    - Intelligent mock mapping
    """

    def __init__(self, llm_client=None, model_name: str = "gpt-4") -> None:
        """
        Initialize the LLM translator.

        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            model_name: Model to use for translation
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.brobot_context = self._load_brobot_context()
        self.qontinui_context = self._load_qontinui_context()

    def translate_test_file(self, test_file: TestFile) -> str:
        """
        Translate a complete test file using LLM.

        Args:
            test_file: The TestFile object to translate

        Returns:
            Translated Python test code
        """
        result = self.translate_with_detailed_result(test_file)
        return result.translated_code

    def translate_with_detailed_result(
        self, test_file: TestFile
    ) -> LLMTranslationResult:
        """
        Translate test file and return detailed results with metadata.

        Args:
            test_file: The TestFile object to translate

        Returns:
            LLMTranslationResult with code and metadata
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured. Cannot perform translation.")

        # Build comprehensive prompt
        prompt = self._build_translation_prompt(test_file)

        # Get LLM response
        response = self._call_llm(prompt)

        # Parse response
        return self._parse_llm_response(response, test_file)

    def translate_test_method(self, method_code: str) -> str:
        """
        Translate a single test method using LLM.

        Args:
            method_code: Java test method code

        Returns:
            Translated Python method code
        """
        prompt = self._build_method_translation_prompt(method_code)
        response = self._call_llm(prompt)
        return self._extract_code_from_response(response)

    def translate_assertions(self, assertion_code: str) -> str:
        """
        Translate assertions using LLM for complex cases.

        Args:
            assertion_code: Java assertion code

        Returns:
            Translated Python assertion code
        """
        prompt = f"""
        Translate this Java assertion to Python pytest format:

        Java assertion:
        ```java
        {assertion_code}
        ```

        Requirements:
        - Use pytest assertion syntax (assert statements)
        - Preserve assertion intent and error messages
        - Handle complex assertion chains

        Return only the Python assertion code:
        """

        response = self._call_llm(prompt)
        return self._extract_code_from_response(response)

    def enhance_utility_translation(
        self, utility_result: str, test_file: TestFile
    ) -> LLMTranslationResult:
        """
        Use LLM to enhance and validate utility-based translation.

        Args:
            utility_result: Result from utility-based translator
            test_file: Original test file information

        Returns:
            Enhanced translation with improvements
        """
        prompt = f"""
        Review and enhance this Python test translation:

        Original Java test info:
        - Class: {test_file.class_name}
        - Type: {test_file.test_type.value}
        - Package: {test_file.package}
        - Mock usage: {len(test_file.mock_usage)} mocks

        Current Python translation:
        ```python
        {utility_result}
        ```

        Please:
        1. Identify any issues or improvements
        2. Enhance the translation while preserving the structure
        3. Add missing imports or setup if needed
        4. Improve test readability and Python idioms
        5. Ensure Brobot→Qontinui patterns are correctly applied

        Respond in JSON format:
        {{
            "enhanced_code": "...",
            "improvements": ["list of improvements made"],
            "confidence": 0.95,
            "notes": ["any important notes"]
        }}
        """

        response = self._call_llm(prompt)
        return self._parse_enhancement_response(response)

    def validate_translation(
        self, python_code: str, original_test: TestFile
    ) -> dict[str, Any]:
        """
        Use LLM to validate a Python translation against the original Java test.

        Args:
            python_code: Translated Python test code
            original_test: Original Java test information

        Returns:
            Validation results with issues and suggestions
        """
        prompt = f"""
        Validate this Python test translation:

        Original Java test:
        - Class: {original_test.class_name}
        - Type: {original_test.test_type.value}
        - Methods: {[m.name for m in original_test.test_methods]}

        Python translation:
        ```python
        {python_code}
        ```

        Check for:
        1. Syntax correctness
        2. Test completeness (all methods translated)
        3. Assertion correctness
        4. Import completeness
        5. Mock setup correctness
        6. Pytest conventions

        Respond in JSON format:
        {{
            "is_valid": true/false,
            "syntax_errors": ["list of syntax issues"],
            "missing_elements": ["list of missing components"],
            "suggestions": ["list of improvement suggestions"],
            "confidence": 0.95
        }}
        """

        response = self._call_llm(prompt)
        return self._parse_validation_response(response)

    def _build_translation_prompt(self, test_file: TestFile) -> str:
        """Build comprehensive translation prompt for LLM."""

        # Extract original Java code if available
        java_code = self._extract_java_code(test_file)

        prompt = f"""
        You are an expert in Java and Python test migration, specifically migrating from Brobot (Java GUI automation) to Qontinui (Python GUI automation).

        CONTEXT:
        {self.brobot_context}

        TARGET FRAMEWORK:
        {self.qontinui_context}

        TASK: Translate this Java test to Python pytest format.

        JAVA TEST INFORMATION:
        - File: {test_file.path.name}
        - Class: {test_file.class_name}
        - Package: {test_file.package}
        - Type: {test_file.test_type.value}
        - Test methods: {len(test_file.test_methods)}
        - Mock usage: {len(test_file.mock_usage)} mocks

        JAVA CODE:
        ```java
        {java_code}
        ```

        REQUIREMENTS:
        1. Use pytest conventions (test_*.py files, Test* classes, test_* methods)
        2. Convert JUnit annotations to pytest equivalents
        3. Map JUnit assertions to pytest assert statements
        4. Convert Brobot mocks to Qontinui mock equivalents
        5. Preserve test intent and business logic
        6. Use Python idioms and best practices
        7. Include proper imports for pytest, unittest.mock, and Qontinui
        8. Add docstrings indicating migration source

        BROBOT → QONTINUI MAPPINGS:
        - Brobot State → Qontinui State
        - Brobot Action → Qontinui Action
        - Brobot Mock → Qontinui Mock with equivalent GUI simulation
        - Spring @MockBean → unittest.mock.Mock with proper setup

        Respond in JSON format:
        {{
            "translated_code": "complete Python test file content",
            "confidence": 0.95,
            "notes": ["important translation notes"],
            "patterns": ["identified Java patterns"],
            "suggestions": ["improvement suggestions"]
        }}
        """

        return prompt

    def _build_method_translation_prompt(self, method_code: str) -> str:
        """Build prompt for single method translation."""
        return f"""
        Translate this Java test method to Python pytest format:

        ```java
        {method_code}
        ```

        Requirements:
        - Use pytest conventions
        - Convert assertions to pytest format
        - Handle any Brobot-specific patterns
        - Preserve test logic and intent

        Return only the Python method code:
        """

    def _load_brobot_context(self) -> str:
        """Load Brobot framework context for LLM."""
        return """
        Brobot is a Java-based GUI automation framework that uses:
        - Model-based approach with State objects representing GUI screens
        - Action objects for GUI interactions (click, type, find, etc.)
        - Mock objects for GUI simulation in tests
        - Spring framework for dependency injection
        - JUnit for testing with annotations like @Test, @MockBean
        - StateImage objects for visual element recognition
        - Transition objects for navigation between states
        """

    def _load_qontinui_context(self) -> str:
        """Load Qontinui framework context for LLM."""
        return """
        Qontinui is a Python-based GUI automation framework that uses:
        - Model-based approach similar to Brobot but in Python
        - Action objects for GUI interactions using Python conventions
        - Mock objects using unittest.mock for test simulation
        - pytest for testing framework
        - State management for GUI screen representation
        - Image-based element recognition with OpenCV/PIL
        - Fluent API for action chaining
        - Configuration management for test environments
        """

    def _extract_java_code(self, test_file: TestFile) -> str:
        """Extract or reconstruct Java code from TestFile object."""
        if hasattr(test_file, "original_content") and test_file.original_content:
            return cast(str, test_file.original_content)

        # Reconstruct basic Java structure from TestFile data
        java_code = f"""
package {test_file.package};

// Reconstructed from TestFile object - may be incomplete
public class {test_file.class_name} {{
"""

        for method in test_file.test_methods:
            java_code += f"""
    @Test
    public void {method.name}() {{
        {method.body}
    }}
"""

        java_code += "}\n"
        return java_code

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            LLM response text
        """
        if not self.llm_client:
            # Return mock response for testing
            return self._generate_mock_response(prompt)

        try:
            # This would be implemented based on your LLM client
            # Examples for different clients:

            # OpenAI
            if hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Low temperature for consistent code generation
                )
                return cast(str, response.choices[0].message.content)

            # Anthropic Claude
            elif hasattr(self.llm_client, "messages"):
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                )
                return cast(str, response.content[0].text)

            # Generic client
            else:
                return cast(str, self.llm_client.complete(prompt))

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}") from e

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock LLM response for testing."""
        if "JSON format" in prompt:
            return json.dumps(
                {
                    "translated_code": '''"""
Migrated test file from Java using LLM translation.
"""

import pytest
from unittest.mock import Mock, patch
from qontinui.core import QontinuiCore

class TestExample:
    """Migrated from Java test class."""

    def test_example_method(self):
        """Test method translated by LLM."""
        # LLM-generated test logic
        result = True
        assert result is True
''',
                    "confidence": 0.85,
                    "notes": ["Mock LLM translation for testing"],
                    "patterns": ["JUnit test pattern"],
                    "suggestions": ["Add more specific assertions"],
                }
            )
        else:
            return '''def test_example_method(self):
    """LLM translated method."""
    assert True'''

    def _parse_llm_response(
        self, response: str, test_file: TestFile
    ) -> LLMTranslationResult:
        """Parse LLM response into structured result."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                data = json.loads(response)
                return LLMTranslationResult(
                    translated_code=data.get("translated_code", ""),
                    confidence_score=data.get("confidence", 0.5),
                    translation_notes=data.get("notes", []),
                    identified_patterns=data.get("patterns", []),
                    suggested_improvements=data.get("suggestions", []),
                )
            else:
                # Treat as raw code
                return LLMTranslationResult(
                    translated_code=response,
                    confidence_score=0.7,
                    translation_notes=["Raw LLM response"],
                    identified_patterns=[],
                    suggested_improvements=[],
                )
        except json.JSONDecodeError:
            # Fallback to extracting code from response
            code = self._extract_code_from_response(response)
            return LLMTranslationResult(
                translated_code=code,
                confidence_score=0.6,
                translation_notes=["Parsed from unstructured response"],
                identified_patterns=[],
                suggested_improvements=[],
            )

    def _parse_enhancement_response(self, response: str) -> LLMTranslationResult:
        """Parse enhancement response from LLM."""
        try:
            data = json.loads(response)
            return LLMTranslationResult(
                translated_code=data.get("enhanced_code", ""),
                confidence_score=data.get("confidence", 0.5),
                translation_notes=data.get("notes", []),
                identified_patterns=[],
                suggested_improvements=data.get("improvements", []),
            )
        except json.JSONDecodeError:
            return LLMTranslationResult(
                translated_code=response,
                confidence_score=0.5,
                translation_notes=["Failed to parse enhancement response"],
                identified_patterns=[],
                suggested_improvements=[],
            )

    def _parse_validation_response(self, response: str) -> dict[str, Any]:
        """Parse validation response from LLM."""
        try:
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            return {
                "is_valid": False,
                "syntax_errors": ["Failed to parse validation response"],
                "missing_elements": [],
                "suggestions": [],
                "confidence": 0.0,
            }

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from LLM response."""
        # Look for code blocks
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return cast(str, matches[0]).strip()

        # If no code blocks, return the whole response
        return response.strip()

    def get_translation_stats(self) -> dict[str, Any]:
        """Get statistics about LLM translations performed."""
        return {
            "model_name": self.model_name,
            "client_configured": self.llm_client is not None,
            "brobot_context_loaded": bool(self.brobot_context),
            "qontinui_context_loaded": bool(self.qontinui_context),
        }


class MockLLMClient:
    """Mock LLM client for testing purposes."""

    def complete(self, prompt: str) -> str:
        """Mock completion method."""
        if "JSON format" in prompt:
            return json.dumps(
                {
                    "translated_code": """import pytest

class TestMockTranslation:
    def test_mock_method(self):
        assert True
""",
                    "confidence": 0.9,
                    "notes": ["Mock translation"],
                    "patterns": ["Basic test pattern"],
                    "suggestions": [],
                }
            )
        else:
            return "def test_mock_method(self):\n    assert True"
