"""
Setup and annotation fix suggestion strategies for JUnit to pytest migration.
"""

import re
from pathlib import Path

from ..fix_models import FixComplexity, FixSuggestion, FixType
from ...core.models import TestFile


class SetupSuggestionStrategy:
    """Strategy for generating setup and annotation-related fix suggestions."""

    def __init__(self) -> None:
        """Initialize the setup suggestion strategy."""
        self._annotation_mappings = self._initialize_annotation_mappings()

    def _initialize_annotation_mappings(self) -> dict[str, str]:
        """Initialize Java annotation to Python decorator mappings."""
        return {
            "@Test": "def test_",
            "@BeforeEach": "@pytest.fixture(autouse=True)",
            "@AfterEach": "@pytest.fixture(autouse=True)",
            "@BeforeAll": "@pytest.fixture(scope='session', autouse=True)",
            "@AfterAll": "@pytest.fixture(scope='session', autouse=True)",
            "@SpringBootTest": "@pytest.fixture",
            "@Mock": "@pytest.fixture",
            "@Autowired": "# Use dependency injection or fixture",
        }

    def generate_junit_annotation_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for JUnit annotation errors."""
        suggestions = []

        # Find JUnit annotations in the error
        annotation_matches = re.findall(r"@\w+", error_message + stack_trace)

        for annotation in annotation_matches:
            python_equivalent = self._annotation_mappings.get(annotation, "")

            if annotation == "@Test":
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.ANNOTATION_FIX,
                        complexity=FixComplexity.SIMPLE,
                        description="Convert JUnit @Test to pytest test function",
                        original_code=f"{annotation}\npublic void testMethod()",
                        suggested_code="def test_method():",
                        confidence=0.9,
                        file_path=python_file_path,
                    )
                )
            elif python_equivalent:
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.ANNOTATION_FIX,
                        complexity=FixComplexity.MODERATE,
                        description=f"Replace {annotation} with pytest equivalent",
                        original_code=annotation,
                        suggested_code=python_equivalent,
                        confidence=0.7,
                        file_path=python_file_path,
                    )
                )

        return suggestions
