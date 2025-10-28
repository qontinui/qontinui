"""
Dependency fix suggestion strategies for Spring Boot and Mockito migration.
"""

from pathlib import Path

from ..fix_models import FixComplexity, FixSuggestion, FixType
from ...core.models import TestFile


class DependencySuggestionStrategy:
    """Strategy for generating dependency-related fix suggestions (Spring, Mockito)."""

    def generate_spring_annotation_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Spring Boot annotation errors."""
        suggestions = []

        spring_annotations = ["@SpringBootTest", "@Autowired", "@Component", "@Service"]

        for annotation in spring_annotations:
            if annotation.lower() in (error_message + stack_trace).lower():
                if annotation == "@SpringBootTest":
                    suggestions.append(
                        FixSuggestion(
                            fix_type=FixType.ANNOTATION_FIX,
                            complexity=FixComplexity.COMPLEX,
                            description="Replace Spring Boot test with pytest fixture",
                            original_code=annotation,
                            suggested_code="@pytest.fixture\ndef app_context():\n    # Set up application context",
                            confidence=0.6,
                            file_path=python_file_path,
                        )
                    )
                elif annotation == "@Autowired":
                    suggestions.append(
                        FixSuggestion(
                            fix_type=FixType.DEPENDENCY_FIX,
                            complexity=FixComplexity.MODERATE,
                            description="Replace @Autowired with manual dependency injection",
                            original_code=f"{annotation}\nprivate Service service;",
                            suggested_code="# Use pytest fixture or manual instantiation\nservice = ServiceImpl()",
                            confidence=0.5,
                            file_path=python_file_path,
                        )
                    )

        return suggestions

    def generate_mockito_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Mockito errors."""
        suggestions = []

        if "mockito" in (error_message + stack_trace).lower():
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.MOCK_FIX,
                    complexity=FixComplexity.MODERATE,
                    description="Replace Mockito with unittest.mock",
                    original_code="import org.mockito.Mockito;\nMockito.when(mock.method()).thenReturn(value);",
                    suggested_code="from unittest.mock import Mock\nmock.method.return_value = value",
                    confidence=0.7,
                    file_path=python_file_path,
                    prerequisites=["Add 'from unittest.mock import Mock' import"],
                )
            )

        if "@mock" in (error_message + stack_trace).lower():
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.MOCK_FIX,
                    complexity=FixComplexity.SIMPLE,
                    description="Replace @Mock annotation with pytest fixture",
                    original_code="@Mock\nprivate Service mockService;",
                    suggested_code="@pytest.fixture\ndef mock_service():\n    return Mock()",
                    confidence=0.8,
                    file_path=python_file_path,
                )
            )

        return suggestions
