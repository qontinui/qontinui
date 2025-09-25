"""
Fix suggestion engine for providing automated repair recommendations for test migration issues.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.models import FailureAnalysis, FailureType, TestFile


class FixType(Enum):
    """Types of fixes that can be suggested."""

    IMPORT_FIX = "import_fix"
    ANNOTATION_FIX = "annotation_fix"
    ASSERTION_FIX = "assertion_fix"
    SYNTAX_FIX = "syntax_fix"
    DEPENDENCY_FIX = "dependency_fix"
    MOCK_FIX = "mock_fix"
    SETUP_FIX = "setup_fix"


class FixComplexity(Enum):
    """Complexity levels for fixes."""

    SIMPLE = "simple"  # Can be automatically applied
    MODERATE = "moderate"  # Requires validation
    COMPLEX = "complex"  # Requires manual intervention


@dataclass
class FixSuggestion:
    """Represents a suggested fix for a migration issue."""

    fix_type: FixType
    complexity: FixComplexity
    description: str
    original_code: str
    suggested_code: str
    confidence: float
    file_path: Path | None = None
    line_number: int | None = None
    additional_context: dict[str, Any] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    validation_steps: list[str] = field(default_factory=list)


@dataclass
class MigrationIssuePattern:
    """Represents a pattern for identifying migration issues."""

    pattern_name: str
    pattern_regex: str
    failure_types: list[FailureType]
    fix_generator: str  # Method name to generate fix
    confidence_threshold: float = 0.7
    description: str = ""


class FixSuggestionEngine:
    """
    Engine for providing automated repair recommendations for test migration issues.
    """

    def __init__(self):
        """Initialize the fix suggestion engine."""
        self._migration_patterns = self._initialize_migration_patterns()
        self._java_to_python_mappings = self._initialize_java_to_python_mappings()
        self._assertion_mappings = self._initialize_assertion_mappings()
        self._annotation_mappings = self._initialize_annotation_mappings()

    def suggest_fixes(
        self,
        failure_analysis: FailureAnalysis,
        test_file: TestFile | None = None,
        python_file_path: Path | None = None,
    ) -> list[FixSuggestion]:
        """
        Generate fix suggestions based on failure analysis.

        Args:
            failure_analysis: Analysis of the test failure
            test_file: Original Java test file (optional)
            python_file_path: Path to migrated Python test file (optional)

        Returns:
            List of fix suggestions
        """
        suggestions = []

        # Extract error information from diagnostic info
        diagnostic_info = failure_analysis.diagnostic_info
        error_message = diagnostic_info.get("error_message", "")
        stack_trace = diagnostic_info.get("stack_trace", "")
        diagnostic_info.get("test_name", "")

        # Try to match against known patterns
        for pattern in self._migration_patterns:
            if self._matches_pattern(pattern, error_message, stack_trace):
                fix_method = getattr(self, pattern.fix_generator, None)
                if fix_method:
                    pattern_suggestions = fix_method(
                        error_message, stack_trace, test_file, python_file_path
                    )
                    suggestions.extend(pattern_suggestions)

        # Add general suggestions based on failure analysis
        if failure_analysis.is_migration_issue:
            suggestions.extend(
                self._generate_migration_suggestions(failure_analysis, test_file, python_file_path)
            )

        # Sort by confidence and complexity
        suggestions.sort(key=lambda x: (x.confidence, x.complexity.value), reverse=True)

        return suggestions

    def apply_simple_fixes(
        self, suggestions: list[FixSuggestion], python_file_path: Path
    ) -> list[FixSuggestion]:
        """
        Apply simple fixes automatically to the Python test file.

        Args:
            suggestions: List of fix suggestions
            python_file_path: Path to the Python test file

        Returns:
            List of successfully applied fixes
        """
        applied_fixes: list[FixSuggestion] = []

        if not python_file_path.exists():
            return applied_fixes

        # Read the current file content
        content = python_file_path.read_text(encoding="utf-8")
        modified_content = content

        # Apply simple fixes in order of confidence
        simple_fixes = [s for s in suggestions if s.complexity == FixComplexity.SIMPLE]

        for fix in simple_fixes:
            try:
                if self._can_apply_fix_safely(fix, modified_content):
                    modified_content = self._apply_fix_to_content(fix, modified_content)
                    applied_fixes.append(fix)
            except Exception as e:
                # Log the error but continue with other fixes
                fix.additional_context["application_error"] = str(e)

        # Write back the modified content if any fixes were applied
        if applied_fixes and modified_content != content:
            python_file_path.write_text(modified_content, encoding="utf-8")

        return applied_fixes

    def recognize_common_patterns(self, error_message: str, stack_trace: str) -> list[str]:
        """
        Recognize common migration issue patterns from error messages and stack traces.

        Args:
            error_message: The error message from test failure
            stack_trace: The stack trace from test failure

        Returns:
            List of recognized pattern names
        """
        recognized_patterns = []

        combined_text = f"{error_message}\n{stack_trace}".lower()

        for pattern in self._migration_patterns:
            if re.search(pattern.pattern_regex, combined_text, re.IGNORECASE):
                recognized_patterns.append(pattern.pattern_name)

        return recognized_patterns

    def _initialize_migration_patterns(self) -> list[MigrationIssuePattern]:
        """Initialize common migration issue patterns."""
        return [
            MigrationIssuePattern(
                pattern_name="brobot_import_error",
                pattern_regex=r"modulenotfounderror.*brobot|no module named.*brobot",
                failure_types=[FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_brobot_import_fix",
                confidence_threshold=0.9,
                description="Brobot library import error",
            ),
            MigrationIssuePattern(
                pattern_name="java_import_error",
                pattern_regex=r"importerror.*java\.|no module named.*java\.",
                failure_types=[FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_java_import_fix",
                confidence_threshold=0.9,
                description="Java-specific import error",
            ),
            MigrationIssuePattern(
                pattern_name="junit_annotation_error",
                pattern_regex=r"nameerror.*@test|@test.*not defined|@beforeeach|@aftereach",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_junit_annotation_fix",
                confidence_threshold=0.8,
                description="JUnit annotation error",
            ),
            MigrationIssuePattern(
                pattern_name="junit_assertion_error",
                pattern_regex=r"assertequals|asserttrue|assertfalse|assertnull|assertthrows",
                failure_types=[FailureType.ASSERTION_ERROR],
                fix_generator="_generate_junit_assertion_fix",
                confidence_threshold=0.8,
                description="JUnit assertion method error",
            ),
            MigrationIssuePattern(
                pattern_name="spring_annotation_error",
                pattern_regex=r"@springboottest|@autowired|@component|@service",
                failure_types=[FailureType.DEPENDENCY_ERROR, FailureType.SYNTAX_ERROR],
                fix_generator="_generate_spring_annotation_fix",
                confidence_threshold=0.7,
                description="Spring Boot annotation error",
            ),
            MigrationIssuePattern(
                pattern_name="mockito_error",
                pattern_regex=r"mockito|@mock|when\(.*\)\.then",
                failure_types=[FailureType.MOCK_ERROR, FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_mockito_fix",
                confidence_threshold=0.7,
                description="Mockito mocking framework error",
            ),
            MigrationIssuePattern(
                pattern_name="java_syntax_error",
                pattern_regex=r"syntaxerror.*{|}|invalid syntax.*{|}",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_java_syntax_fix",
                confidence_threshold=0.8,
                description="Java syntax in Python code",
            ),
            MigrationIssuePattern(
                pattern_name="indentation_error",
                pattern_regex=r"indentationerror|expected an indented block",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_indentation_fix",
                confidence_threshold=0.9,
                description="Python indentation error",
            ),
        ]

    def _initialize_java_to_python_mappings(self) -> dict[str, str]:
        """Initialize Java to Python import mappings."""
        return {
            "org.junit.jupiter.api.Test": "pytest",
            "org.junit.jupiter.api.BeforeEach": "pytest",
            "org.junit.jupiter.api.AfterEach": "pytest",
            "org.junit.jupiter.api.BeforeAll": "pytest",
            "org.junit.jupiter.api.AfterAll": "pytest",
            "org.junit.jupiter.api.Assertions": "",  # Use built-in assert
            "org.mockito.Mockito": "unittest.mock",
            "org.mockito.Mock": "unittest.mock",
            "org.springframework.boot.test.context.SpringBootTest": "pytest",
            "org.springframework.test.context.junit.jupiter.SpringJUnitConfig": "pytest",
            "java.util.List": "list",
            "java.util.Map": "dict",
            "java.util.Set": "set",
            "brobot.library.Action": "qontinui.actions.Action",
            "brobot.library.State": "qontinui.model.state.State",
        }

    def _initialize_assertion_mappings(self) -> dict[str, str]:
        """Initialize JUnit to pytest assertion mappings."""
        return {
            "assertEquals": "assert {1} == {0}",
            "assertTrue": "assert {0}",
            "assertFalse": "assert not {0}",
            "assertNull": "assert {0} is None",
            "assertNotNull": "assert {0} is not None",
            "assertThrows": "with pytest.raises({0}):",
            "assertThat": "assert {0}",  # Simplified mapping
        }

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

    def _matches_pattern(
        self, pattern: MigrationIssuePattern, error_message: str, stack_trace: str
    ) -> bool:
        """Check if error matches a migration pattern."""
        combined_text = f"{error_message}\n{stack_trace}"
        return bool(re.search(pattern.pattern_regex, combined_text, re.IGNORECASE))

    def _generate_brobot_import_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Brobot import errors."""
        suggestions = []

        # Extract the specific Brobot import from the error
        brobot_import_match = re.search(r"brobot\.[\w.]+", error_message + stack_trace)
        if brobot_import_match:
            brobot_import = brobot_import_match.group()

            # Map to Qontinui equivalent
            qontinui_equivalent = self._map_brobot_to_qontinui(brobot_import)

            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.IMPORT_FIX,
                    complexity=FixComplexity.SIMPLE,
                    description="Replace Brobot import with Qontinui equivalent",
                    original_code=f"from {brobot_import} import",
                    suggested_code=f"from {qontinui_equivalent} import",
                    confidence=0.9,
                    file_path=python_file_path,
                    additional_context={
                        "brobot_import": brobot_import,
                        "qontinui_equivalent": qontinui_equivalent,
                    },
                )
            )

        return suggestions

    def _generate_java_import_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Java import errors."""
        suggestions = []

        # Extract Java import from error
        java_import_match = re.search(r"java\.[\w.]+", error_message + stack_trace)
        if java_import_match:
            java_import = java_import_match.group()
            python_equivalent = self._java_to_python_mappings.get(java_import, "")

            if python_equivalent:
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.IMPORT_FIX,
                        complexity=FixComplexity.SIMPLE,
                        description="Replace Java import with Python equivalent",
                        original_code=f"from {java_import} import",
                        suggested_code=f"# Use Python built-in: {python_equivalent}",
                        confidence=0.8,
                        file_path=python_file_path,
                    )
                )
            else:
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.IMPORT_FIX,
                        complexity=FixComplexity.COMPLEX,
                        description="Remove Java-specific import and find Python alternative",
                        original_code=f"from {java_import} import",
                        suggested_code=f"# TODO: Find Python equivalent for {java_import}",
                        confidence=0.6,
                        file_path=python_file_path,
                    )
                )

        return suggestions

    def _generate_junit_annotation_fix(
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

    def _generate_junit_assertion_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for JUnit assertion errors."""
        suggestions = []

        # Find JUnit assertion methods
        assertion_pattern = (
            r"(assertEquals|assertTrue|assertFalse|assertNull|assertNotNull|assertThrows)\s*\("
        )
        assertion_matches = re.findall(
            assertion_pattern, error_message + stack_trace, re.IGNORECASE
        )

        for assertion_method in assertion_matches:
            assertion_template = self._assertion_mappings.get(assertion_method, "")

            if assertion_template:
                suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.ASSERTION_FIX,
                        complexity=FixComplexity.SIMPLE,
                        description=f"Convert {assertion_method} to pytest assertion",
                        original_code=f"{assertion_method}(expected, actual)",
                        suggested_code=assertion_template.format("expected", "actual"),
                        confidence=0.8,
                        file_path=python_file_path,
                        additional_context={"assertion_method": assertion_method},
                    )
                )

        return suggestions

    def _generate_spring_annotation_fix(
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

    def _generate_mockito_fix(
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

    def _generate_java_syntax_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Java syntax errors in Python code."""
        suggestions = []

        # Check for common Java syntax patterns
        if "{" in stack_trace or "}" in stack_trace:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.SYNTAX_FIX,
                    complexity=FixComplexity.SIMPLE,
                    description="Replace Java braces with Python indentation",
                    original_code="if (condition) {\n    statement;\n}",
                    suggested_code="if condition:\n    statement",
                    confidence=0.9,
                    file_path=python_file_path,
                )
            )

        if ";" in stack_trace:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.SYNTAX_FIX,
                    complexity=FixComplexity.SIMPLE,
                    description="Remove Java semicolons",
                    original_code="statement;",
                    suggested_code="statement",
                    confidence=0.9,
                    file_path=python_file_path,
                )
            )

        return suggestions

    def _generate_indentation_fix(
        self,
        error_message: str,
        stack_trace: str,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate fixes for Python indentation errors."""
        suggestions = []

        suggestions.append(
            FixSuggestion(
                fix_type=FixType.SYNTAX_FIX,
                complexity=FixComplexity.MODERATE,
                description="Fix Python indentation",
                original_code="# Indentation error detected",
                suggested_code="# Use consistent 4-space indentation",
                confidence=0.8,
                file_path=python_file_path,
                validation_steps=[
                    "Check that all code blocks are properly indented",
                    "Use 4 spaces for each indentation level",
                    "Ensure no mixing of tabs and spaces",
                ],
            )
        )

        return suggestions

    def _generate_migration_suggestions(
        self,
        failure_analysis: FailureAnalysis,
        test_file: TestFile | None,
        python_file_path: Path | None,
    ) -> list[FixSuggestion]:
        """Generate general migration suggestions based on failure analysis."""
        suggestions = []

        if failure_analysis.is_migration_issue and failure_analysis.confidence > 0.7:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.IMPORT_FIX,
                    complexity=FixComplexity.MODERATE,
                    description="Review and update import statements",
                    original_code="# Migration issue detected",
                    suggested_code="# Check imports: replace Java/Brobot imports with Python/Qontinui equivalents",
                    confidence=0.6,
                    file_path=python_file_path,
                    validation_steps=[
                        "Verify all imports are Python-compatible",
                        "Replace Java-specific imports with Python equivalents",
                        "Add missing pytest imports if needed",
                    ],
                )
            )

        return suggestions

    def _map_brobot_to_qontinui(self, brobot_import: str) -> str:
        """Map Brobot import to Qontinui equivalent."""
        mappings = {
            "brobot.library.Action": "qontinui.actions.Action",
            "brobot.library.State": "qontinui.model.state.State",
            "brobot.library.Find": "qontinui.find.Find",
            "brobot.library.Image": "qontinui.model.Image",
            "brobot.library.Region": "qontinui.model.Region",
        }

        return mappings.get(brobot_import, f"qontinui.{brobot_import.split('.')[-1].lower()}")

    def _can_apply_fix_safely(self, fix: FixSuggestion, content: str) -> bool:
        """Check if a fix can be applied safely to the content."""
        # Only apply simple fixes with high confidence
        if fix.complexity != FixComplexity.SIMPLE or fix.confidence < 0.8:
            return False

        # Check if the original code pattern exists in the content
        if fix.original_code and fix.original_code not in content:
            return False

        return True

    def _apply_fix_to_content(self, fix: FixSuggestion, content: str) -> str:
        """Apply a fix to the file content."""
        if fix.fix_type == FixType.IMPORT_FIX:
            return self._apply_import_fix(fix, content)
        elif fix.fix_type == FixType.ASSERTION_FIX:
            return self._apply_assertion_fix(fix, content)
        elif fix.fix_type == FixType.SYNTAX_FIX:
            return self._apply_syntax_fix(fix, content)
        else:
            # For other fix types, just do a simple replacement
            return content.replace(fix.original_code, fix.suggested_code)

    def _apply_import_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply import-related fixes."""
        # Simple string replacement for now
        # In a more sophisticated implementation, this would parse the AST
        return content.replace(fix.original_code, fix.suggested_code)

    def _apply_assertion_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply assertion-related fixes."""
        # Use regex to find and replace assertion patterns
        assertion_method = fix.additional_context.get("assertion_method", "")
        if assertion_method:
            pattern = rf"{assertion_method}\s*\((.*?)\)"
            replacement = fix.suggested_code
            return re.sub(pattern, replacement, content)

        return content.replace(fix.original_code, fix.suggested_code)

    def _apply_syntax_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply syntax-related fixes."""
        if "braces" in fix.description.lower():
            # Remove braces and fix indentation
            content = re.sub(r"\s*{\s*", ":", content)
            content = re.sub(r"\s*}\s*", "", content)

        if "semicolon" in fix.description.lower():
            # Remove semicolons at end of lines
            content = re.sub(r";\s*$", "", content, flags=re.MULTILINE)

        return content
