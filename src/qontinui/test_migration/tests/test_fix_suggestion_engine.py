"""
Unit tests for the FixSuggestionEngine class.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

from qontinui.test_migration.core.models import (
    FailureAnalysis,
    FailureType,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.validation.fix_suggestion_engine import (
    FixComplexity,
    FixSuggestion,
    FixSuggestionEngine,
    FixType,
    MigrationIssuePattern,
)


class TestFixSuggestionEngine:
    """Test cases for FixSuggestionEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = FixSuggestionEngine()

    def test_initialization(self):
        """Test that engine initializes with proper patterns and mappings."""
        assert len(self.engine._migration_patterns) > 0
        assert len(self.engine._java_to_python_mappings) > 0
        assert len(self.engine._assertion_mappings) > 0
        assert len(self.engine._annotation_mappings) > 0

        # Check some key mappings exist
        assert "org.junit.jupiter.api.Test" in self.engine._java_to_python_mappings
        assert "assertEquals" in self.engine._assertion_mappings
        assert "@Test" in self.engine._annotation_mappings

    def test_suggest_fixes_brobot_import_error(self):
        """Test fix suggestions for Brobot import errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.9,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'brobot.library'",
                "stack_trace": "from brobot.library import Action",
                "test_name": "test_action",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have import fix suggestion
        import_fixes = [s for s in suggestions if s.fix_type == FixType.IMPORT_FIX]
        assert len(import_fixes) > 0

        brobot_fix = import_fixes[0]
        assert "brobot" in brobot_fix.original_code.lower()
        assert "qontinui" in brobot_fix.suggested_code.lower()
        assert brobot_fix.confidence > 0.8

    def test_suggest_fixes_java_import_error(self):
        """Test fix suggestions for Java import errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.9,
            diagnostic_info={
                "error_message": "ImportError: cannot import name 'java.util.List'",
                "stack_trace": "from java.util import List",
                "test_name": "test_list",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should suggest using Python list
        import_fixes = [s for s in suggestions if s.fix_type == FixType.IMPORT_FIX]
        assert len(import_fixes) > 0

        java_fix = import_fixes[0]
        assert "java.util" in java_fix.original_code
        assert java_fix.confidence > 0.7

    def test_suggest_fixes_junit_annotation_error(self):
        """Test fix suggestions for JUnit annotation errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "NameError: name '@Test' is not defined",
                "stack_trace": "@Test\ndef testMethod():",
                "test_name": "testMethod",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have annotation fix suggestion
        annotation_fixes = [s for s in suggestions if s.fix_type == FixType.ANNOTATION_FIX]
        assert len(annotation_fixes) > 0

        test_fix = annotation_fixes[0]
        assert "@Test" in test_fix.original_code
        assert "def test_" in test_fix.suggested_code
        assert test_fix.confidence > 0.8

    def test_suggest_fixes_junit_assertion_error(self):
        """Test fix suggestions for JUnit assertion errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "AttributeError: module 'unittest' has no attribute 'assertEquals'",
                "stack_trace": "assertEquals(expected, actual)",
                "test_name": "test_calculation",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have assertion fix suggestion
        assertion_fixes = [s for s in suggestions if s.fix_type == FixType.ASSERTION_FIX]
        assert len(assertion_fixes) > 0

        assertion_fix = assertion_fixes[0]
        assert "assertEquals" in assertion_fix.original_code
        assert "assert" in assertion_fix.suggested_code
        assert assertion_fix.confidence > 0.7

    def test_suggest_fixes_spring_annotation_error(self):
        """Test fix suggestions for Spring Boot annotation errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.7,
            diagnostic_info={
                "error_message": "AttributeError: 'TestClass' has no attribute 'SpringBootTest'",
                "stack_trace": "@SpringBootTest\nclass TestClass:",
                "test_name": "test_integration",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have annotation fix suggestion
        annotation_fixes = [s for s in suggestions if s.fix_type == FixType.ANNOTATION_FIX]
        assert len(annotation_fixes) > 0

        spring_fix = annotation_fixes[0]
        assert "@SpringBootTest" in spring_fix.original_code
        assert "pytest.fixture" in spring_fix.suggested_code

    def test_suggest_fixes_mockito_error(self):
        """Test fix suggestions for Mockito errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.7,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'mockito'",
                "stack_trace": "import org.mockito.Mockito",
                "test_name": "test_mock",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have mock fix suggestion
        mock_fixes = [s for s in suggestions if s.fix_type == FixType.MOCK_FIX]
        assert len(mock_fixes) > 0

        mockito_fix = mock_fixes[0]
        assert "mockito" in mockito_fix.original_code.lower()
        assert "unittest.mock" in mockito_fix.suggested_code

    def test_suggest_fixes_java_syntax_error(self):
        """Test fix suggestions for Java syntax errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "SyntaxError: invalid syntax",
                "stack_trace": "if (condition) {\n    statement;\n}",
                "test_name": "test_syntax",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have syntax fix suggestion
        syntax_fixes = [s for s in suggestions if s.fix_type == FixType.SYNTAX_FIX]
        assert len(syntax_fixes) > 0

        syntax_fix = syntax_fixes[0]
        assert "{" in syntax_fix.original_code or ";" in syntax_fix.original_code
        assert syntax_fix.confidence > 0.8

    def test_suggest_fixes_indentation_error(self):
        """Test fix suggestions for indentation errors."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.9,
            diagnostic_info={
                "error_message": "IndentationError: expected an indented block",
                "stack_trace": "if condition:\nreturn result",
                "test_name": "test_indentation",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        assert len(suggestions) > 0

        # Should have syntax fix suggestion for indentation
        syntax_fixes = [s for s in suggestions if s.fix_type == FixType.SYNTAX_FIX]
        assert len(syntax_fixes) > 0

        indent_fix = syntax_fixes[0]
        assert "indentation" in indent_fix.description.lower()
        assert indent_fix.confidence > 0.7

    def test_apply_simple_fixes_with_temp_file(self):
        """Test applying simple fixes to a temporary file."""
        # Create a temporary Python file with issues
        python_content = """
import org.junit.Test

@Test
def testMethod():
    assertEquals(expected, actual)
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            # Create simple fix suggestions
            suggestions = [
                FixSuggestion(
                    fix_type=FixType.SYNTAX_FIX,
                    complexity=FixComplexity.SIMPLE,
                    description="Remove semicolons",
                    original_code="statement;",
                    suggested_code="statement",
                    confidence=0.9,
                ),
                FixSuggestion(
                    fix_type=FixType.IMPORT_FIX,
                    complexity=FixComplexity.COMPLEX,  # Should not be applied
                    description="Complex import fix",
                    original_code="import complex",
                    suggested_code="import simple",
                    confidence=0.5,
                ),
            ]

            applied_fixes = self.engine.apply_simple_fixes(suggestions, python_path)

            # Should only apply simple fixes with high confidence
            assert len(applied_fixes) <= 1  # Only simple fixes should be applied

        finally:
            python_path.unlink()

    def test_apply_simple_fixes_nonexistent_file(self):
        """Test applying fixes to a nonexistent file."""
        nonexistent_path = Path("nonexistent_file.py")
        suggestions = [
            FixSuggestion(
                fix_type=FixType.SYNTAX_FIX,
                complexity=FixComplexity.SIMPLE,
                description="Test fix",
                original_code="test",
                suggested_code="fixed",
                confidence=0.9,
            )
        ]

        applied_fixes = self.engine.apply_simple_fixes(suggestions, nonexistent_path)

        # Should return empty list for nonexistent file
        assert len(applied_fixes) == 0

    def test_recognize_common_patterns(self):
        """Test recognition of common migration patterns."""
        # Test Brobot import pattern
        brobot_patterns = self.engine.recognize_common_patterns(
            "ModuleNotFoundError: No module named 'brobot'", "from brobot.library import Action"
        )
        assert "brobot_import_error" in brobot_patterns

        # Test Java import pattern
        java_patterns = self.engine.recognize_common_patterns(
            "ImportError: cannot import name 'java.util.List'", "from java.util import List"
        )
        assert "java_import_error" in java_patterns

        # Test JUnit annotation pattern
        junit_patterns = self.engine.recognize_common_patterns(
            "NameError: name '@Test' is not defined", "@Test\ndef testMethod():"
        )
        assert "junit_annotation_error" in junit_patterns

        # Test assertion pattern
        assertion_patterns = self.engine.recognize_common_patterns(
            "AttributeError: module has no attribute 'assertEquals'",
            "assertEquals(expected, actual)",
        )
        assert "junit_assertion_error" in assertion_patterns

    def test_migration_issue_pattern_dataclass(self):
        """Test the MigrationIssuePattern dataclass."""
        pattern = MigrationIssuePattern(
            pattern_name="test_pattern",
            pattern_regex=r"test.*error",
            failure_types=[FailureType.SYNTAX_ERROR],
            fix_generator="_generate_test_fix",
            confidence_threshold=0.8,
            description="Test pattern description",
        )

        assert pattern.pattern_name == "test_pattern"
        assert pattern.pattern_regex == r"test.*error"
        assert FailureType.SYNTAX_ERROR in pattern.failure_types
        assert pattern.fix_generator == "_generate_test_fix"
        assert pattern.confidence_threshold == 0.8
        assert pattern.description == "Test pattern description"

    def test_fix_suggestion_dataclass(self):
        """Test the FixSuggestion dataclass."""
        suggestion = FixSuggestion(
            fix_type=FixType.IMPORT_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Test fix description",
            original_code="original code",
            suggested_code="suggested code",
            confidence=0.9,
            file_path=Path("test.py"),
            line_number=10,
            additional_context={"key": "value"},
            prerequisites=["prerequisite1"],
            validation_steps=["step1", "step2"],
        )

        assert suggestion.fix_type == FixType.IMPORT_FIX
        assert suggestion.complexity == FixComplexity.SIMPLE
        assert suggestion.description == "Test fix description"
        assert suggestion.original_code == "original code"
        assert suggestion.suggested_code == "suggested code"
        assert suggestion.confidence == 0.9
        assert suggestion.file_path == Path("test.py")
        assert suggestion.line_number == 10
        assert suggestion.additional_context["key"] == "value"
        assert "prerequisite1" in suggestion.prerequisites
        assert "step1" in suggestion.validation_steps

    def test_map_brobot_to_qontinui(self):
        """Test Brobot to Qontinui mapping."""
        # Test known mappings
        assert "qontinui.actions.Action" in self.engine._map_brobot_to_qontinui(
            "brobot.library.Action"
        )
        assert "qontinui.model.state.State" in self.engine._map_brobot_to_qontinui(
            "brobot.library.State"
        )
        assert "qontinui.find.Find" in self.engine._map_brobot_to_qontinui("brobot.library.Find")

        # Test unknown mapping (should use fallback)
        unknown_mapping = self.engine._map_brobot_to_qontinui("brobot.library.Unknown")
        assert "qontinui" in unknown_mapping
        assert "unknown" in unknown_mapping.lower()

    def test_can_apply_fix_safely(self):
        """Test safe fix application checking."""
        content = "import org.junit.Test\n@Test\ndef testMethod():\n    assertEquals(a, b)"

        # Simple fix with high confidence - should be safe
        safe_fix = FixSuggestion(
            fix_type=FixType.SYNTAX_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Safe fix",
            original_code="@Test",
            suggested_code="def test_",
            confidence=0.9,
        )
        assert self.engine._can_apply_fix_safely(safe_fix, content)

        # Complex fix - should not be safe
        complex_fix = FixSuggestion(
            fix_type=FixType.SYNTAX_FIX,
            complexity=FixComplexity.COMPLEX,
            description="Complex fix",
            original_code="@Test",
            suggested_code="def test_",
            confidence=0.9,
        )
        assert not self.engine._can_apply_fix_safely(complex_fix, content)

        # Low confidence fix - should not be safe
        low_confidence_fix = FixSuggestion(
            fix_type=FixType.SYNTAX_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Low confidence fix",
            original_code="@Test",
            suggested_code="def test_",
            confidence=0.5,
        )
        assert not self.engine._can_apply_fix_safely(low_confidence_fix, content)

        # Fix with non-existent original code - should not be safe
        missing_code_fix = FixSuggestion(
            fix_type=FixType.SYNTAX_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Missing code fix",
            original_code="@NonExistent",
            suggested_code="def test_",
            confidence=0.9,
        )
        assert not self.engine._can_apply_fix_safely(missing_code_fix, content)

    def test_apply_fix_to_content_import_fix(self):
        """Test applying import fixes to content."""
        content = "from org.junit import Test\ndef test_method():\n    pass"

        fix = FixSuggestion(
            fix_type=FixType.IMPORT_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Import fix",
            original_code="from org.junit import Test",
            suggested_code="import pytest",
            confidence=0.9,
        )

        result = self.engine._apply_fix_to_content(fix, content)
        assert "import pytest" in result
        assert "org.junit" not in result

    def test_apply_fix_to_content_assertion_fix(self):
        """Test applying assertion fixes to content."""
        content = "def test_method():\n    assertEquals(expected, actual)"

        fix = FixSuggestion(
            fix_type=FixType.ASSERTION_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Assertion fix",
            original_code="assertEquals(expected, actual)",
            suggested_code="assert actual == expected",
            confidence=0.9,
            additional_context={"assertion_method": "assertEquals"},
        )

        result = self.engine._apply_fix_to_content(fix, content)
        assert "assert actual == expected" in result or "assertEquals" not in result

    def test_apply_fix_to_content_syntax_fix(self):
        """Test applying syntax fixes to content."""
        content = "if (condition) {\n    statement;\n}"

        # Test brace removal
        brace_fix = FixSuggestion(
            fix_type=FixType.SYNTAX_FIX,
            complexity=FixComplexity.SIMPLE,
            description="Replace Java braces with Python indentation",
            original_code="",
            suggested_code="",
            confidence=0.9,
        )

        result = self.engine._apply_syntax_fix(brace_fix, content)
        assert "{" not in result
        assert "}" not in result
        assert ":" in result

    def test_suggest_fixes_with_test_file_context(self):
        """Test fix suggestions with Java test file context."""
        test_file = TestFile(
            path=Path("TestClass.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            test_methods=[
                TestMethod(
                    name="testMethod",
                    annotations=["@Test"],
                    assertions=["assertEquals(expected, actual)"],
                )
            ],
        )

        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'brobot'",
                "stack_trace": "from brobot.library import Action",
                "test_name": "testMethod",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis, test_file)

        assert len(suggestions) > 0
        # Should provide context-aware suggestions
        assert any("brobot" in s.description.lower() for s in suggestions)

    def test_suggest_fixes_sorting(self):
        """Test that fix suggestions are sorted by confidence and complexity."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "Multiple errors: ModuleNotFoundError and SyntaxError",
                "stack_trace": "from brobot.library import Action\nif (condition) {",
                "test_name": "test_multiple",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        if len(suggestions) > 1:
            # Should be sorted by confidence (descending)
            for i in range(len(suggestions) - 1):
                assert suggestions[i].confidence >= suggestions[i + 1].confidence

    def test_empty_error_handling(self):
        """Test handling of empty error messages and stack traces."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.5,
            diagnostic_info={"error_message": "", "stack_trace": "", "test_name": "test_empty"},
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        # Should still provide general migration suggestions
        assert isinstance(suggestions, list)
        # May be empty or contain general suggestions

    def test_case_insensitive_pattern_matching(self):
        """Test that pattern matching is case insensitive."""
        patterns = self.engine.recognize_common_patterns(
            "MODULENOTFOUNDERROR: No module named 'BROBOT'", "FROM BROBOT.LIBRARY IMPORT ACTION"
        )

        # Should still match despite different case
        assert "brobot_import_error" in patterns

    def test_multiple_pattern_matches(self):
        """Test behavior when multiple patterns match the same error."""
        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            diagnostic_info={
                "error_message": "ModuleNotFoundError: No module named 'brobot'. SyntaxError: invalid syntax",
                "stack_trace": "from brobot.library import Action\nif (condition) {",
                "test_name": "test_multiple",
            },
        )

        suggestions = self.engine.suggest_fixes(failure_analysis)

        # Should provide suggestions for multiple issues
        assert len(suggestions) > 0

        # Should have different types of fixes
        fix_types = {s.fix_type for s in suggestions}
        assert len(fix_types) > 1  # Multiple types of fixes

    def test_fix_type_enum(self):
        """Test the FixType enum."""
        assert FixType.IMPORT_FIX.value == "import_fix"
        assert FixType.ANNOTATION_FIX.value == "annotation_fix"
        assert FixType.ASSERTION_FIX.value == "assertion_fix"
        assert FixType.SYNTAX_FIX.value == "syntax_fix"
        assert FixType.DEPENDENCY_FIX.value == "dependency_fix"
        assert FixType.MOCK_FIX.value == "mock_fix"
        assert FixType.SETUP_FIX.value == "setup_fix"

    def test_fix_complexity_enum(self):
        """Test the FixComplexity enum."""
        assert FixComplexity.SIMPLE.value == "simple"
        assert FixComplexity.MODERATE.value == "moderate"
        assert FixComplexity.COMPLEX.value == "complex"
