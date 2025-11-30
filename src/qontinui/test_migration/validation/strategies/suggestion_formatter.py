"""
Suggestion formatting and application utilities.
"""

from pathlib import Path

from ...core.models import FailureAnalysis, TestFile
from ..fix_models import FixComplexity, FixSuggestion, FixType
from .assertion_suggestions import AssertionSuggestionStrategy
from .import_suggestions import ImportSuggestionStrategy
from .syntax_suggestions import SyntaxSuggestionStrategy


class SuggestionFormatter:
    """Handles suggestion formatting and safe application to files."""

    def __init__(
        self,
        assertion_strategy: AssertionSuggestionStrategy,
        import_strategy: ImportSuggestionStrategy,
        syntax_strategy: SyntaxSuggestionStrategy,
    ) -> None:
        """
        Initialize the suggestion formatter.

        Args:
            assertion_strategy: Strategy for assertion fixes
            import_strategy: Strategy for import fixes
            syntax_strategy: Strategy for syntax fixes
        """
        self._assertion_strategy = assertion_strategy
        self._import_strategy = import_strategy
        self._syntax_strategy = syntax_strategy

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

    def generate_migration_suggestions(
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
            return self._import_strategy.apply_import_fix(fix, content)
        elif fix.fix_type == FixType.ASSERTION_FIX:
            return self._assertion_strategy.apply_assertion_fix(fix, content)
        elif fix.fix_type == FixType.SYNTAX_FIX:
            return self._syntax_strategy.apply_syntax_fix(fix, content)
        else:
            # For other fix types, just do a simple replacement
            return content.replace(fix.original_code, fix.suggested_code)
