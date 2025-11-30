"""
Syntax fix suggestion strategies for Java to Python migration.
"""

import re
from pathlib import Path

from ...core.models import TestFile
from ..fix_models import FixComplexity, FixSuggestion, FixType


class SyntaxSuggestionStrategy:
    """Strategy for generating syntax-related fix suggestions."""

    def generate_java_syntax_fix(
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

    def generate_indentation_fix(
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

    def apply_syntax_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply syntax-related fixes."""
        if "braces" in fix.description.lower():
            # Remove braces and fix indentation
            content = re.sub(r"\s*{\s*", ":", content)
            content = re.sub(r"\s*}\s*", "", content)

        if "semicolon" in fix.description.lower():
            # Remove semicolons at end of lines
            content = re.sub(r";\s*$", "", content, flags=re.MULTILINE)

        return content
