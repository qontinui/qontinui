"""
Assertion fix suggestion strategies for JUnit to pytest migration.
"""

import re
from pathlib import Path

from ..fix_models import FixComplexity, FixSuggestion, FixType
from ...core.models import TestFile


class AssertionSuggestionStrategy:
    """Strategy for generating assertion-related fix suggestions."""

    def __init__(self) -> None:
        """Initialize the assertion suggestion strategy."""
        self._assertion_mappings = self._initialize_assertion_mappings()

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

    def generate_junit_assertion_fix(
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

    def apply_assertion_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply assertion-related fixes."""
        # Use regex to find and replace assertion patterns
        assertion_method = fix.additional_context.get("assertion_method", "")
        if assertion_method:
            pattern = rf"{assertion_method}\s*\((.*?)\)"
            replacement = fix.suggested_code
            return re.sub(pattern, replacement, content)

        return content.replace(fix.original_code, fix.suggested_code)
