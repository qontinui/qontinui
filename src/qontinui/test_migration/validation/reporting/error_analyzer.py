"""
Error analysis and assertion comparison for diagnostic reporting.
"""

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import TestFile
else:
    try:
        from ...core.models import TestFile
    except ImportError:
        from core.models import TestFile

from .report_models import AssertionDifference


class ErrorAnalyzer:
    """
    Analyzes errors, assertions, and test logic differences.

    Responsibilities:
    - Compare assertion logic between Java and Python
    - Calculate assertion confidence scores
    - Classify assertion types
    - Suggest assertion improvements
    """

    def __init__(self) -> None:
        """Initialize the error analyzer."""
        self._assertion_patterns = self._initialize_assertion_patterns()

    def compare_assertion_logic(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[AssertionDifference]:
        """
        Compare assertion logic between original and migrated tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of assertion differences
        """
        differences = []

        # Read Python test content
        python_content = python_test_path.read_text(encoding="utf-8")

        # Extract Python assertions
        python_assertions = self._extract_python_assertions(python_content)

        for method in java_test.test_methods:
            for java_assertion in method.assertions:
                # Find the assertion type
                assertion_type = self._classify_assertion(java_assertion)

                # Look for equivalent Python assertion
                python_equivalent = self._find_equivalent_python_assertion(
                    java_assertion, python_assertions
                )

                if python_equivalent:
                    # Check semantic equivalence
                    semantic_equivalent = self._check_semantic_equivalence(
                        java_assertion, python_equivalent
                    )

                    confidence = self._calculate_assertion_confidence(  # type: ignore[attr-defined]
                        java_assertion, python_equivalent
                    )

                    differences.append(
                        AssertionDifference(
                            java_assertion=java_assertion,
                            python_assertion=python_equivalent,
                            assertion_type=assertion_type,
                            semantic_equivalent=semantic_equivalent,
                            confidence=confidence,
                            suggested_improvement=self._suggest_assertion_improvement(  # type: ignore[attr-defined]
                                java_assertion, python_equivalent
                            ),
                        )
                    )
                else:
                    # No equivalent found
                    differences.append(
                        AssertionDifference(
                            java_assertion=java_assertion,
                            python_assertion="",
                            assertion_type=assertion_type,
                            semantic_equivalent=False,
                            confidence=0.0,
                            suggested_improvement=f"Missing Python assertion for: {java_assertion}",
                        )
                    )

        return differences

    def calculate_assertion_confidence(self, java_assertion: str, python_assertion: str) -> float:
        """
        Calculate confidence in assertion equivalence.

        Args:
            java_assertion: Java assertion statement
            python_assertion: Python assertion statement

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on content similarity and semantic equivalence
        content_similarity = self._calculate_content_similarity(java_assertion, python_assertion)
        semantic_equivalent = self._check_semantic_equivalence(java_assertion, python_assertion)

        confidence = content_similarity
        if semantic_equivalent:
            confidence = min(1.0, confidence + 0.3)

        return confidence

    def suggest_assertion_improvement(self, java_assertion: str, python_assertion: str) -> str:
        """
        Suggest improvements for assertion migration.

        Args:
            java_assertion: Java assertion statement
            python_assertion: Python assertion statement

        Returns:
            Suggestion for improvement
        """
        if not python_assertion:
            return f"Add Python assertion equivalent to: {java_assertion}"

        confidence = self.calculate_assertion_confidence(java_assertion, python_assertion)

        if confidence < 0.5:
            return "Review assertion migration - low confidence match"
        elif confidence < 0.8:
            return "Consider improving assertion clarity"
        else:
            return "Assertion migration looks good"

    def _initialize_assertion_patterns(self) -> dict[str, str]:
        """Initialize assertion pattern mappings."""
        return {
            r"assertEquals\((.*?),\s*(.*?)\)": r"assert \2 == \1",
            r"assertTrue\((.*?)\)": r"assert \1",
            r"assertFalse\((.*?)\)": r"assert not \1",
            r"assertNull\((.*?)\)": r"assert \1 is None",
            r"assertNotNull\((.*?)\)": r"assert \1 is not None",
            r"assertThrows\((.*?),\s*(.*?)\)": r"with pytest.raises(\1): \2",
        }

    def _extract_python_assertions(self, python_content: str) -> list[str]:
        """Extract assertion statements from Python test content."""
        assertions = []

        # Extract assert statements and pytest.raises
        assert_pattern = r"(assert\s+.*?)(?:\n|$)"
        pytest_raises_pattern = r"(with\s+pytest\.raises\(.*?\):.*?)(?:\n|$)"

        for pattern in [assert_pattern, pytest_raises_pattern]:
            matches = re.findall(pattern, python_content, re.MULTILINE)
            assertions.extend([match.strip() for match in matches])

        return assertions

    def _classify_assertion(self, java_assertion: str) -> str:
        """Classify the type of Java assertion."""
        if "assertEquals" in java_assertion:
            return "junit_equals"
        elif "assertTrue" in java_assertion or "assertFalse" in java_assertion:
            return "junit_boolean"
        elif "assertNull" in java_assertion or "assertNotNull" in java_assertion:
            return "junit_null"
        elif "assertThrows" in java_assertion:
            return "junit_exception"
        else:
            return "custom"

    def _find_equivalent_python_assertion(
        self, java_assertion: str, python_assertions: list[str]
    ) -> str | None:
        """Find equivalent Python assertion for a Java assertion."""
        # Simple heuristic matching based on assertion content
        java_content = self._extract_assertion_content(java_assertion)

        for python_assertion in python_assertions:
            python_content = self._extract_assertion_content(python_assertion)

            # Check for content similarity
            if self._calculate_content_similarity(java_content, python_content) > 0.7:
                return python_assertion

        return None

    def _extract_assertion_content(self, assertion: str) -> str:
        """Extract the core content from an assertion statement."""
        # Remove assertion method names and focus on the actual values being tested
        content = re.sub(r"assert\w*\s*\(", "", assertion)
        content = re.sub(r"\)$", "", content)
        content = re.sub(r"assert\s+", "", content)
        return content.strip()

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two assertion contents."""
        # Use difflib to calculate similarity ratio
        return difflib.SequenceMatcher(None, content1, content2).ratio()

    def _check_semantic_equivalence(self, java_assertion: str, python_assertion: str) -> bool:
        """Check if Java and Python assertions are semantically equivalent."""
        # Extract and compare the logical structure
        java_logic = self._extract_assertion_logic(java_assertion)
        python_logic = self._extract_assertion_logic(python_assertion)

        return java_logic == python_logic

    def _extract_assertion_logic(self, assertion: str) -> str:
        """Extract the logical structure of an assertion."""
        # Normalize assertion to focus on logical structure
        logic = assertion.lower()
        logic = re.sub(r"\s+", " ", logic)
        logic = re.sub(r'["\'].*?["\']', "STRING", logic)  # Replace string literals
        logic = re.sub(r"\d+", "NUMBER", logic)  # Replace numbers
        return logic.strip()
