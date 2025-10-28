"""
Output validation and comparison strategy.

This module implements validation logic for comparing test outputs,
calculating similarity scores, and finding specific differences.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import TestResult
else:
    try:
        from ...core.models import TestResult
    except ImportError:
        pass


class OutputValidator:
    """
    Validates and compares test outputs for equivalence.

    This validator handles output normalization, difference detection,
    and similarity score calculation.
    """

    def __init__(self, ignore_whitespace: bool = True, ignore_case: bool = False) -> None:
        """
        Initialize the output validator.

        Args:
            ignore_whitespace: Whether to ignore whitespace differences
            ignore_case: Whether to ignore case differences
        """
        self.ignore_whitespace = ignore_whitespace
        self.ignore_case = ignore_case

    def normalize_output(self, output: str) -> str:
        """
        Normalize output for comparison.

        Args:
            output: Raw output string

        Returns:
            Normalized output string
        """
        if self.ignore_whitespace:
            output = " ".join(output.split())

        if self.ignore_case:
            output = output.lower()

        return output.strip()

    def compare_outputs(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str], float]:
        """
        Compare Java and Python test outputs.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (validation_result, differences, similarity_score)
            validation_result is one of: "equivalent", "different"
        """
        java_output = self.normalize_output(java_result.output)
        python_output = self.normalize_output(python_result.output)

        if java_output == python_output:
            return "equivalent", [], 1.0

        differences = self.find_output_differences(java_output, python_output)
        similarity_score = self.calculate_similarity_score(java_output, python_output)

        return "different", differences, similarity_score

    def find_output_differences(self, output1: str, output2: str) -> list[str]:
        """
        Find specific differences between outputs.

        Args:
            output1: First normalized output
            output2: Second normalized output

        Returns:
            List of difference descriptions (limited to first 10)
        """
        differences = []

        lines1 = output1.split("\n")
        lines2 = output2.split("\n")

        if len(lines1) != len(lines2):
            differences.append(f"Line count differs: {len(lines1)} vs {len(lines2)}")

        max_lines = max(len(lines1), len(lines2))
        for i in range(max_lines):
            line1 = lines1[i] if i < len(lines1) else ""
            line2 = lines2[i] if i < len(lines2) else ""

            if line1 != line2:
                differences.append(f"Line {i+1}: '{line1}' vs '{line2}'")

        return differences[:10]  # Limit to first 10 differences

    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts.

        Uses character-based matching at same positions.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 and not text2:
            return 1.0

        if not text1 or not text2:
            return 0.0

        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        # Count matching characters at same positions
        matches = sum(1 for i in range(min(len(text1), len(text2))) if text1[i] == text2[i])

        return matches / max_len

    def get_normalized_outputs(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, str]:
        """
        Get normalized outputs for both Java and Python results.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (normalized_java_output, normalized_python_output)
        """
        return (
            self.normalize_output(java_result.output),
            self.normalize_output(python_result.output),
        )
