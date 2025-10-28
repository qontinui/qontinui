"""
Import fix suggestion strategies for Java/Brobot to Python/Qontinui migration.
"""

import re
from pathlib import Path

from ..fix_models import FixComplexity, FixSuggestion, FixType
from ...core.models import TestFile


class ImportSuggestionStrategy:
    """Strategy for generating import-related fix suggestions."""

    def __init__(self) -> None:
        """Initialize the import suggestion strategy."""
        self._java_to_python_mappings = self._initialize_java_to_python_mappings()
        self._brobot_to_qontinui_mappings = self._initialize_brobot_to_qontinui_mappings()

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

    def _initialize_brobot_to_qontinui_mappings(self) -> dict[str, str]:
        """Initialize Brobot to Qontinui import mappings."""
        return {
            "brobot.library.Action": "qontinui.actions.Action",
            "brobot.library.State": "qontinui.model.state.State",
            "brobot.library.Find": "qontinui.find.Find",
            "brobot.library.Image": "qontinui.model.Image",
            "brobot.library.Region": "qontinui.model.Region",
        }

    def generate_brobot_import_fix(
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

    def generate_java_import_fix(
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

    def _map_brobot_to_qontinui(self, brobot_import: str) -> str:
        """Map Brobot import to Qontinui equivalent."""
        return self._brobot_to_qontinui_mappings.get(
            brobot_import, f"qontinui.{brobot_import.split('.')[-1].lower()}"
        )

    def apply_import_fix(self, fix: FixSuggestion, content: str) -> str:
        """Apply import-related fixes."""
        # Simple string replacement for now
        # In a more sophisticated implementation, this would parse the AST
        return content.replace(fix.original_code, fix.suggested_code)
