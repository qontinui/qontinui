"""
Data models for fix suggestion engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.models import FailureType


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
