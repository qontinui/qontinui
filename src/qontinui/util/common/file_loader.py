"""
File loading utilities for Python code execution.

Provides secure file loading with path validation, preventing directory
traversal and unauthorized file access for local automation workflows.
"""

import re
from pathlib import Path


class FilePathValidator:
    """Validates file paths for security.

    Prevents:
    - Directory traversal (../)
    - Absolute paths
    - Access outside project directory
    - Non-Python files
    """

    # Dangerous path patterns
    DANGEROUS_PATTERNS = [
        r"\.\.",  # Parent directory reference
        r"^/",  # Absolute path (Unix)
        r"^[A-Za-z]:",  # Absolute path (Windows)
        r"~",  # Home directory
        r"\$",  # Environment variables
    ]

    @classmethod
    def validate_path(cls, file_path: str, project_root: Path | None = None) -> Path:
        """Validate and normalize file path.

        Args:
            file_path: Relative path to validate
            project_root: Project root directory (optional)

        Returns:
            Normalized absolute path

        Raises:
            ValueError: If path is invalid or dangerous
            FileNotFoundError: If file doesn't exist
        """
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, file_path):
                raise ValueError(f"Invalid file path: contains forbidden pattern '{pattern}'")

        # Ensure .py extension
        if not file_path.endswith(".py"):
            raise ValueError("File path must have .py extension")

        # Determine base path
        if project_root:
            base_path = project_root.resolve()
        else:
            base_path = Path.cwd()

        # Construct full path
        full_path = (base_path / file_path).resolve()

        # Validate path is within project directory
        if project_root:
            try:
                full_path.relative_to(base_path)
            except ValueError as e:
                raise ValueError("File path must be within project directory") from e

        # Check if file exists
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if it's a file (not directory)
        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        return full_path


class PythonFileLoader:
    """Loads and caches Python files for execution."""

    def __init__(self, project_root: Path | None = None):
        """Initialize file loader.

        Args:
            project_root: Root directory for file resolution
        """
        self.project_root = project_root or Path.cwd()
        self.validator = FilePathValidator()
        self._cache: dict[str, str] = {}  # Simple in-memory cache

    def load_file(self, file_path: str, use_cache: bool = True) -> str:
        """Load Python file content.

        Args:
            file_path: Relative path to Python file
            use_cache: Whether to use cached content

        Returns:
            File content as string

        Raises:
            ValueError: If file is invalid
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file has encoding issues
        """
        # Check cache first
        if use_cache and file_path in self._cache:
            return self._cache[file_path]

        # Validate path
        absolute_path = self.validator.validate_path(file_path, project_root=self.project_root)

        # Load file content
        try:
            with open(absolute_path, encoding="utf-8") as f:
                content = f.read()

            # Cache content
            if use_cache:
                self._cache[file_path] = content

            return content

        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding, e.object, e.start, e.end, f"File encoding error: {str(e)}"
            ) from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied: {file_path}") from e

    def clear_cache(self, file_path: str | None = None):
        """Clear file cache.

        Args:
            file_path: Specific file to clear, or None to clear all
        """
        if file_path:
            self._cache.pop(file_path, None)
        else:
            self._cache.clear()

    def list_python_files(self, directory: str = ".") -> list[str]:
        """List all Python files in directory (recursive).

        Args:
            directory: Directory to search (relative to project root)

        Returns:
            List of relative file paths

        Raises:
            ValueError: If directory is invalid
        """
        # Validate directory path
        dir_path = self.project_root / directory

        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Find all .py files
        python_files = []
        for file_path in dir_path.rglob("*.py"):
            # Make path relative to project root
            relative_path = file_path.relative_to(self.project_root)
            python_files.append(str(relative_path))

        return sorted(python_files)
