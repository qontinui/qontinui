"""
Python wrapper for the Node.js TypeScript/JavaScript parser.

This module provides a Python interface to the TypeScript parser that uses
Node.js and Babel to analyze TypeScript and JavaScript source code.
"""

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PropInfo:
    """Information about a component prop."""

    name: str
    default: str | None = None


@dataclass
class ComponentInfo:
    """Information about a React component."""

    name: str
    type: str  # 'function', 'arrow_function', or 'class'
    line: int
    props: list[PropInfo]
    children: list[str]  # Names of child components
    returns_jsx: bool = True
    extends: str | None = None  # For class components


@dataclass
class StateVariableInfo:
    """Information about a React state variable."""

    name: str | None
    hook: str  # 'useState', 'useReducer', 'useContext', etc.
    line: int
    initial_value: str | None = None
    type: str = "unknown"
    setter: str | None = None


@dataclass
class ConditionalRenderInfo:
    """Information about conditional rendering."""

    condition: str
    line: int
    pattern: str  # 'AND', 'TERNARY', 'EARLY_RETURN'
    renders: list[str] = field(default_factory=list)
    renders_true: list[str] = field(default_factory=list)
    renders_false: list[str] = field(default_factory=list)


@dataclass
class EventHandlerInfo:
    """Information about an event handler."""

    event: str  # 'click', 'change', etc.
    line: int
    name: str | None = None
    state_changes: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import statement."""

    source: str
    specifiers: list[dict[str, str]]
    line: int


@dataclass
class ExportInfo:
    """Information about an export statement."""

    type: str  # 'named' or 'default'
    name: str
    line: int


@dataclass
class JSXElementInfo:
    """Information about a JSX element."""

    name: str
    line: int
    props: list[dict[str, Any]]
    self_closing: bool


@dataclass
class NavigationLinkInfo:
    """Information about a navigation link."""

    type: str  # 'link' or 'anchor'
    target: str  # The href destination
    line: int
    component: str | None = None  # The component containing this link


@dataclass
class FileParseResult:
    """Result of parsing a single file."""

    components: list[ComponentInfo] = field(default_factory=list)
    state_variables: list[StateVariableInfo] = field(default_factory=list)
    conditional_renders: list[ConditionalRenderInfo] = field(default_factory=list)
    event_handlers: list[EventHandlerInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    exports: list[ExportInfo] = field(default_factory=list)
    jsx_elements: list[JSXElementInfo] = field(default_factory=list)
    navigation_links: list[NavigationLinkInfo] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for downstream processing."""
        from dataclasses import asdict

        return {
            "components": [asdict(c) for c in self.components],
            "state_variables": [asdict(s) for s in self.state_variables],
            "conditional_renders": [asdict(c) for c in self.conditional_renders],
            "event_handlers": [asdict(h) for h in self.event_handlers],
            "imports": [asdict(i) for i in self.imports],
            "exports": [asdict(e) for e in self.exports],
            "jsx_elements": [asdict(j) for j in self.jsx_elements],
            "navigation_links": [asdict(n) for n in self.navigation_links],
            "function_declarations": [
                {"name": c.name, "line": c.line, "async": False, "parameters": []}
                for c in self.components
            ],
            "error": self.error,
        }


@dataclass
class ParseResult:
    """Result of parsing multiple files."""

    files: dict[str, FileParseResult]
    errors: list[str] = field(default_factory=list)


class TypeScriptParser:
    """
    Python wrapper for the Node.js TypeScript parser.

    This class provides a Python interface to analyze TypeScript and JavaScript
    files using the Node.js-based parser that leverages Babel and the TypeScript
    compiler API.
    """

    def __init__(self, node_path: str = "node"):
        """
        Initialize the TypeScript parser.

        Args:
            node_path: Path to the Node.js executable (default: "node")
        """
        self.node_path = node_path
        self.parser_script = Path(__file__).parent / "parser.js"

        if not self.parser_script.exists():
            raise FileNotFoundError(f"Parser script not found at {self.parser_script}")

    async def parse_files(
        self,
        files: list[Path],
        extract: list[str] | None = None,
    ) -> ParseResult:
        """
        Parse TypeScript/JavaScript files and extract information.

        Args:
            files: List of file paths to parse
            extract: List of what to extract. Options:
                - 'components': Component definitions
                - 'state': React hooks and state
                - 'conditionals': Conditional rendering patterns
                - 'handlers': Event handlers
                - 'imports': Import/export statements

        Returns:
            ParseResult containing extracted information from all files

        Raises:
            subprocess.CalledProcessError: If the parser fails
            json.JSONDecodeError: If the parser output is invalid
        """
        if extract is None:
            extract = ["components", "state", "conditionals", "handlers", "imports"]

        # Use a temporary file to avoid stdout buffer limits
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output_file:
            output_path = output_file.name

        try:
            config = {
                "files": [str(f.resolve()) for f in files],
                "extract": extract,
                "outputFile": output_path,
            }

            # Run the Node.js parser
            process = await asyncio.create_subprocess_exec(
                self.node_path,
                str(self.parser_script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Send configuration to stdin
            config_json = json.dumps(config).encode("utf-8")
            stdout, stderr = await process.communicate(input=config_json)

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                raise RuntimeError(
                    f"Parser failed with exit code {process.returncode}: {error_msg}"
                )

            # Read results from the output file
            with open(output_path, encoding="utf-8") as f:
                result_data = json.load(f)

            return self._convert_to_dataclasses(result_data)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to run Node.js parser: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse parser output: {e}") from e
        finally:
            # Clean up the temporary file
            try:
                Path(output_path).unlink()
            except Exception:
                pass

    async def parse_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude: list[str] | None = None,
        extract: list[str] | None = None,
    ) -> ParseResult:
        """
        Parse all matching files in a directory.

        Args:
            directory: Directory to search for files
            patterns: File patterns to include (e.g., ['*.ts', '*.tsx'])
            exclude: Patterns to exclude (e.g., ['node_modules/**', '*.test.ts'])
            extract: List of what to extract (see parse_files)

        Returns:
            ParseResult containing extracted information from all matching files
        """
        if patterns is None:
            patterns = ["*.ts", "*.tsx", "*.js", "*.jsx"]

        if exclude is None:
            exclude = [
                "node_modules/**",
                "dist/**",
                "build/**",
                "*.test.ts",
                "*.test.tsx",
                "*.test.js",
                "*.test.jsx",
                "*.spec.ts",
                "*.spec.tsx",
                "*.spec.js",
                "*.spec.jsx",
            ]

        # Find all matching files
        files = []
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                # Check if file should be excluded
                should_exclude = False
                for exclude_pattern in exclude:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break

                if not should_exclude and file_path.is_file():
                    files.append(file_path)

        if not files:
            return ParseResult(files={})

        return await self.parse_files(files, extract)

    def parse_files_sync(
        self,
        files: list[Path],
        extract: list[str] | None = None,
    ) -> ParseResult:
        """
        Synchronous version of parse_files.

        Args:
            files: List of file paths to parse
            extract: List of what to extract (see parse_files)

        Returns:
            ParseResult containing extracted information from all files
        """
        return asyncio.run(self.parse_files(files, extract))

    def parse_directory_sync(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude: list[str] | None = None,
        extract: list[str] | None = None,
    ) -> ParseResult:
        """
        Synchronous version of parse_directory.

        Args:
            directory: Directory to search for files
            patterns: File patterns to include
            exclude: Patterns to exclude
            extract: List of what to extract (see parse_files)

        Returns:
            ParseResult containing extracted information from all matching files
        """
        return asyncio.run(self.parse_directory(directory, patterns, exclude, extract))

    async def parse_file(
        self,
        file_path: str,
        extract: list[str] | None = None,
    ) -> FileParseResult:
        """
        Parse a single TypeScript/JavaScript file.

        Args:
            file_path: Path to the file to parse
            extract: List of what to extract (see parse_files)

        Returns:
            FileParseResult containing extracted information from the file
        """
        path = Path(file_path)
        result = await self.parse_files([path], extract)

        # Return the FileParseResult for this file, or an empty one if not found
        return result.files.get(str(path.resolve()), FileParseResult())

    def parse_file_sync(
        self,
        file_path: str,
        extract: list[str] | None = None,
    ) -> FileParseResult:
        """
        Synchronous version of parse_file.

        Args:
            file_path: Path to the file to parse
            extract: List of what to extract (see parse_files)

        Returns:
            FileParseResult containing extracted information from the file
        """
        return asyncio.run(self.parse_file(file_path, extract))

    def _convert_to_dataclasses(self, result_data: dict) -> ParseResult:
        """
        Convert raw JSON result to typed dataclasses.

        Args:
            result_data: Raw JSON data from the parser

        Returns:
            ParseResult with properly typed data
        """
        files = {}
        errors = []

        for file_path, file_data in result_data.get("files", {}).items():
            # Check for file-level errors
            if "error" in file_data and file_data["error"]:
                errors.append(f"{file_path}: {file_data['error']}")

            # Convert components
            components = [
                ComponentInfo(
                    name=c["name"],
                    type=c["type"],
                    line=c["line"],
                    props=[PropInfo(**p) for p in c.get("props", [])],
                    children=c.get("children", []),
                    returns_jsx=c.get("returns_jsx", True),
                    extends=c.get("extends"),
                )
                for c in file_data.get("components", [])
            ]

            # Convert state variables
            state_variables = [
                StateVariableInfo(
                    name=s.get("name"),
                    hook=s["hook"],
                    line=s["line"],
                    initial_value=s.get("initial_value"),
                    type=s.get("type", "unknown"),
                    setter=s.get("setter"),
                )
                for s in file_data.get("state_variables", [])
            ]

            # Convert conditional renders
            conditional_renders = [
                ConditionalRenderInfo(
                    condition=c["condition"],
                    line=c["line"],
                    pattern=c["pattern"],
                    renders=c.get("renders", []),
                    renders_true=c.get("renders_true", []),
                    renders_false=c.get("renders_false", []),
                )
                for c in file_data.get("conditional_renders", [])
            ]

            # Convert event handlers
            event_handlers = [
                EventHandlerInfo(
                    event=h["event"],
                    line=h["line"],
                    name=h.get("name"),
                    state_changes=h.get("state_changes", []),
                )
                for h in file_data.get("event_handlers", [])
            ]

            # Convert imports
            imports = [
                ImportInfo(
                    source=i["source"],
                    specifiers=i["specifiers"],
                    line=i["line"],
                )
                for i in file_data.get("imports", [])
            ]

            # Convert exports
            exports = [
                ExportInfo(
                    type=e["type"],
                    name=e["name"],
                    line=e["line"],
                )
                for e in file_data.get("exports", [])
            ]

            # Convert JSX elements
            jsx_elements = [
                JSXElementInfo(
                    name=j["name"],
                    line=j["line"],
                    props=j.get("props", []),
                    self_closing=j.get("self_closing", False),
                )
                for j in file_data.get("jsx_elements", [])
            ]

            # Convert navigation links
            navigation_links = [
                NavigationLinkInfo(
                    type=n["type"],
                    target=n["target"],
                    line=n["line"],
                    component=n.get("component"),
                )
                for n in file_data.get("navigation_links", [])
            ]

            files[file_path] = FileParseResult(
                components=components,
                state_variables=state_variables,
                conditional_renders=conditional_renders,
                event_handlers=event_handlers,
                imports=imports,
                exports=exports,
                jsx_elements=jsx_elements,
                navigation_links=navigation_links,
                error=file_data.get("error"),
            )

        return ParseResult(files=files, errors=errors)


def create_parser(node_path: str = "node") -> TypeScriptParser:
    """
    Create a new TypeScript parser instance.

    Args:
        node_path: Path to the Node.js executable

    Returns:
        TypeScriptParser instance
    """
    return TypeScriptParser(node_path)
