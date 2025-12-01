"""
Pages Router (pages/) extraction for Next.js.

Handles the traditional Pages Router including:
- Index routes (index.tsx)
- Dynamic routes: [id].tsx, [...slug].tsx
- API routes: pages/api/
- _app.tsx and _document.tsx
- Data fetching: getServerSideProps, getStaticProps, getStaticPaths
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qontinui.extraction.models.static import (
        APICallDefinition,
        RouteDefinition,
    )


# Dynamic route patterns
DYNAMIC_SEGMENT_PATTERN = re.compile(r"\[([^\]]+)\]")
CATCH_ALL_PATTERN = re.compile(r"\[\.\.\.([^\]]+)\]")


def extract_pages_routes(pages_dir: Path) -> list["RouteDefinition"]:
    """
    Extract routes from pages/ directory structure.

    Handles:
    - Index routes (index.tsx)
    - Dynamic routes: [id].tsx, [...slug].tsx
    - API routes: pages/api/
    - _app.tsx and _document.tsx

    Args:
        pages_dir: Path to the pages/ directory

    Returns:
        List of route definitions
    """
    from qontinui.extraction.models.static import RouteDefinition

    routes = []

    if not pages_dir.exists():
        return routes

    # Process all page files
    for extension in ["tsx", "ts", "jsx", "js"]:
        for file_path in pages_dir.glob(f"**/*.{extension}"):
            # Skip special files
            if file_path.name.startswith("_"):
                continue

            # Check if it's an API route
            is_api_route = _is_api_route(file_path, pages_dir)

            route_path = _pages_file_to_route(file_path, pages_dir)
            dynamic_segments = _parse_pages_dynamic_segments(file_path)
            is_catch_all = bool(CATCH_ALL_PATTERN.search(file_path.stem))

            route = RouteDefinition(
                path=route_path,
                file_path=file_path,
                method="GET" if not is_api_route else "ALL",
                dynamic_segments=dynamic_segments,
                is_catch_all=is_catch_all,
                is_optional_catch_all=False,
                route_type="api" if is_api_route else "page",
                is_server_component=False,  # Pages Router uses client-side rendering
                metadata={
                    "router": "pages",
                    "is_api": is_api_route,
                    "relative_path": str(file_path.relative_to(pages_dir)),
                },
            )
            routes.append(route)

    # Add API routes separately
    api_dir = pages_dir / "api"
    if api_dir.exists():
        api_routes = extract_api_routes(api_dir)
        routes.extend(api_routes)

    return routes


def extract_get_server_side_props(parse_results: dict) -> list["APICallDefinition"]:
    """
    Extract getServerSideProps data fetching.

    getServerSideProps runs on the server on every request.

    Args:
        parse_results: Parsed AST results

    Returns:
        List of API call definitions for getServerSideProps
    """

    calls = []

    # Placeholder - real implementation would:
    # 1. Look for exported function named "getServerSideProps"
    # 2. Extract any fetch/API calls within it
    # 3. Extract props returned

    return calls


def extract_get_static_props(parse_results: dict) -> list["APICallDefinition"]:
    """
    Extract getStaticProps data fetching.

    getStaticProps runs at build time to pre-render pages.

    Args:
        parse_results: Parsed AST results

    Returns:
        List of API call definitions for getStaticProps
    """

    calls = []

    # Placeholder - real implementation would:
    # 1. Look for exported function named "getStaticProps"
    # 2. Extract any fetch/API calls within it
    # 3. Extract props returned

    return calls


def extract_api_routes(api_dir: Path) -> list["RouteDefinition"]:
    """
    Extract API route definitions from pages/api/.

    Args:
        api_dir: Path to the pages/api/ directory

    Returns:
        List of API route definitions
    """
    from qontinui.extraction.models.static import RouteDefinition

    routes = []

    if not api_dir.exists():
        return routes

    for extension in ["tsx", "ts", "jsx", "js"]:
        for file_path in api_dir.glob(f"**/*.{extension}"):
            route_path = _api_file_to_route(file_path, api_dir)
            dynamic_segments = _parse_pages_dynamic_segments(file_path)
            is_catch_all = bool(CATCH_ALL_PATTERN.search(file_path.stem))

            route = RouteDefinition(
                path=route_path,
                file_path=file_path,
                method="ALL",  # API routes can handle multiple methods
                dynamic_segments=dynamic_segments,
                is_catch_all=is_catch_all,
                is_optional_catch_all=False,
                route_type="api",
                is_server_component=False,
                metadata={
                    "router": "pages",
                    "is_api": True,
                    "relative_path": str(file_path.relative_to(api_dir.parent)),
                },
            )
            routes.append(route)

    return routes


def _is_api_route(file_path: Path, pages_dir: Path) -> bool:
    """
    Check if a file is an API route.

    Args:
        file_path: Path to the file
        pages_dir: Path to the pages/ directory

    Returns:
        True if file is in pages/api/ directory
    """
    try:
        relative_path = file_path.relative_to(pages_dir)
        return relative_path.parts[0] == "api"
    except ValueError:
        return False


def _pages_file_to_route(file_path: Path, pages_dir: Path) -> str:
    """
    Convert pages/ file path to route path.

    Examples:
    - pages/index.tsx -> /
    - pages/about.tsx -> /about
    - pages/users/[id].tsx -> /users/[id]
    - pages/blog/[...slug].tsx -> /blog/[...slug]

    Args:
        file_path: Path to the file
        pages_dir: Path to the pages/ directory

    Returns:
        Route path string
    """
    relative_path = file_path.relative_to(pages_dir)

    # Remove extension
    path_without_ext = relative_path.with_suffix("")

    # Get parts
    parts = list(path_without_ext.parts)

    # Handle index.tsx -> /
    if parts[-1] == "index":
        parts = parts[:-1]

    # Build route
    if not parts:
        return "/"

    route_path = "/" + "/".join(parts)
    return route_path


def _api_file_to_route(file_path: Path, api_dir: Path) -> str:
    """
    Convert pages/api/ file path to API route path.

    Examples:
    - pages/api/users.ts -> /api/users
    - pages/api/users/[id].ts -> /api/users/[id]

    Args:
        file_path: Path to the file
        api_dir: Path to the pages/api/ directory

    Returns:
        API route path string
    """
    relative_path = file_path.relative_to(api_dir)

    # Remove extension
    path_without_ext = relative_path.with_suffix("")

    # Get parts
    parts = list(path_without_ext.parts)

    # Handle index files
    if parts[-1] == "index":
        parts = parts[:-1]

    # Build route
    if not parts:
        return "/api"

    route_path = "/api/" + "/".join(parts)
    return route_path


def _parse_pages_dynamic_segments(file_path: Path) -> list[str]:
    """
    Parse dynamic segments from pages file path.

    Args:
        file_path: Path to the file

    Returns:
        List of dynamic segment names
    """
    dynamic_segments = []

    # Get the file name without extension
    file_stem = file_path.stem

    # Check for catch-all [...slug]
    catch_all_match = CATCH_ALL_PATTERN.match(file_stem)
    if catch_all_match:
        segment_name = catch_all_match.group(1)
        dynamic_segments.append(segment_name)
        return dynamic_segments

    # Check for regular dynamic segment [id]
    dynamic_match = DYNAMIC_SEGMENT_PATTERN.match(file_stem)
    if dynamic_match:
        segment_name = dynamic_match.group(1)
        dynamic_segments.append(segment_name)
        return dynamic_segments

    # Check parent directories for dynamic segments
    for part in file_path.parts:
        catch_all_match = CATCH_ALL_PATTERN.match(part)
        if catch_all_match:
            dynamic_segments.append(catch_all_match.group(1))
            continue

        dynamic_match = DYNAMIC_SEGMENT_PATTERN.match(part)
        if dynamic_match:
            dynamic_segments.append(dynamic_match.group(1))

    return dynamic_segments
