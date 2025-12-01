"""
App Router (app/) extraction for Next.js 13+.

Handles the new App Router file-system based routing including:
- page.tsx, layout.tsx, loading.tsx, error.tsx, not-found.tsx
- Dynamic routes: [id], [...slug], [[...slug]]
- Route groups: (group)
- Parallel routes: @modal
- Server Components and Server Actions
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qontinui.extraction.models.static import (
        APICallDefinition,
        ComponentDefinition,
        RouteDefinition,
    )


# Next.js App Router special file names
SPECIAL_FILES = {
    "page": "page",
    "layout": "layout",
    "loading": "loading",
    "error": "error",
    "not-found": "not-found",
    "template": "template",
    "default": "default",
}

# Dynamic route patterns
DYNAMIC_SEGMENT_PATTERN = re.compile(r"\[([^\]]+)\]")
CATCH_ALL_PATTERN = re.compile(r"\[\.\.\.([^\]]+)\]")
OPTIONAL_CATCH_ALL_PATTERN = re.compile(r"\[\[\.\.\.([^\]]+)\]\]")
ROUTE_GROUP_PATTERN = re.compile(r"\(([^)]+)\)")
PARALLEL_ROUTE_PATTERN = re.compile(r"@([^/]+)")


def extract_app_routes(app_dir: Path) -> list["RouteDefinition"]:
    """
    Extract routes from app/ directory structure.

    Handles:
    - page.tsx files
    - layout.tsx files
    - loading.tsx files
    - error.tsx files
    - not-found.tsx files
    - Dynamic routes: [id], [...slug], [[...slug]]
    - Route groups: (group)
    - Parallel routes: @modal

    Args:
        app_dir: Path to the app/ directory

    Returns:
        List of route definitions
    """
    from qontinui.extraction.models.static import RouteDefinition, RouteType

    routes: list[RouteDefinition] = []

    if not app_dir.exists():
        return routes

    # Find all special files in app directory
    for extension in ["tsx", "ts", "jsx", "js"]:
        for special_file in SPECIAL_FILES.keys():
            pattern = f"**/{special_file}.{extension}"
            for file_path in app_dir.glob(pattern):
                route_path = parse_route_path(file_path, app_dir)
                dynamic_segments, is_catch_all, is_optional_catch_all = parse_dynamic_segments(
                    file_path, app_dir
                )

                # Map special file names to RouteType
                if special_file == "layout":
                    route_type = RouteType.LAYOUT
                else:
                    route_type = RouteType.PAGE

                route = RouteDefinition(
                    id=f"app_{special_file}_{route_path.replace('/', '_')}",
                    path=route_path,
                    file_path=file_path,
                    route_type=route_type,
                    metadata={
                        "router": "app",
                        "relative_path": str(file_path.relative_to(app_dir)),
                        "special_file": special_file,
                        "method": "GET",
                        "dynamic_segments": dynamic_segments,
                        "is_catch_all": is_catch_all,
                        "is_optional_catch_all": is_optional_catch_all,
                        "is_server_component": True,  # Default in App Router
                    },
                )
                routes.append(route)

    return routes


def extract_server_components(app_dir: Path, parse_results: dict) -> list["ComponentDefinition"]:
    """
    Identify server components (default in app/).

    In Next.js App Router, all components are Server Components by default
    unless they use 'use client' directive.

    Args:
        app_dir: Path to the app/ directory
        parse_results: Parsed AST results

    Returns:
        List of server component definitions
    """
    from qontinui.extraction.models.static import ComponentDefinition, ComponentType

    components: list[ComponentDefinition] = []

    if not app_dir.exists():
        return components

    # Find all component files in app directory
    for extension in ["tsx", "ts", "jsx", "js"]:
        for file_path in app_dir.glob(f"**/*.{extension}"):
            # Check if file has 'use client' directive
            has_use_client = _has_use_client_directive(file_path)

            if not has_use_client:
                # This is a server component
                component_name = file_path.stem
                if component_name in SPECIAL_FILES:
                    # Use parent directory name for special files
                    parent = file_path.parent
                    if parent != app_dir:
                        component_name = f"{parent.name}_{component_name}"

                component = ComponentDefinition(
                    id=f"server_{component_name}_{file_path.parent.name}",
                    name=component_name,
                    file_path=file_path,
                    line_number=1,  # Server components start at line 1
                    component_type=ComponentType.SERVER,
                    framework="nextjs",
                    metadata={
                        "router": "app",
                        "type": "server_component",
                        "is_default_export": True,
                        "is_server_component": True,
                    },
                )
                components.append(component)

    return components


def extract_server_actions(parse_results: dict) -> list["APICallDefinition"]:
    """
    Extract 'use server' functions (Server Actions).

    Server Actions are async functions marked with 'use server' directive
    that can be called from client components.

    Args:
        parse_results: Parsed AST results

    Returns:
        List of server action definitions
    """
    from qontinui.extraction.models.static import APICallDefinition

    actions: list[APICallDefinition] = []

    # Placeholder - real implementation would:
    # 1. Look for 'use server' directive at top of file
    # 2. Look for 'use server' inside async functions
    # 3. Extract function signatures and parameters

    return actions


def parse_route_path(file_path: Path, app_dir: Path) -> str:
    """
    Convert file path to route path.

    Examples:
    - app/page.tsx -> /
    - app/about/page.tsx -> /about
    - app/users/[id]/page.tsx -> /users/[id]
    - app/blog/[...slug]/page.tsx -> /blog/[...slug]
    - app/(marketing)/about/page.tsx -> /about (route groups ignored)
    - app/@modal/login/page.tsx -> /@modal/login (parallel routes preserved)

    Args:
        file_path: Path to the file
        app_dir: Path to the app/ directory

    Returns:
        Route path string
    """
    # Get relative path from app directory
    relative_path = file_path.relative_to(app_dir)

    # Remove the file name (page.tsx, layout.tsx, etc.)
    path_parts = list(relative_path.parts[:-1])

    # Process each part
    processed_parts = []
    for part in path_parts:
        # Remove route groups (marketing) -> ""
        if ROUTE_GROUP_PATTERN.match(part):
            continue

        # Keep parallel routes @modal
        # Keep dynamic segments [id], [...slug], [[...slug]]
        processed_parts.append(part)

    # Build route path
    if not processed_parts:
        return "/"

    route_path = "/" + "/".join(processed_parts)
    return route_path


def parse_dynamic_segments(file_path: Path, app_dir: Path) -> tuple[list[str], bool, bool]:
    """
    Parse dynamic segments from file path.

    Args:
        file_path: Path to the file
        app_dir: Path to the app/ directory

    Returns:
        Tuple of (dynamic_segments, is_catch_all, is_optional_catch_all)
    """
    relative_path = file_path.relative_to(app_dir)
    path_parts = list(relative_path.parts[:-1])

    dynamic_segments = []
    is_catch_all = False
    is_optional_catch_all = False

    for part in path_parts:
        # Check for optional catch-all [[...slug]]
        optional_match = OPTIONAL_CATCH_ALL_PATTERN.match(part)
        if optional_match:
            segment_name = optional_match.group(1)
            dynamic_segments.append(segment_name)
            is_optional_catch_all = True
            continue

        # Check for catch-all [...slug]
        catch_all_match = CATCH_ALL_PATTERN.match(part)
        if catch_all_match:
            segment_name = catch_all_match.group(1)
            dynamic_segments.append(segment_name)
            is_catch_all = True
            continue

        # Check for regular dynamic segment [id]
        dynamic_match = DYNAMIC_SEGMENT_PATTERN.match(part)
        if dynamic_match:
            segment_name = dynamic_match.group(1)
            dynamic_segments.append(segment_name)

    return dynamic_segments, is_catch_all, is_optional_catch_all


def _has_use_client_directive(file_path: Path) -> bool:
    """
    Check if file has 'use client' directive.

    Args:
        file_path: Path to the file

    Returns:
        True if file has 'use client' directive
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            # Read first few lines
            for _ in range(10):
                line = f.readline().strip()
                if not line or line.startswith("//") or line.startswith("/*"):
                    continue
                if "'use client'" in line or '"use client"' in line:
                    return True
                # If we hit actual code, stop looking
                if line and not line.startswith("import"):
                    break
    except Exception:
        pass

    return False
