#!/usr/bin/env python3
"""Fix imports after restructuring to match Brobot's model/* structure."""

import re
from pathlib import Path

# Define the base path
BASE_PATH = Path("/home/jspinak/qontinui_parent_directory/qontinui-core/src/qontinui")

# Define import mappings
IMPORT_MAPPINGS = [
    # Old datatypes imports -> new model.element imports
    (r"from \.\.datatypes import", "from ..element import"),
    (r"from \.datatypes import", "from .element import"),
    (r"from qontinui\.datatypes import", "from qontinui.model.element import"),
    # Fix state imports
    (r"from \.\.state import", "from ..model.state import"),
    (r"from \.state import", "from .model.state import"),
    (r"from qontinui\.state import", "from qontinui.model.state import"),
    # Fix relative imports in model/state files
    (r"from \.state_transition import", "from ..transition.state_transition import"),
    (r"from \.state_transitions import", "from ..transition.state_transitions import"),
    # Fix find imports
    (r"from \.\.find import", "from ...find import"),  # From model/state to find
    (r"from qontinui\.find import", "from qontinui.find import"),  # Keep absolute
    # Fix element-specific imports
    (r"from \.location import", "from ..element.location import"),
    (r"from \.region import", "from ..element.region import"),
    (r"from \.pattern import", "from ..element.pattern import"),
    (r"from \.image import", "from ..element.image import"),
    (r"from \.color import", "from ..element.color import"),
    # Fix match imports
    (r"from \.match import MatchObject", "from ..match.match import MatchObject"),
    (r"from \.\.datatypes import MatchObject", "from ..match import MatchObject"),
    (
        r"from qontinui\.datatypes import MatchObject",
        "from qontinui.model.match import MatchObject",
    ),
]


def fix_file_imports(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path) as f:
            content = f.read()

        original = content
        for old_pattern, new_pattern in IMPORT_MAPPINGS:
            content = re.sub(old_pattern, new_pattern, content)

        if content != original:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False


def main():
    """Fix all Python files in the project."""
    fixed_count = 0

    # Process all Python files
    for py_file in BASE_PATH.rglob("*.py"):
        if "fix_imports.py" in str(py_file):
            continue
        if fix_file_imports(py_file):
            fixed_count += 1

    print(f"\nFixed imports in {fixed_count} files")


if __name__ == "__main__":
    main()
