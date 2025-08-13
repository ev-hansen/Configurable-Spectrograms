import pathlib
import re

BARE_EXCEPTION_PATTERN = re.compile(r"^\s*except Exception:\s*$")
EXCLUDED_DIR_NAMES = {
    "backup of old files",
    "prototyping  (ipynb etc)",
    "prototyping (ipynb etc)",
    "__pycache__",
    ".idea",
    ".ipynb_checkpoints",
    ".spyproject",
    ".vscode",
    "animations",
    "emma_data",
    "FAST_data",
    "FAST_plots",
    "plots"
}


def test_no_bare_except():
    """Fail if any production Python file contains an unbound 'except Exception:' clause.

    This acts as a lightweight lint enforcing descriptive exception variable naming.
    Directories considered archival or prototyping are excluded.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for py_path in repo_root.rglob("*.py"):
        # Skip excluded directories
        if any(part in EXCLUDED_DIR_NAMES for part in py_path.parts):
            continue
        # Skip tests themselves (this file) to avoid self-reference noise
        if py_path.name.startswith("test_"):
            continue
        try:
            lines = py_path.read_text().splitlines()
        except Exception as test_file_read_exception:  # pragma: no cover - read error edge case
            violations.append(f"{py_path}: read error: {test_file_read_exception}")
            continue
        for lineno, line in enumerate(lines, start=1):
            if BARE_EXCEPTION_PATTERN.search(line):
                violations.append(
                    f"{py_path.relative_to(repo_root)}:{lineno}: bare 'except Exception:' without variable name"
                )
    if violations:
        raise AssertionError("\n".join(violations))
