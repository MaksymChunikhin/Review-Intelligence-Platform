"""Project path discovery utilities."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing the project marker files."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "PLAN.md").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the Review Intelligence project root")
