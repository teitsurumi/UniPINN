"""Workspace and result directory management.

Handles auto-incrementing result directories and experiment workspace setup.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the project root directory (parent of unipinn/ package)."""
    return Path(__file__).resolve().parent.parent.parent


def get_results_dir(problem_name: str, framework_name: str,
                    base_dir: Optional[Path] = None) -> Path:
    """Get or create the results directory for a problem/framework combination.

    Args:
        problem_name: e.g., 'poisson1d'
        framework_name: e.g., '01_pinn'
        base_dir: Override base results directory. Defaults to <project>/.tmp_results/

    Returns:
        Path to results directory (e.g., .tmp_results/poisson1d/01_pinn/).
    """
    if base_dir is None:
        base_dir = get_project_root() / ".tmp_results"
    results = base_dir / problem_name / framework_name
    results.mkdir(parents=True, exist_ok=True)
    return results


def get_next_test_case_dir(problem_name: str, framework_name: str,
                           base_dir: Optional[Path] = None) -> Path:
    """Auto-increment and return the next available test_case directory.

    Scans for existing test_case_* directories and returns the next one.

    Returns:
        Path like .tmp_results/poisson1d/01_pinn/test_case_0/
    """
    results = get_results_dir(problem_name, framework_name, base_dir)

    existing = list(results.glob("test_case_*"))
    if not existing:
        case_dir = results / "test_case_0"
    else:
        max_i = -1
        for folder in existing:
            try:
                i = int(folder.name.split("_")[-1])
                max_i = max(max_i, i)
            except ValueError:
                continue
        case_dir = results / f"test_case_{max_i + 1}"

    return case_dir
