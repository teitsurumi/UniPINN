# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Merged `batch.py` into `vanilla_pinn.py` — single script now handles both
  single-config and batch/parallel execution via `--batch` flag
- Removed separate `experiments/poisson1d/batch.py`
- `vanilla_pinn.py` now bootstraps `sys.path` internally (no longer requires
  `pip install -e .` to run)

## [0.1.0] - Project Restructure

### Added
- `pyproject.toml` with proper dependency management
- Flat layout package structure (`unipinn/`)
- Automated test suite under `tests/`
- `scripts/generate_config.py` for JSON config generation
- Unified mesh I/O dispatcher (`geometry/io/`)
- Merged Gaussian quadrature module with full Jacobian support
- Element quality metrics module (`geometry/quality.py`)

### Changed
- Decomposed monolithic `Poisson2D.py` (1450 lines) into focused modules
- Unified `load_q4.py`, `load_q8.py`, `load_q9.py` into single `gmsh_io.py`
- Merged `gaussian_quad.py` and `gaussian_quad_eval.py` into `numerics/quadrature.py`
- Split `weak_pde.py` into `numerics/quadrature.py` + `numerics/test_functions.py`
- Deduplicated `Poisson1DVanillaPINNConfig` into `config/poisson1d.py`
- Renamed `test_case/` to `experiments/`
- Renamed `.tmp_data/` to `data/`
- Fixed typos: `save_essencial` -> `save` (artifacts), `add_exp_grop` -> `add_exp_group`
- All `sys.path.append` hacks removed; proper package imports throughout

### Removed
- Original `utils/` junk drawer (contents redistributed to semantic modules)
