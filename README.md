# UniPINN

Unified Physics-Informed Neural Networks framework for solving PDEs with neural networks.

## Installation

```bash
pip install -e ".[dev]"
```

For mesh I/O support (Gmsh files):
```bash
pip install -e ".[mesh]"
```

## Quick Start

```bash
# Run a 1D Poisson PINN experiment with default config
python experiments/poisson1d/vanilla_pinn.py

# Run with a custom config
python experiments/poisson1d/vanilla_pinn.py --config configs/my_config.json

# Generate a config JSON file
python scripts/generate_config.py -o configs/test.json -m optimizer=lbfgs epochs=4000

# Analyze results
python experiments/poisson1d/analyze.py --case 0
```

## Project Structure

```
unipinn/
  core/           Training engine (Trainer, Callback system)
  nn/             Neural network architectures and activations
  pde/            PDE problem definitions, loss functions, benchmarks
  geometry/       Mesh data structures, generation, I/O, quality metrics
  numerics/       Gaussian quadrature and variational test functions (VPINN)
  metrics/        Error metrics for evaluation
  config/         Experiment configuration dataclasses
  io/             Artifact saving, workspace management, console utilities
  utils/          Tensor conversion utilities
experiments/      Experiment runner scripts (training and analysis)
scripts/          CLI utilities (config generation, metadata management)
tests/            Automated test suite (pytest)
data/             Static data files (meshes, etc.)
.tmp_results/     Experiment outputs (gitignored)
```

## Configuration

Configurations are defined as Python dataclasses in `unipinn/config/`. Use the
`scripts/generate_config.py` script to produce JSON files that can be passed
to experiment runners:

```bash
python scripts/generate_config.py -o configs/exp1.json \
    -m optimizer=lbfgs epochs=5000 precision=float64 bc_weight=50.0
```

## Running Tests

```bash
pytest
pytest tests/test_benchmarks.py -v
```
