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
# 1D Poisson — single experiment
python experiments/poisson1d/vanilla_pinn.py

# 1D Poisson — batch run (14 configs, 4 workers)
python experiments/poisson1d/vanilla_pinn.py --batch --parallel 4

# 2D Poisson — single experiment
python experiments/poisson2d/vanilla_pinn.py

# 2D Poisson — batch run (8 configs, 4 workers)
python experiments/poisson2d/vanilla_pinn.py --batch --parallel 4

# Analyze 1D results
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
  config/         Experiment configuration dataclasses (1D + 2D)
  io/             Artifact saving, workspace management, console utilities
  utils/          Tensor conversion utilities
experiments/
  poisson1d/
    vanilla_pinn.py    Single and batch runner (train + evaluate + save)
    registry.py        Pre-defined configs (cfg_0–13), configuration as code
    analyze.py         Result analysis and comparison
  poisson2d/
    vanilla_pinn.py    Single and batch runner (train + evaluate + save)
    registry.py        Pre-defined configs (cfg_0–7), configuration as code
scripts/               CLI utilities (config generation, metadata management)
tests/                 Automated test suite (pytest)
data/                  Static data files (meshes, etc.)
doc/                   Architecture diagrams and design notes
.tmp_results/          Experiment outputs (gitignored)
```

## Configuration

Configurations follow a **configuration-as-code** pattern: all experiment
parameters are defined as Python dataclass instances, not JSON/YAML files.
This keeps configs type-safe, version-controlled, and composable.

### 1D Poisson experiments

#### Single run

```bash
# Run with the default config
python experiments/poisson1d/vanilla_pinn.py

# Run with a custom JSON config (generated separately, if needed)
python experiments/poisson1d/vanilla_pinn.py --config configs/my_config.json

# Generate a JSON config from the dataclass defaults
python scripts/generate_config.py -o configs/exp1.json \
    -m optimizer=lbfgs epochs=5000 precision=float64 bc_weight=50.0
```

#### Batch / parallel runs

Pre-defined experiment configurations live in
`experiments/poisson1d/registry.py`. Each config is a named
`Poisson1DVanillaPINNConfig` instance (cfg_0 .. cfg_13), organised into groups:

| Group          | Configs    | What it varies                          |
|----------------|------------|-----------------------------------------|
| SamplingTest   | cfg_0 – 3  | uniform vs LHS × Adam vs LBFGS         |
| RandomSeedTest | cfg_4 – 5  | 10 random seeds × Adam vs LBFGS        |
| SampleNumTest  | cfg_6 – 13 | n_colloc ∈ {101, 201, 401} × optimiser |

```bash
# List all available configs
python experiments/poisson1d/vanilla_pinn.py --list

# Run all 14 configs sequentially (34 tasks after seed expansion)
python experiments/poisson1d/vanilla_pinn.py --batch

# Run with 4 parallel workers
python experiments/poisson1d/vanilla_pinn.py --batch --parallel 4

# Run only specific configs
python experiments/poisson1d/vanilla_pinn.py --batch --configs cfg_0 cfg_1 cfg_2

# Run all configs in a group
python experiments/poisson1d/vanilla_pinn.py --batch --group SampleNumTest --parallel 3

# Assign specific GPUs (round-robin across workers)
python experiments/poisson1d/vanilla_pinn.py --batch --parallel 4 --gpus 0 1
```

Multi-seed configs (e.g. cfg_4 with 10 seeds) are automatically expanded into
individual tasks. A `BATCH_SUMMARY.md` report is generated after each run.

### Adding new configurations

Edit the appropriate `registry.py` and add a new entry:

```python
# 1D — experiments/poisson1d/registry.py
cfg_14 = Poisson1DVanillaPINNConfig(
    exp_group=["MyNewGroup"],
    optimizer="adam", epochs=20000, lr=1e-3,
    benchmark_sampling_method="lhs", benchmark_lhs="cm",
)

# 2D — experiments/poisson2d/registry.py
cfg_8 = Poisson2DVanillaPINNConfig(
    exp_group=["MyNewGroup"],
    n_colloc=8000, pde_weight=5.0, bc_weight=100.0,
)
```

Then register it in the `REGISTRY` dict at the bottom of the file.

### 2D Poisson experiments

The 2D runner (`experiments/poisson2d/vanilla_pinn.py`) shares the same CLI
interface as the 1D runner. Pre-defined configs target the `steep_product_2d`
benchmark and are organised into two groups:

| Group               | Configs   | What it varies                                 |
|---------------------|-----------|------------------------------------------------|
| UnsupervisedColloc  | cfg_0 – 4 | sampling method, loss weights, Adam vs LBFGS  |
| SupervisedRecon     | cfg_5 – 7 | sparse data count (50 / 200 / 500 points)      |

```bash
# List 2D configs
python experiments/poisson2d/vanilla_pinn.py --list

# Run unsupervised configs only
python experiments/poisson2d/vanilla_pinn.py --batch --group UnsupervisedColloc --parallel 4

# Run supervised reconstruction configs
python experiments/poisson2d/vanilla_pinn.py --batch --group SupervisedRecon
```

## Running Tests

```bash
pytest
pytest tests/test_benchmarks.py -v
```
