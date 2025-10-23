<!-- docs/rep_structure.md -->
# Repository structure

This document provides an overview of the **pythiatransformer**
repository layout and the role of each main component.  
It also explains the integrated **Continuous Integration (CI)** system that ensures
the codebase, tests, and documentation remain consistent across updates.

---

## Directory Layout

```
pythiatransformer/
│
├── LICENSE
├── README.md                 # Main project description
├── pyproject.toml            # Build configuration (dependencies, optional extras)
│
├── pythiatransformer/        # Core source code package
│   ├── pythia_generator.py   # Generates events with Pythia8
│   ├── data_processing.py    # Converts ROOT → Awkward → PyTorch tensors
│   ├── transformer.py        # Defines the Transformer model
│   ├── main.py               # Trains and validates the Transformer
│   ├── inference.py          # Runs autoregressive inference and plotting
│   └── toy/                  # Toy model for debugging and minimal validation
│       ├── toy_model.py
│       ├── evaluate_toy.py
│       └── toy_model.pt
│
├── tests/                    # Automated test suite (unittest framework)
│   ├── test_pythia_generator.py
│   ├── test_data_processing.py
│   ├── test_transformer.py
│   └── test_toy.py
│
├── docs/                     # Sphinx + MyST documentation
│   ├── index.rst
│   ├── installation.md
│   ├── usage.md
│   ├── repository_structure.md
│   └── _api/                 # Auto-generated API reference by sphinx-apidoc
│
└── .github/workflows/        # Continuous Integration (CI) configurations
    ├── tests.yml             # Runs automated tests on push / pull request
    └── docs.yml              # Builds and deploys documentation to GitHub Pages
```

---

## Automated Tests

All core modules include dedicated unit tests using the Python built-in **unittest** framework.  
Tests can be executed locally with the following command:

```bash
PYTHONPATH=. python -m unittest discover tests -v
```

These tests verify:

- correct event generation and Awkward conversion;
- dataset preprocessing, truncation and batching;
- Transformer training and loss evaluation;
- inference, histogram generation, and toy model reproducibility.

---

## Continuous Integration (CI)

The repository includes **two GitHub Actions workflows** under `.github/workflows/`:

### `tests.yml`

- Runs on every `push` or `pull_request`.
- Creates a Conda environment and installs the package in editable mode.
- Executes all **unittest** test suites to verify that the codebase remains stable.

### `docs.yml`

- Builds the Sphinx documentation using `myst-parser` and `sphinxawesome-theme`.
- Deploys the generated HTML site to **GitHub Pages**.
- Ensures that the public documentation is always in sync with the latest code.

The live documentation is automatically published at:

**[https://albertomontanelli.github.io/pythiatransformer/](https://albertomontanelli.github.io/pythiatransformer/)**

