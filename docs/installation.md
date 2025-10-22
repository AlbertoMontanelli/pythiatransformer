<!-- docs/installation.md -->
# Installation

## Requirements
- Python >= 3.9
- Conda (recommended)
- ROOT + Pythia8

## Quick start
```bash
git clone https://github.com/<user>/pythiatransformer.git
cd pythiatransformer
# conda env, install dipendenze, ecc.

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlbertoMontanelli/pythiatransformer.git
cd pythiatransformer
```

### Step 2: Create the Environment

It is recommended to create a new Conda environment and activate it.
This will install all the dependencies, including:

- ROOT (via conda-forge)

- Pythia8 (bundled with ROOT)

- Python packages: ```numpy```, ```uproot```, ```loguru```

### Step 3: Verify Installation

Check if ROOT and Pythia8 are installed correctly:

```bash
python
>>> import ROOT
>>> import pythia8
```

If no errors are raised, the setup is complete.
