# CMEPDA-project
Repository about CMEPDA project

## Requirements

- Python >= 3.9
- Conda (recommended for managing the environment)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlbertoMontanelli/CMEPDA-project.git
cd CMEPDA-project
```

### Step 2: Create the Environment

Using the provided environment.yml file, create a new Conda environment:

```bash
conda env create -f environment.yml
conda activate myrootenv
```

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

## Run the code

To generate events using the provided ```pythia_generator.py``` script, run the following command:

```bash
python pythia_generator/pythia_generator.py --output events.root --n_events 100
```

Replace ```events.root``` and ```100``` with your desired output file name and number of events, respectively.

## Run the tests

This repository includes automated tests using ```pytest```. To run the tests, execute:

```bash
pytest tests/
```

Ensure that all tests pass before proceeding with your work.

## Repository structure
```
CMEPDA-project/
│
├── environment.yml          # Conda environment file
├── README.md                # Project documentation
├── pythiatransformer/        # Source code
│   ├── __init__.py
│   ├── pythia_generator.py
|   ├── data_processing.py
│   └── transformer.py
├── tests/                   # Automated tests
│   ├── test_pythia-generator.py
│   └── test_transformer.py
├── pyproject.toml           # Build system configuration
└── .github/workflows/       # GitHub Actions CI configuration
    └── tests.yml      

```