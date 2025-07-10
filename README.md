# CMEPDA-project
Repository about CMEPDA project
A framework to simulate high-energy physics events using Pythia8 and train a Transformer model to predict final state observables such as particle pT.


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
python pythiatransformer/pythia_generator.py --output events.root --n_events 100
```

Replace ```events.root``` and ```100``` with your desired output file name and number of events, respectively.

To train the Transformer model using the provided ```main.py``` script, run the following command:

```bash
python pythiatransformer/main.py
```

To perform inference and predict particles' transverse momentum (pT) using the provided ```inference.py``` script, run the following command:

```bash
python pythiatransformer/inference.py
```

## Run the tests

This repository includes automated tests using ```unittest```. To run the tests, execute:

```bash
PYTHONPATH=. python -m tests.test_name -v
```

Replace ```test_name``` with your desired test to run.
Ensure that all tests pass before proceeding with your work.

## Repository structure
```
CMEPDA-project/
│
├── .gitignore              # Specifies which files and directories should be ignored by Git.
├── LICENCE
├── README.md               # Project documentation.
├── requirements.txt        # Lists the Python packages required to run the CMEPDA project.
├── pythiatransformer/      # Source code.
│   ├── __init__.py
│   ├── pythia_generator.py
|   ├── data_processing.py
│   ├── transformer.py
│   ├── fastjet_preparation.py
│   ├── jet_clustering.py
│   ├── main.py
│   ├── inference.py
│   ├── learning_curve_1M.pdf
│   ├── transformer_model_1M.pt
│   └── toy/
│       ├── toy.py
│       ├── evaluate_toy.py
│       ├── toy_transformer.pt
├── tests/                  # Automated tests.
│   ├── test_pythia_generator.py
│   ├── test_data_processing.py
│   ├── test_transformer.py
│   └── toy_testing_save.root
├── pyproject.toml          # Build system configuration.
└── .github/workflows/      # GitHub Actions CI configuration.
    └── tests.yml      

```