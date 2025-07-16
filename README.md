# CMEPDA-project
Repository about CMEPDA project. A framework to simulate high-energy physics events using Pythia8 and train a Transformer model to predict final state observables such as particle pT.


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

## Run the code

Firstly, you have to generate events using the provided ```pythia_generator.py``` script, running the following command:

```bash
python pythiatransformer/pythia_generator.py
```

Remember to choose the name of the ROOT file that will be created during the execution, and the desired number of events to be generated.

Before training the model, you need to run ```data_processing.py``` to preprocess the dataset, running the following command:

```bash
python pythiatransformer/data_processing.py
```

Preprocessing includes: converting Awkward arrays into padded PyTorch tensors; truncating final particles per event to retain only those carrying 50% of the total transverse momentum; splitting the data into training, validation, and test sets; and saving the resulting tensors and padding masks to disk.

Then, train the Transformer model using the provided ```main.py``` script, by running the following command:

```bash
python pythiatransformer/main.py
```

You can choose the number of epochs and, optionally, the model parameters. A pdf of the learning curve and a file containing the model will be created.

Finally, to perform inference and predict particles' transverse momentum (pT), use the provided ```inference.py``` script by running the following command:

```bash
python pythiatransformer/inference.py
```

A PDF plot of the residuals between the true and predicted transverse momenta is generated.

## Run the tests

This repository includes automated tests using ```unittest```. To run the tests, execute:

```bash
PYTHONPATH=. python -m tests.test_name -v
```

Replace ```test_name``` with the name of the test you want to run. Available tests cover the following scripts: ```transformer.py```, ```data_processing.py```, ```pythia_generator.py``` and ```toy_model.py```.
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
│   ├── learning_curve.pdf
│   ├── transformer_model.pt
│   ├── diff_hist.pdf
│   └── toy/
│       ├── toy_model.py
│       ├── evaluate_toy.py
│       ├── toy_model.pt
│       ├── toy_learning_curve.pdf
│       └── toy_residuals.pdf
├── tests/                  # Automated tests.
│   ├── test_pythia_generator.py
│   ├── test_data_processing.py
│   ├── test_transformer.py
│   ├── test_toy_model.py
│   └── toy_testing_save.root
├── pyproject.toml          # Build system configuration.
└── .github/workflows/      # GitHub Actions CI configuration.
    └── tests.yml      
```