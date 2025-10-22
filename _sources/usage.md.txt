<!-- docs/usage.md -->
# Usage

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