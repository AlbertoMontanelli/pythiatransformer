<!-- docs/usage.md -->
# Usage

This guide explains how to use the **pythiatransformer** package, from event generation to inference and plotting.  
It also defines the **naming convention** used for all datasets, models, and plots produced by the framework.

---

## 1. Generate events

Use the `pythiatransformer/pythia_generator.py` script to generate events with **Pythia8**:

```bash
python pythiatransformer/pythia_generator.py --events 12500
```

This script simulates high-energy collisions (with `HardQCD=on`) and stores:
- particles with **status = 23** (inputs);
- **stable final particles** (targets);

Both are saved in a ROOT file inside the `data/` directory as:

```
data/events_<events>.root
```

For example, `events_12500.root` corresponds to a dataset with 12500 events.

---

## 2. Preprocess data

Convert ROOT → Awkward → PyTorch tensors and prepare the dataset using:

```bash
python pythiatransformer/data_processing.py --suffix 12500
```

This performs:
- padding and batching of particles per event;
- truncation of final particles so that their total pT ≈ 50% of the status-23 sum;
- dataset split into train/validation/test sets;
- saving tensors and padding masks in:

```
data/dataset_<events>.pt
```

---

## 3. Train the Transformer model

Train the model using:

```bash
python pythiatransformer/main.py --suffix 12500 --batch_size 256
```

It's possibile to optionally append an extra label for different hyperparameter sets:

```bash
python pythiatransformer/main.py --suffix 12500 --info_suffix runA
```

This creates:
```
data/transformer_model_12500_runA.pt
data/meta_12500_runA.json
plots/learning_curve_12500_runA.pdf
```

If `info_suffix` is omitted, filenames use only the event suffix (`12500`).

---

## 4. Run inference and plotting

After training, run autoregressive inference to reconstruct final-state particles:

```bash
python pythiatransformer/inference.py infer --suffix 12500_runA
```

This loads the trained model and generates:
```
data/results_12500_runA.npz
```

To reproduce plots later without re-running inference:

```bash
python pythiatransformer/inference.py plot --suffix 12500_runA
```

This regenerates plots such as:
```
plots/residuals_hist_12500_runA.pdf
plots/wd_hist_12500_runA.pdf
plots/pt_hist_12500_runA.pdf
plots/token_hist_12500_runA.pdf
```

---

## 5. Naming conventions

All input, output, and plot filenames follow a consistent and automatic naming convention:

```
<category>_<suffix>.ext
```

Where:

- **`<category>`** identifies the file type (e.g. `events`, `dataset`, `transformer_model`, `results`, `learning_curve`, `token_hist`);
- **`<suffix>`** uniquely identifies the dataset or model configuration;
- **`<ext>`** is the file extension (`.root`, `.pt`, `.npz`, `.pdf`, etc).

### Default suffix structure
1. **Base suffix:** number of generated events  
   → identifies a unique dataset.  
   Example: `12500` → `events_12500.root`.

2. **Extended suffix (optional):** adds descriptors for model configurations or hyperparameters.  
   Example:  
   `12500_runA`, `12500_lr5e-4_heads8`.

3. **Consistency rule:**  
   The same suffix must be used **throughout all stages** (generation → preprocessing → training → inference → plotting).

This guarantees reproducibility and traceability.  
Each stage reuses the same suffix, so results stay linked automatically:
```
events_12500.root
dataset_12500.pt
transformer_model_12500.pt
results_12500.npz
learning_curve_12500.pdf
```

**Do not rename files manually.**  
All names are managed by the pipeline to ensure consistent data linkage.

---

## 6. Run the tests

Automated tests use Python’s built-in `unittest` framework.  
Run the suite with:

```bash
PYTHONPATH=. python -m unittest discover tests -v
```

Or run a specific test, for example:

```bash
PYTHONPATH=. python -m tests.test_data_processing -v
```

Available test modules:
- `test_pythia_generator.py`
- `test_data_processing.py`
- `test_transformer.py`
- `test_toy.py`

These ensure the full workflow — generation, preprocessing, training, and inference — runs without errors.