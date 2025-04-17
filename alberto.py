"""Process data from pythia_generator.py using Awkward Arrays.

This script reads ROOT files, preprocesses the data (standardization
and normalization), and converts them into Torch tensors, split
into training, validation, and test sets.

"""

import awkward as ak
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import uproot

# Load the ROOT files into Awkward Arrays
with uproot.open("events.root") as file:
    data_23 = file["tree_23"].arrays(library="ak")
    data_final = file["tree_final"].arrays(library="ak")


# === Preprocessing Functions ===

def standardize_features(data, features):
    """Standardize features (mean=0, std=1) directly on Awkward Arrays."""
    for feature in features:
        mean = ak.mean(data[feature])
        std = ak.std(data[feature])
        data[feature] = (data[feature] - mean) / std
    return data

def log_scale_features(data, features):
    """Apply log-scaling to specified features directly on Awkward Arrays."""
    for feature in features:
        data[feature] = np.log1p(data[feature])
    return data

def preprocess_awkward_data(data, standardize_cols, log_cols):
    """
    Preprocess Awkward Array data by standardizing and log-scaling features.

    Args:
        data (ak.Array): Input Awkward Array.
        standardize_cols (list): Features to standardize.
        log_cols (list): Features to log-scale.

    Returns:
        data (ak.Array): Preprocessed Awkward Array.
    """
    data = standardize_features(data, standardize_cols)
    data = log_scale_features(data, log_cols)
    return data

# Preprocess the data
data_23 = preprocess_awkward_data(
    data_23, 
    standardize_cols=["px_23", "py_23", "pz_23"], 
    log_cols=["e_23", "m_23"]
)

data_final = preprocess_awkward_data(
    data_final, 
    standardize_cols=["px_final", "py_final", "pz_final"], 
    log_cols=["e_final", "m_final"]
)

# === Convert Awkward Array to Padded Torch Tensor ===

def awkward_to_padded_tensor(data, selected_features):
    """
    Convert Awkward Array to padded Torch tensor.

    Args:
        data (ak.Array): Input Awkward Array.
        feature_cols (list): List of feature columns to stack.

    Returns:
        padded_tensor (torch.Tensor): Padded tensor (num_events, max_particles, num_features).
        attention_mask (torch.Tensor): Attention mask (0 for actual, 1 for padding).
    """
    # Calcolare il numero massimo di particelle
    max_particles = ak.max(ak.num(data[selected_features[0]]))

    # Applicare il padding su ciascuna feature e raccoglierle in un dizionario
    padded_events = {
        feature: ak.fill_none(ak.pad_none(data[feature], target=max_particles, axis=1), 0)
        for feature in selected_features
    }

    # Convertire le feature in NumPy array
    padded_arrays = [ak.to_numpy(padded_events[feature]) for feature in selected_features]
    # Combinare le feature lungo l'asse finale (axis=-1)
    padded_array = np.stack(padded_arrays, axis=-1)

    # Convertire in tensore Torch
    padded_tensor = torch.tensor(padded_array, dtype=torch.float32)

    # Compute attention mask (1 for padding, 0 for actual particles)
    attention_mask = ak.num(data[selected_features[0]], axis=1)
    attention_mask = torch.tensor([[0] * num + [1] * (padded_array.shape[1] - num)
                                    for num in attention_mask], dtype=torch.bool)

    return padded_tensor, attention_mask

# Convert the data to padded tensors
padded_tensor_23, attention_mask_23 = awkward_to_padded_tensor(
    data_23, ["id_23", "px_23", "py_23", "pz_23", "e_23", "m_23"]
)

padded_tensor_final, attention_mask_final = awkward_to_padded_tensor(
    data_final, ["id_final", "px_final", "py_final", "pz_final", "e_final", "m_final"]
)

# === Split Data ===

def train_val_test_split(tensor, train_perc=0.6, val_perc=0.2, test_perc=0.2):
    """
    Split a tensor into training, validation, and test sets.

    Args:
        tensor (torch.Tensor): Input tensor.
        train_perc (float): Fraction for training set.
        val_perc (float): Fraction for validation set.
        test_perc (float): Fraction for test set.

    Returns:
        tuple: Training, validation, and test tensors.
    """
    if not np.isclose(train_perc + val_perc + test_perc, 1.0):
        raise ValueError("Splits must sum to 1.0")
    
    invalids = []
    if not (0 <= train_perc <= 1):
        invalids.append(f"train_perc = {train_perc}")
    if not (0 <= val_perc <= 1):
        invalids.append(f"val_perc = {val_perc}")
    if not (0 <= test_perc <= 1):
        invalids.append(f"test_perc = {test_perc}")
    if invalids:
        raise ValueError(f"Invalid value(s) for {','.join(invalids)}. Expected value(s) between 0 and 1, included.")

    nn = len(tensor)
    len_train = int(train_perc*nn)
    len_val = int(val_perc*nn)

    training_set = tensor[:len_train]
    validation_set = tensor[len_train:(len_train + len_val)]
    test_set = tensor[(len_train + len_val):]

    return training_set, validation_set, test_set

# Split data into train/val/test sets

training_set_23, validation_set_23, test_set_23 = train_val_test_split(padded_tensor_23)
attention_train_23, attention_val_23, attention_test_23 = train_val_test_split(attention_mask_23)
training_set_final, validation_set_final, test_set_final = train_val_test_split(padded_tensor_final)
attention_train_final, attention_val_final, attention_test_final = train_val_test_split(attention_mask_final)
print(training_set_23)

# Save to file or proceed with training as needed
