"""In this code the data from pythia_generator.py is processed.
First, the ROOT files are imported as awkward arrays.
Then, the features are standardized.
Last, the data is converted to Torch tensors and split
between training, validation and test sets.
"""
import awkward as ak
import numpy as np
import torch
import uproot


def standardize_features(data, features):
    """Standardize features (mean=0, std=1) directly on Awkward Arrays.
        Args:
        data (ak.Array): Input Awkward Array.
        feature (list): Features to standardize.

    Returns:
        data (ak.Array): Standardized Awkward Array.
    """
    for feature in features:
        mean = ak.mean(data[feature])
        std = ak.std(data[feature])
        data[feature] = (data[feature] - mean) / std
    return data

def awkward_to_padded_tensor(data, features):
    """
    Convert Awkward Array to padded Torch tensor.

    Args:
        data (ak.Array): Input Awkward Array.
        feature_cols (list): List of feature columns to stack.

    Returns:
        padded_tensor (torch.Tensor): Padded tensor of  shape:
                                      (num_events, max_particles,
                                      num_features).
        attention_mask (torch.Tensor): Attention mask (0 for actual,
                                       1 for padding).
    """
    # Find max number of particles for all the events.
    max_particles = ak.max(ak.num(data[features[0]]))
    # Pad each feature to ensure an equal number of particles
    # per event. Collect each feature in a new dictionary.
    padded_events = {
        feature: ak.fill_none(
            ak.pad_none(data[feature], target=max_particles, axis=1), 0
            )
        for feature in features
    }
    # Convert the features in numpy arrays and stack them to obtain the
    # desired shape. Then convert into a Torch tensor with the same
    # shape.
    padded_arrays = [
        ak.to_numpy(padded_events[feature]) for feature in features
    ]
    padded_array = np.stack(padded_arrays, axis=-1)
    padded_tensor = torch.tensor(padded_array, dtype=torch.float32)

    # Compute attention mask (1 for padding, 0 for actual particles).
    attention_mask = ak.num(data[features[0]], axis=1)
    attention_mask = torch.tensor([[0] * num + [1] * (padded_array.shape[1] - num)
                                    for num in attention_mask], dtype=torch.bool)

    return padded_tensor, attention_mask

def train_val_test_split(
        tensor, train_perc = 0.6, val_perc = 0.2, test_perc = 0.2
        ):
    """Split a tensor into training, validation, and test sets.

    Args:
        tensor (Torch tensor): data in the form of a Torch tensor.
        train_perc (float): fraction of the data used for training.
        val_perc (float): fraction of the data used for validation.
        test_perc (float): fraction of the data used for testing.

    Return:
        training_set (Torch tensor): training set.
        validation_set (Torch tensor): validation set.
        test_set (Torch tensor): test set.
    """
    if not (train_perc + val_perc + test_perc == 1):
        raise ValueError(
            f"Invalid values for data splitting fractions."
            f" Expected positive fractions that sum up to 1."
        )
    
    invalids = []
    if not (0 <= train_perc <= 1):
        invalids.append(f"train_perc = {train_perc}")
    if not (0 <= val_perc <= 1):
        invalids.append(f"val_perc = {val_perc}")
    if not (0 <= test_perc <= 1):
        invalids.append(f"test_perc = {test_perc}")
    if invalids:
        raise ValueError(
            f"Invalid value(s) for {','.join(invalids)}."
            f" Expected value(s) between 0 and 1, included."
        )
    
    nn = len(tensor)
    len_train = int(train_perc*nn)
    len_val = int(val_perc*nn)

    training_set = tensor[:len_train]
    validation_set = tensor[len_train:(len_train + len_val)]
    test_set = tensor[(len_train + len_val):]

    return training_set, validation_set, test_set

if __name__== "__main__":
    with uproot.open("events.root") as file:
        data_23 = file["tree_23"].arrays(library="ak")
        data_final = file["tree_final"].arrays(library="ak")

    # Standardization.
    data_23 = standardize_features(
        data_23, 
        features=["px_23", "py_23", "pz_23", "pT_23"]
    )
    data_final = standardize_features(
        data_final, 
        features=["px_final", "py_final", "pz_final", "pT_final"]
    )

    padded_tensor_23, attention_mask_23 = awkward_to_padded_tensor(
        data_23,
        features=["id_23", "px_23", "py_23", "pz_23", "pT_23"]
    )
    padded_tensor_final, attention_mask_final = awkward_to_padded_tensor(
        data_final,
        features=["id_final", "px_final", "py_final", "pz_final", "pT_final"]
    )

    training_set_23, validation_set_23, test_set_23 = (
        train_val_test_split(padded_tensor_23)
    )
    attention_train_23, attention_val_23, attention_test_23 = (
        train_val_test_split(attention_mask_23)
    )
    training_set_final, validation_set_final, test_set_final = (
        train_val_test_split(padded_tensor_final)
    )
    attention_train_final, attention_val_final, attention_test_final = (
        train_val_test_split(attention_mask_final)
    )

    print(training_set_23)
