import awkward as ak
import numpy as np
import torch
import uproot
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

#######################################
#           UTILITY FUNCTIONS         #
#######################################


def standardize_features(data, features):
    """Standardize features (mean=0, std=1) directly on Awkward Arrays.

    Args:
        data (ak.Array): Input Awkward Array.
        features (list): List of features to standardize.

    Returns:
        tuple: (standardized data, list of means, list of stds)
    """
    means, stds = [], []
    for feature in features:
        mean = ak.mean(data[feature])
        std = ak.std(data[feature])
        data[feature] = (data[feature] - mean) / std
        means.append(mean)
        stds.append(std)
    return data, means, stds


def compute_pt(data, px_key, py_key, new_key):
    """Compute transverse momentum and add it as a new column."""
    data[new_key] = np.sqrt(data[px_key] ** 2 + data[py_key] ** 2)
    return data


def awkward_to_padded_tensor(data, features):
    """Convert Awkward Array to padded tensor.

    Args:
        data (ak.Array): Input Awkward Array.
        features (list): List of feature names to extract.

    Returns:
        tuple: (padded_tensor, padding_mask)
    """
    event_particles = ak.num(data[features[0]], axis=1)
    max_particles = ak.max(event_particles)
    num_features = len(features)

    padded_events = {
        f: ak.fill_none(ak.pad_none(data[f], target=max_particles, axis=1), 0)
        for f in features
    }
    base_tensor = torch.tensor(
        np.stack([ak.to_numpy(padded_events[f]) for f in features], axis=-1),
        dtype=torch.float32,
    )

    indices = torch.argsort(base_tensor[:, :, -1], dim=1, descending=True)
    padded_tensor = torch.gather(
        base_tensor,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, num_features),
    )

    padding_mask = torch.tensor(
        [[0] * n + [1] * (max_particles - n) for n in event_particles],
        dtype=torch.bool,
    )
    return padded_tensor, padding_mask


def batching(input, target, batch_size, shuffle=True):
    """Create a DataLoader for batching and shuffling input/target pairs."""
    if not isinstance(batch_size, int):
        raise TypeError(f"Batch size must be int, got {type(batch_size)}")
    if not (batch_size <= input.shape[0]):
        raise ValueError(
            f"Batch size must be smaller than the input dataset size."
        )
    generator = torch.Generator().manual_seed(1)
    loader = DataLoader(
        TensorDataset(input, target),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        generator=generator if shuffle else None,
    )
    return loader


def train_val_test_split(
    tensor, train_perc=0.8, val_perc=0.19, test_perc=0.01
):
    """Split a tensor into training, validation, and test sets.

    Args:
        tensor (torch.Tensor): Tensor to split.
        train_perc (float): Percentage for training set.
        val_perc (float): Percentage for validation set.
        test_perc (float): Percentage for test set.

    Returns:
        tuple: (train_tensor, val_tensor, test_tensor)
    """
    if not (train_perc + val_perc + test_perc == 1):
        raise ValueError(
            "Invalid values for split percentages. Must sum to 1."
        )
    if not (0 <= train_perc <= 1):
        raise ValueError(f"Invalid train_perc={train_perc}. Must be in [0,1].")
    if not (0 <= val_perc <= 1):
        raise ValueError(f"Invalid val_perc={val_perc}. Must be in [0,1].")
    if not (0 <= test_perc <= 1):
        raise ValueError(f"Invalid test_perc={test_perc}. Must be in [0,1].")

    n = len(tensor)
    i1 = int(train_perc * n)
    i2 = i1 + int(val_perc * n)
    return tensor[:i1], tensor[i1:i2], tensor[i2:]


def pdg_to_index(tensor, padding_mask):
    dict_ids = {
        21: 30,  # gluon
        22: 31,  # photon
        -12: 32,  # antineutrino e
        12: 33,  # neutrino e
        -14: 34,  # antineutrino mu
        14: 35,  # neutrino mu
        -16: 36,  # antineutrino tau
        16: 37,  # neutrino tau
        -11: 38,  # positron
        11: 39,  # electron
        -2: 40,  # antiup
        2: 41,  # up
        -1: 42,  # antidown
        1: 43,  # down
        -3: 44,  # antistrange
        3: 45,  # strange
        -13: 46,  # antimu
        13: 47,  # mu
        -211: 48,  # pi-
        211: 49,  # pi +
        -321: 50,  # K-
        321: 51,  # K+
        130: 52,  # K0 long
        -2212: 53,  # antiproton
        2212: 54,  # proton
        -2112: 55,  # antineutron
        2112: 56,  # neutron
        -4: 57,  # anticharm
        4: 58,  # charm
        -5: 59,  # antibottom
        5: 60,  # bottom
    }
    for pdg_id, index in dict_ids.items():
        mask = tensor[:, :, 0] == pdg_id
        tensor[:, :, 0][mask] = index
    tensor[~padding_mask] = tensor[~padding_mask] - 29
    tensor = tensor.long()
    return tensor


def load_and_save_tensor(filename):

    logger.info("Beginning data_processing")

    with uproot.open(filename) as file:
        data_23 = file["tree_23"].arrays(library="ak")
        data_final = file["tree_final"].arrays(library="ak")

    logger.info("Opening of root file trees with uproot terminated")

    padded_tensor_23, padding_mask_23 = awkward_to_padded_tensor(
        data_23, ["e_23"]
    )
    padded_tensor_final, padding_mask_final = awkward_to_padded_tensor(
        data_final,
        ["e_final"],
    )

    logger.info("Padded tensors created")

    for i in range(10):
        print(f"evento {i}")
        print(padded_tensor_23[i, :, :])
        print(padding_mask_23[i, :])
        print(padded_tensor_final[i, :, :])
        print(padding_mask_final[i, :])
        print("\n")

    train_23, val_23, test_23 = train_val_test_split(padded_tensor_23)
    train_final, val_final, test_final = train_val_test_split(
        padded_tensor_final
    )
    mask_train_23, mask_val_23, mask_test_23 = train_val_test_split(
        padding_mask_23
    )
    mask_train_final, mask_val_final, mask_test_final = train_val_test_split(
        padding_mask_final
    )

    logger.info("Train/Val/Test splitting terminated")

    # # Salvataggio dei tensori per ripristino futuro
    # torch.save(train_23, "train_23_1M_7Gev.pt")
    # logger.info("Tensor train_23 saved")
    # torch.save(train_final, "train_final_1M_7Gev.pt")
    # logger.info("Tensor train_final saved")
    # torch.save(val_23, "val_23_1M_7Gev.pt")
    # logger.info("Tensor val_23 saved")
    # torch.save(val_final, "val_final_1M_7Gev.pt")
    # logger.info("Tensor val_final saved")
    # torch.save(test_23, "test_23_1M_7Gev.pt")
    # logger.info("Tensor test_23 saved")
    # torch.save(test_final, "test_final_1M_7Gev.pt")
    # logger.info("Tensor test_final saved")
    # torch.save(mask_train_23, "mask_train_23_1M_7Gev.pt")
    # logger.info("Tensor mask_train_23 saved")
    # torch.save(mask_train_final, "mask_train_final_1M_7Gev.pt")
    # logger.info("Tensor mask_train_final saved")
    # torch.save(mask_val_23, "mask_val_23_1M_7Gev.pt")
    # logger.info("Tensor mask_val_23 saved")
    # torch.save(mask_val_final, "mask_val_final_1M_7Gev.pt")
    # logger.info("Tensor mask_val_final saved")
    # torch.save(mask_test_23, "mask_test_23_1M_7Gev.pt")
    # logger.info("Tensor mask_test_23 saved")
    # torch.save(mask_test_final, "mask_test_final_1M_7Gev.pt")
    # logger.info("Tensor mask_test_final saved")


def load_saved_dataloaders(batch_size):
    """Load tensors salvati da file .pt e ricrea i DataLoader."""

    # Caricamento tensori da file .pt
    train_23 = torch.load("train_23_1M_7Gev.pt")
    train_final = torch.load("train_final_1M_7Gev.pt")
    val_23 = torch.load("val_23_1M_7Gev.pt")
    val_final = torch.load("val_final_1M_7Gev.pt")
    test_23 = torch.load("test_23_1M_7Gev.pt")
    test_final = torch.load("test_final_1M_7Gev.pt")

    mask_train_23 = torch.load("mask_train_23_1M_7Gev.pt")
    mask_train_final = torch.load("mask_train_final_1M_7Gev.pt")
    mask_val_23 = torch.load("mask_val_23_1M_7Gev.pt")
    mask_val_final = torch.load("mask_val_final_1M_7Gev.pt")
    mask_test_23 = torch.load("mask_test_23_1M_7Gev.pt")
    mask_test_final = torch.load("mask_test_final_1M_7Gev.pt")

    # Ricostruzione dei DataLoader
    loader_train = batching(train_23, train_final, batch_size)
    loader_val = batching(val_23, val_final, batch_size)
    loader_test = batching(test_23, test_final, batch_size)

    loader_padding_train = batching(
        mask_train_23, mask_train_final, batch_size
    )
    loader_padding_val = batching(mask_val_23, mask_val_final, batch_size)
    loader_padding_test = batching(mask_test_23, mask_test_final, batch_size)

    # Informazioni extra
    subset = train_23[0, 0, :]  # esempio primo vettore features

    return (
        loader_train,
        loader_val,
        loader_test,
        loader_padding_train,
        loader_padding_val,
        loader_padding_test,
        subset,
    )


if __name__ == "__main__":
    load_and_save_tensor("events_10k.root")
