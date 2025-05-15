import awkward as ak
import numpy as np
import torch
import uproot
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


def awkward_to_padded_targets(data, features, eos_token=-999, sos_token=-998):
    """Convert Awkward Array to padded torch.Tensor and insert EOS.

    Args:
        data (ak.Array): Input Awkward array.
        features (list): List of feature keys.
        eos_token (int): Token used to represent EOS.

    Returns:
        tuple: (padded_tensor, padding_mask)
    """
    event_particles = ak.num(data[features[0]], axis=1)
    max_particles = ak.max(event_particles)
    batch_size = len(event_particles)
    num_features = len(features)
    new_max_len = max_particles + 2

    padded_events = {
        f: ak.fill_none(ak.pad_none(data[f], target=max_particles, axis=1), 0)
        for f in features
    }
    base_tensor = torch.tensor(
        np.stack([ak.to_numpy(padded_events[f]) for f in features], axis=-1),
        dtype=torch.float32,
    )

    indices = torch.argsort(base_tensor[:, :, -1], dim=1, descending=True)
    base_tensor_sorted = torch.gather(
        base_tensor,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, num_features),
    )

    padded_tensor = torch.zeros((batch_size, new_max_len, num_features))
    padding_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

    for i, true_particles in enumerate(event_particles):
        n = true_particles.item()
        padded_tensor[i, 0, 0] = sos_token
        padded_tensor[i, 1 : n + 1] = base_tensor_sorted[i, :n]
        padded_tensor[i, n + 1, 0] = eos_token
        padding_mask[i, : n + 2] = 0

    return padded_tensor, padding_mask


def awkward_to_padded_inputs(data, features):
    """Convert Awkward Array to padded tensor (no EOS).

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


def one_hot_encoding(tensor, dict_ids, num_classes, padding_token=0):
    """Apply one-hot encoding to IDs in a tensor, handling EOS and padding.

    Args:
        tensor (torch.Tensor): Input tensor.
        dict_ids (dict): Mapping of PDG IDs to one-hot indices.
        num_classes (int): Total number of classes.
        eos_token (int): EOS token.
        padding_token (int): Token to treat as padding.

    Returns:
        torch.Tensor: One-hot-encoded tensor.
    """
    ids = tensor[:, :, 0].long()
    one_hot = torch.zeros(tensor.size(0), tensor.size(1), num_classes)
    for pdg_id, index in dict_ids.items():
        one_hot[ids == pdg_id] = torch.nn.functional.one_hot(
            torch.tensor(index), num_classes=num_classes
        ).float()
    one_hot[ids == padding_token] = 0
    return one_hot


def batching(input, target, shuffle=True, batch_size=100):
    """Create a DataLoader for batching and shuffling input/target pairs."""
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


def train_val_test_split(tensor, train_perc=0.6, val_perc=0.2, test_perc=0.2):
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


#######################################
#        ESECUZIONE DATI ROOT        #
#######################################

with uproot.open("events.root") as file:
    data_23 = file["tree_23"].arrays(library="ak")
    data_final = file["tree_final"].arrays(library="ak")

# Standardization.
data_23, mean_23, std_23 = standardize_features(
    data_23, features=["px_23", "py_23", "pz_23"]
)
data_final, mean_final, std_final = standardize_features(
    data_final, features=["px_final", "py_final", "pz_final"]
)

data_23 = compute_pt(data_23, "px_23", "py_23", "pT_23")
data_final = compute_pt(data_final, "px_final", "py_final", "pT_final")

# Padding.
padded_tensor_23, padding_mask_23 = awkward_to_padded_inputs(
    data_23, features=["id_23", "px_23", "py_23", "pz_23", "pT_23"]
)
padded_tensor_final, padding_mask_final = awkward_to_padded_targets(
    data_final,
    features=["id_final", "px_final", "py_final", "pz_final", "pT_final"],
)

# Remove pT before one-hot
drop_pt = lambda t: t[:, :, :-1]
padded_tensor_23 = drop_pt(padded_tensor_23)
padded_tensor_final = drop_pt(padded_tensor_final)

# Dict ID & One-hot
eos_token = -999
sos_token = -998
id_all = np.unique(
    np.concatenate(
        [ak.flatten(data_23["id_23"]), ak.flatten(data_final["id_final"])]
    )
)
dict_ids = {int(pid): idx for idx, pid in enumerate(id_all)}
dict_ids[sos_token] = len(dict_ids)
dict_ids[eos_token] = len(dict_ids)
print(dict_ids)
num_classes = len(dict_ids)

one_hot_23 = one_hot_encoding(padded_tensor_23, dict_ids, num_classes)
one_hot_final = one_hot_encoding(padded_tensor_final, dict_ids, num_classes)

padded_tensor_23 = torch.cat((one_hot_23, padded_tensor_23[:, :, 1:]), dim=-1)
padded_tensor_final = torch.cat(
    (one_hot_final, padded_tensor_final[:, :, 1:]), dim=-1
)

# Split
train_23, val_23, test_23 = train_val_test_split(padded_tensor_23)
train_final, val_final, test_final = train_val_test_split(padded_tensor_final)
mask_train_23, mask_val_23, mask_test_23 = train_val_test_split(
    padding_mask_23
)
mask_train_final, mask_val_final, mask_test_final = train_val_test_split(
    padding_mask_final
)

# Loader
loader_train = batching(train_23, train_final)
loader_val = batching(val_23, val_final)
loader_test = batching(test_23, test_final)
loader_padding_train = batching(mask_train_23, mask_train_final)
loader_padding_val = batching(mask_val_23, mask_val_final)
loader_padding_test = batching(mask_test_23, mask_test_final)

subset = train_23[0, 0, :]

if __name__ == "__main__":
    # ===================== DEBUG: verifica EOS nei dati batchati ===============
    inputs, targets = next(iter(loader_train))
    _, targets_mask = next(iter(loader_padding_train))

    id_pred = torch.argmax(targets[:, :, : len(dict_ids)], dim=-1)
    eos_index = dict_ids[eos_token]
    valid_mask = ~targets_mask
    num_real = valid_mask.sum(dim=1)
    last_index = num_real - 1
    B = targets.size(0)
    last_ids = id_pred[torch.arange(B), last_index]

    num_eos = (last_ids == eos_index).sum().item()
    print(f"\n✅ DEBUG BATCH: {num_eos}/{B} eventi terminano con EOS")
    assert torch.all(
        last_ids == eos_index
    ), "❌ Alcuni target batchati non terminano con EOS!"
    print("✅ Tutti i target nel batch terminano con EOS")
    # ==================================================================
