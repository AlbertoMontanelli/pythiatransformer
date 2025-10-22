"""
Process the data from ``pythia_generator.py``.

This code implements:

- the transformation of Awkward arrays into padded tensors, necessary
  for the transformer architecture;
- the truncation of the final particles until the sum of their pT is
  50% of the sum of the status 23 particles pT: less particles make
  the data more palatable to the transformer;
- the batching of the data;
- the splitting of the data into training, validation and test sets;
- the saving of the tensors and the loaders.
"""

import argparse
import math

import awkward as ak
import numpy as np
import torch
import uproot
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from pythiatransformer.pythia_generator import _dir_path_finder


def _validate_args(data, features):
    if not isinstance(data, ak.Array):
        raise TypeError(
            f"Parameter 'data' must be of type 'ak.Array',"
            f" got '{type(data)}' instead."
        )
    if not isinstance(features, list):
        raise TypeError(
            "Parameter 'features' must be of type 'list', "
            f"got '{type(features)}' instead."
        )
    if not all(isinstance(f, str) for f in features):
        raise TypeError("Parameter 'features' must be a list of strings.")
    missing_features = [f for f in features if f not in data.fields]
    if missing_features:
        raise KeyError(
            f"{missing_features} are not features present in 'data'."
        )
    event_particles = ak.num(data[features[0]], axis=1)
    max_particles = ak.max(event_particles)
    num_features = len(features)
    return event_particles, int(max_particles), num_features


def _pad_and_sort_by_pt(data, features, max_particles, list_pt=None):
    padded_events = {
        f: ak.fill_none(ak.pad_none(data[f], target=max_particles, axis=1), 0)
        for f in features
    }
    base = torch.tensor(
        np.stack([ak.to_numpy(padded_events[f]) for f in features], axis=-1),
        dtype=torch.float32,
    )  # shape (nr_events, max_particles, num_features).

    # dictionary to find "pT" feature index.
    feat2idx = {f: i for i, f in enumerate(features)}
    if ((list_pt is None) and ("pT_23" not in feat2idx)) or (
        (list_pt is not None) and ("pT_final" not in feat2idx)
    ):
        raise KeyError("Feature 'pT' is not in 'features'.")
    if list_pt is None:
        pt_idx = feat2idx["pT_23"]
    else:
        pt_idx = feat2idx["pT_final"]
    pt_tensor = base[:, :, pt_idx]  # shape (nr_events, max_particles).

    indices = torch.argsort(pt_tensor, dim=1, descending=True)
    padded_tensor = torch.gather(
        base, 1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1))
    )  # shape (nr_events, max_particles, num_features).
    return padded_tensor, pt_idx


def awkward_to_padded_tensor(data, features, list_pt=None):
    """
    Convert an Awkward Array to a padded PyTorch tensor.

    Optionally truncate each event based on the cumulative transverse
    momentum (pT).

    Parameters
    ----------
    data : ak.Array
        Input Awkward Array.
    features : list[str]
        Feature names to extract. The last feature is assumed to be
        ``pT``.
    list_pt : torch.Tensor or None
        Event-wise total transverse momentum. Required when building
        the final particle truncated tensors so that the cumulative
        `pT` first exceeding 50% of ``list_pt`` is kept.

    Returns
    -------
    padded_tensor : torch.Tensor
        Padded (and possibly truncated) tensor of shape
        ``[nr_events, max_len, num_features]``.
    padding_mask : torch.Tensor
        Boolean mask of shape ``[nr_events, max_len]`` with 0 for real
        tokens and 1 for padding.
    total_pt : torch.Tensor or None
        Total pT per event of shape ``[nr_events]`` when not truncated,
        otherwise ``None`` when truncation was applied.
    """
    event_particles, max_particles, num_features = _validate_args(
        data, features
    )

    padded_tensor, pt_idx = _pad_and_sort_by_pt(
        data, features, max_particles, list_pt
    )
    pt_padded_tensor = padded_tensor[
        :, :, pt_idx
    ]  # shape (nr_events, max_particles).

    if list_pt is not None:
        # This part of the function is used for the data processing of
        # the final particles, for which truncation is applicated.
        if list_pt == []:
            raise ValueError("Parameter 'list_pt' can not be an empty list.")
        if not isinstance(list_pt, torch.Tensor):
            raise TypeError(
                "Parameter 'list_pt' must be of type 'torch.Tensor', "
                f"got '{type(list_pt)}' instead."
            )
        if not all(isinstance(pt, (int, float)) for pt in list_pt.tolist()):
            raise ValueError("Parameter 'list_pt' must be a list of numbers.")
        if len(list_pt) != len(data):
            raise ValueError(
                f"'list_pt' must have length equal to number of events "
                f"({len(data)}), got {len(list_pt)} instead."
            )
        nr_events = len(event_particles)

        # Establish an arbitrary threshold to truncate the particles in
        # each event.
        threshold = 0.5 * list_pt  # shape (nr_events,).
        cum_pt = torch.cumsum(
            pt_padded_tensor, dim=1
        )  # cumulative sum of pT. Shape (nr_events, max_particles).

        # The index s.t. cum_pt < threshold is found for each event.
        keep_lengths = (cum_pt < threshold.unsqueeze(1)).sum(
            dim=1
        ) + 1  # + 1 in order to consider the particle above 50%.
        keep_lengths = torch.clamp(
            keep_lengths, max=max_particles
        )  # shape (nr_events,).

        new_max_len = int(keep_lengths.max().item())  # scalar.

        padded_tensor_trunc = torch.zeros(
            (nr_events, new_max_len, num_features), dtype=torch.float32
        )  # shape (nr_events, new_max_len, num_features).
        padding_mask = torch.ones(
            (nr_events, new_max_len), dtype=torch.bool
        )  # shape (nr_events, new_max_len).

        for i in range(nr_events):
            k = keep_lengths[
                i
            ].item()  # the index for which the batch is truncated.
            padded_tensor_trunc[i, :k, :] = padded_tensor[i, :k, :]
            padding_mask[i, :k] = 0

        total_pt = None
        return padded_tensor_trunc, padding_mask, total_pt

    else:
        # This part of the function is used for the data processing of
        # status 23 particles, for which no truncation is necessary.
        padding_mask = torch.tensor(
            [[0] * n + [1] * (max_particles - n) for n in event_particles],
            dtype=torch.bool,
        )
        total_pt = pt_padded_tensor.sum(dim=1)  # shape (nr_events,).
        return padded_tensor, padding_mask, total_pt


def batching(input, target, batch_size, shuffle=True):
    """
    Create a DataLoader for batching and shuffling input/target pairs.

    Parameters
    ----------
    input : torch.Tensor
        Input dataset.
    target : torch.Tensor
        Target dataset.
    batch_size : int
        Size of the mini-batches.
    shuffle : bool
        If ``True``, performs shuffling of the data.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        ``DataLoader`` object containing the batched and shuffled
        input-target pairs.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(
            f"Parameter 'input' must be a torch.Tensor, got '{type(input)}' "
            f"instead"
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"Parameter 'target' must be a torch.Tensor, got '{type(target)}'"
            f" instead"
        )
    if input.shape[0] != target.shape[0]:
        raise ValueError(
            f"Parameters 'input' and 'target' must have the same number of "
            f"samples. Got {input.shape[0]} and {target.shape[0]}"
            f" respectively."
        )
    if not isinstance(batch_size, int):
        raise TypeError(
            "Parameter 'batch_size' must be of type 'int', "
            f"got '{type(batch_size)}' instead."
        )
    if not batch_size <= input.shape[0]:
        raise ValueError(
            "Parameter 'batch_size' must be smaller than or equal "
            f"to the input dataset size {input.shape[0]}, "
            f"got {batch_size} instead."
        )
    if batch_size < 1:
        raise ValueError(
            f"Parameter 'batch_size' must be at least 1, got {batch_size} "
            f"instead."
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


def train_val_test_split(tensor, train_perc=0.8, val_perc=0.1, test_perc=0.1):
    """
    Split a tensor into training, validation, and test sets.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to split into subsets.
    train_perc : float
        Fraction of the dataset to use for the training set.
    val_perc : float
        Fraction of the dataset to use for the validation set.
    test_perc : float
        Fraction of the dataset to use for the test set.
    min_size : int
        Minimum acceptable size for each subset.

    Returns
    -------
    tuple: torch.Tensor
        A tuple ``(train_tensor, val_tensor, test_tensor)`` containing
        the corresponding splits of the input tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Parameter 'tensor' must be of type torch.Tensor, "
            f"got '{type(tensor)}' instead."
        )
    if not math.isclose(
        train_perc + val_perc + test_perc, 1.0, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            "Invalid values for split percentages. Must sum to 1."
        )
    if not 0 <= train_perc <= 1:
        raise ValueError(
            f"Invalid train_perc={train_perc}. Must be in [0,1]."
        )
    if not 0 <= val_perc <= 1:
        raise ValueError(f"Invalid val_perc={val_perc}. Must be in [0,1].")
    if not 0 <= test_perc <= 1:
        raise ValueError(f"Invalid test_perc={test_perc}. Must be in [0,1].")

    n = len(tensor)
    i1 = int(train_perc * n)
    i2 = i1 + int(val_perc * n)

    return tensor[:i1], tensor[i1:i2], tensor[i2:]


def load_and_save_tensor(suffix):
    """
    Load the ak.Array data and save the Torch Tensors.

    When the data is uploaded using uproot and converted back to an
    Awkward Array, the original structure is preserved.
    So the Awkward Array will look like:
    ::

        [
        {feature_1: [...], feature_2: [...], ...},   # event 0
        {feature_1: [...], feature_2: [...], ...},   # event 1
        ...
        ]

    Parameters
    ----------
        suffix: str
            String appended to the output data tensor filename
            identifying the number of events in the dataset.
    """
    logger.info("Beginning data_processing.")

    data_dir = _dir_path_finder(data=True)
    file_path = data_dir / f"events_{suffix}.root"
    file = uproot.open(file_path)
    if "tree_23" not in file:
        raise KeyError("'tree_23' not found in file.")
    if "tree_final" not in file:
        raise KeyError("'tree_final' not found in file.")
    data_23 = file["tree_23"].arrays(library="ak")
    data_final = file["tree_final"].arrays(library="ak")

    logger.info("Opening of root file trees with uproot terminated.")

    padded_tensor_23, padding_mask_23, pt_23 = awkward_to_padded_tensor(
        data_23, ["pT_23"]
    )
    padded_tensor_final, padding_mask_final, _ = awkward_to_padded_tensor(
        data_final, ["pT_final"], list_pt=pt_23
    )
    logger.info("Padded tensors created.")

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

    logger.info("Train/Val/Test splitting terminated.")

    # Saving the tensors for future recovery.
    filename = data_dir / f"dataset_{suffix}.pt"
    tensor_dict = {
        "train_23": train_23,
        "train_final": train_final,
        "val_23": val_23,
        "val_final": val_final,
        "test_23": test_23,
        "test_final": test_final,
        "mask_train_23": mask_train_23,
        "mask_train_final": mask_train_final,
        "mask_val_23": mask_val_23,
        "mask_val_final": mask_val_final,
        "mask_test_23": mask_test_23,
        "mask_test_final": mask_test_final,
    }
    torch.save(tensor_dict, filename)
    logger.info(f"Saved tensor dict to {filename}")


def load_saved_dataloaders(batch_size, suffix):
    """
    Load pre-saved tensors and build DataLoaders for train/val/test.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch in each DataLoader.
    suffix: str
            String appended to the data tensor loaded filename
            identyfing the number of events in the dataset.
    Returns
    -------
    loader_train : torch.utils.data.DataLoader
        Training DataLoader yielding ``(input_23, target_final)``
        batches.
    loader_val : torch.utils.data.DataLoader
        Validation DataLoader.
    loader_test : torch.utils.data.DataLoader
        Test DataLoader.
    loader_padding_train : torch.utils.data.DataLoader
        Training mask DataLoader yielding ``(mask_23, mask_final)``
        batches.
    loader_padding_val : torch.utils.data.DataLoader
        Validation mask DataLoader.
    loader_padding_test : torch.utils.data.DataLoader
        Test mask DataLoader.
    """
    # Loading tensors.
    data_dir = _dir_path_finder(data=True)
    filename = data_dir / f"dataset_{suffix}.pt"
    tensor_dict = torch.load(filename, weights_only=False)

    train_23 = tensor_dict["train_23"]
    train_final = tensor_dict["train_final"]
    val_23 = tensor_dict["val_23"]
    val_final = tensor_dict["val_final"]
    test_23 = tensor_dict["test_23"]
    test_final = tensor_dict["test_final"]
    mask_train_23 = tensor_dict["mask_train_23"]
    mask_train_final = tensor_dict["mask_train_final"]
    mask_val_23 = tensor_dict["mask_val_23"]
    mask_val_final = tensor_dict["mask_val_final"]
    mask_test_23 = tensor_dict["mask_test_23"]
    mask_test_final = tensor_dict["mask_test_final"]

    if train_23.shape[0] != train_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in training sets, respectively "
            f"{train_23.shape[0]} and {train_final.shape[0]}."
        )
    if val_23.shape[0] != val_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in validation sets, respectively"
            f" {val_23.shape[0]} and {val_final.shape[0]}."
        )
    if test_23.shape[0] != test_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in test sets, respectively "
            f"{test_23.shape[0]} and {test_final.shape[0]}."
        )

    # Rebuilding the DataLoaders.
    loader_train = batching(train_23, train_final, batch_size)
    loader_val = batching(val_23, val_final, batch_size)
    loader_test = batching(test_23, test_final, batch_size)

    loader_padding_train = batching(
        mask_train_23, mask_train_final, batch_size
    )
    loader_padding_val = batching(mask_val_23, mask_val_final, batch_size)
    loader_padding_test = batching(mask_test_23, mask_test_final, batch_size)

    return (
        loader_train,
        loader_val,
        loader_test,
        loader_padding_train,
        loader_padding_val,
        loader_padding_test,
    )


def main():
    """
    Call ``load_and_save_tensor`` with parser arguments.

    CLI Parameters
    --------------
    suffix: str, required
            String appended to the data tensor loaded filename
            identyfing the number of events in the dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        required=True,
        help="string identyfing the number of events in the dataset.",
    )
    args = parser.parse_args()

    load_and_save_tensor(args.suffix)


if __name__ == "__main__":
    main()
