"""
Processing the data from pythia_generator.py.
This code implements: 
i. the transformation of Awkward arrays into padded tensors,
necessary for the transformer architecture;
ii. the truncation of the final particles until the sum of their pT is
50% of the sum of the status 23 particles pT: less particles make the
data more palatable to the transformer;
iii. the batching of the data;
iv. the splitting of the data into training, validation and test sets;
v. the saving of the tensors and the loaders.
"""

import awkward as ak
import math
import numpy as np
import torch
import uproot
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

#######################################
#          UTILITY FUNCTIONS          #
#######################################


def awkward_to_padded_tensor(
    data,
    features,
    list_pt=None,
    truncate_pt=False
):
    """
    Convert Awkward Array to padded Pytorch tensor.
    Optionally, truncate each event based on cumulative transverse
    momentum (pT).

    Args:
        data (ak.Array): input Awkward Array;
        features (list): list of feature names to extract.
                         The last feature is assumed to be pT;
        list_pt (list): total pT of the event particles;
        truncate_pt (bool): if True, truncates the dataset to 
                            when the sum of pT is 50% of list_pt
                            for the event. False by default.

    if truncate_pt=False:
        Returns:
            padded_tensor (torch.Tensor): padded tensor of data;
            padding_mask (torch.Tensor): padding mask relative to
                                         padded_tensor, where 0 marks
                                         real particles and 1 marks
                                         padding;
            total_pt (torch.Tensor): total pT of each event.
    if truncate_pt=True:
        Returns:
            padded_tensor_trunc (torch.Tensor): padded tensor of
                                                truncated data;
            padding_mask (torch.Tensor): padding mask relative to
                                         padded_tensor_trunc.

    """
    # Loading of the data, defining salient variables.
    if not isinstance(data, ak.Array):
        raise TypeError(
            "Parameter 'data' must be of type 'ak.Array', "
            f"got '{type(data)}' instead."
        )
    if not isinstance(features, list):
        raise TypeError(
            "Parameter 'features' must be of type 'list', "
            f"got '{type(features)}' instead."
        )
    if not all(isinstance(f, str) for f in features):
        raise TypeError(
            f"Parameter 'features' must be a list of strings."
        )
    missing_features = [f for f in features if f not in data.fields]
    if missing_features:
        raise KeyError(
            f"{missing_features} are not features present in 'data'."
        )
    event_particles = ak.num(data[features[0]], axis=1)
    max_particles = ak.max(event_particles)
    num_features = len(features)

    # Creation of tensors.
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

    if truncate_pt:
        if list_pt is None:
            raise ValueError(
                f"Parameter 'list_pt' is required when 'truncate_pT=True'."
            )
        if list_pt==[]:
            raise ValueError(
                f"Parameter 'list_pt' can not be an empty list."
            )
        if not isinstance(list_pt, torch.Tensor):
            raise TypeError(
                "Parameter 'list_pt' must be of type 'torch.Tensor', "
                f"got '{type(list_pt)}' instead."
            )
        if not all(isinstance(pt, (int, float)) for pt in list_pt.tolist()):
            raise ValueError(
                "Parameter 'list_pt' must be a list of numbers."
            )
        if len(list_pt) != len(data):
            raise ValueError(
                f"'list_pt' must have length equal to number of events " 
                f"({len(data)}), got {len(list_pt)} instead."
            )
        # This part of the function is used for the data processing of
        # the final particles.
        batch_size = len(event_particles)

        threshold = 0.5 * list_pt # 0.5 is arbitrary.
        cum_pt = torch.cumsum(
            padded_tensor.squeeze(-1), dim=1
        )  # cumulative sum of pT.
        # The index s.t. cum_pt < threshold is found for each event.
        keep_lengths = (cum_pt < threshold.unsqueeze(1)).sum(
            dim=1
        ) + 1  # + 1 in order to consider the particle above 50%.
        keep_lengths = torch.clamp(keep_lengths, max=max_particles)

        new_max_len = keep_lengths.max().item()

        padded_tensor_trunc = torch.zeros(
            (batch_size, new_max_len, num_features)
        )
        padding_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

        for i in range(batch_size):
            k = keep_lengths[
                i
            ].item()  # the index for which the batch is truncated.
            padded_tensor_trunc[i, :k, :] = padded_tensor[i, :k, :]
            padding_mask[i, :k] = 0

    else:
        # This part of the function is used for the data processing of
        # status 23 particles, for which no truncation is necessary.
        padding_mask = torch.tensor(
            [[0] * n + [1] * (max_particles - n) for n in event_particles],
            dtype=torch.bool,
        )
        total_pt = padded_tensor.sum(dim=1).squeeze(-1)

    if truncate_pt:
        return padded_tensor_trunc, padding_mask
    else:
        return padded_tensor, padding_mask, total_pt


def batching(input, target, batch_size, shuffle=True):
    """
    Create a Pytorch DataLoader for batching and optionally shuffling
    input/target pairs.

    Args:
        input (torch.Tensor): input dataset;
        target (torch.Tensor): target dataset;
        batch_size (int): size of the mini batches;
        shuffle (bool): if True, performs shuffling of the data.
                        True by default.
    
    Returns:
        loader ( torch.utils.data.DataLoader): DataLoader object with
                                               batched and shuffled
                                               input and target.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(
            "Parameter 'input' must be a torch.Tensor, "
            f"got '{type(input)}' instead."
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            "Parameter 'target' must be a torch.Tensor, "
            f"got '{type(target)}' instead."
        )
    if input.shape[0] != target.shape[0]:
        raise ValueError(
            f"Parameters 'input' and 'target' must have the same number of "
            f"samples. Got {input.shape[0]} and {target.shape[0]} respectively."
        )
    if not isinstance(batch_size, int):
        raise TypeError(
            "Parameter 'batch_size' must be of type 'int', "
            f"got '{type(batch_size)}' instead."
        )
    if not (batch_size <= input.shape[0]):
        raise ValueError(
            "Parameter 'batch_size' must be smaller than or equal "
            f"to the input dataset size {input.shape[0]}, "
            f"got {batch_size} instead."
        )
    if batch_size < 1:
        raise ValueError(
            "Parameter 'batch_size' must be at least 1, "
            f"got {batch_size} instead."
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
    """
    Split a tensor into training, validation, and test sets.

    Args:
        tensor (torch.Tensor): tensor to split;
        train_perc (float): fraction for training set;
        val_perc (float): fraction for validation set;
        test_perc (float): fraction for test set;
        min_size (int): minimum acceptable size of the sets.

    Returns:
        tuple: (train_tensor, val_tensor, test_tensor)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Parameter 'tensor' must be of type torch.Tensor, "
            f"got '{type(tensor)}' instead."
        )
    if not math.isclose(train_perc + val_perc + test_perc, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "Invalid values for split percentages. Must sum to 1."
        )
    if not (0 <= train_perc <= 1):
        raise ValueError(
            f"Invalid train_perc={train_perc}. Must be in [0,1]."
        )
    if not (0 <= val_perc <= 1):
        raise ValueError(
            f"Invalid val_perc={val_perc}. Must be in [0,1]."
        )
    if not (0 <= test_perc <= 1):
        raise ValueError(
            f"Invalid test_perc={test_perc}. Must be in [0,1]."
        )

    n = len(tensor)
    i1 = int(train_perc * n)
    i2 = i1 + int(val_perc * n)

    n_train = i1
    n_val = i2 - i1
    n_test = n -i2
    return tensor[:i1], tensor[i1:i2], tensor[i2:]


def load_and_save_tensor(filename):
    """
    Loading of the ak.Array data and saving of the Torch Tensors.

    Args:
        filename (str): name of the data file.

    Returns:
        None
    """
    logger.info("Beginning data_processing.")

    with uproot.open(filename) as file:
        if "tree_23" not in file:
            raise KeyError(
                f"'tree_23' not found in file."
            )
        if "tree_final" not in file:
            raise KeyError(
                f"'tree_final' not found in file."
            )
        data_23 = file["tree_23"].arrays(library="ak")
        data_final = file["tree_final"].arrays(library="ak")

    logger.info("Opening of root file trees with uproot terminated.")

    padded_tensor_23, padding_mask_23, pt_23 = awkward_to_padded_tensor(
        data_23, ["pT_23"]
    )
    padded_tensor_final, padding_mask_final = awkward_to_padded_tensor(
        data_final, ["pT_final"], list_pt=pt_23, truncate_pt=True
    )
    if padded_tensor_23.size(0) != padded_tensor_final.size(0):
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events, respectively "
            f"{padded_tensor_23.size(0)} and {padded_tensor_final.size(0)}."
        )

    logger.info("Padded tensors created.")

    for i in range(padded_tensor_23.size(0)):
        sum_23 = padded_tensor_23[i].sum()
        sum_final = padded_tensor_final[i].sum()
        if sum_final / sum_23 < 1:
            print(f"pT_final/pT_23: {sum_final/sum_23}")

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
    torch.save(train_23, "train_23_1M.pt")
    logger.info("Tensor train_23 saved")
    torch.save(train_final, "train_final_1M.pt")
    logger.info("Tensor train_final saved")
    torch.save(val_23, "val_23_1M.pt")
    logger.info("Tensor val_23 saved")
    torch.save(val_final, "val_final_1M.pt")
    logger.info("Tensor val_final saved")
    torch.save(test_23, "test_23_1M.pt")
    logger.info("Tensor test_23 saved")
    torch.save(test_final, "test_final_1M.pt")
    logger.info("Tensor test_final saved")
    torch.save(mask_train_23, "mask_train_23_1M.pt")
    logger.info("Tensor mask_train_23 saved")
    torch.save(mask_train_final, "mask_train_final_1M.pt")
    logger.info("Tensor mask_train_final saved")
    torch.save(mask_val_23, "mask_val_23_1M.pt")
    logger.info("Tensor mask_val_23 saved")
    torch.save(mask_val_final, "mask_val_final_1M.pt")
    logger.info("Tensor mask_val_final saved")
    torch.save(mask_test_23, "mask_test_23_1M.pt")
    logger.info("Tensor mask_test_23 saved")
    torch.save(mask_test_final, "mask_test_final_1M.pt")
    logger.info("Tensor mask_test_final saved")


def load_saved_dataloaders(batch_size):
    """
    Load tensors and forming the DataLoader objects.

    Args:
        batch_size (int): number of samples per batch in DataLoader.
    
    Returns:
        tuple: (
            loader_train,
            loader_val,
            loader_test,
            loader_padding_train,
            loader_padding_val,
            loader_padding_test,
            subset (torch.Tensor): example feature vector from train_23
        )
    """
    # Loading tensors.
    train_23 = torch.load("train_23_1M.pt")
    train_final = torch.load("train_final_1M.pt")
    val_23 = torch.load("val_23_1M.pt")
    val_final = torch.load("val_final_1M.pt")
    test_23 = torch.load("test_23_1M.pt")
    test_final = torch.load("test_final_1M.pt")

    if train_23.shape[0] != train_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in training sets, respectively "
            f"{train_23.shape[0]} and {train_final.shape[0]}."
        )
    if val_23.shape[0] != val_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in validation sets, respectively "
            f"{val_23.shape[0]} and {val_final.shape[0]}."
        )
    if test_23.shape[0] != test_final.shape[0]:
        raise ValueError(
            f"Status 23 particles tensor and final particles tensor do not "
            f"have the same number of events in test sets, respectively "
            f"{test_23.shape[0]} and {test_final.shape[0]}."
        )

    mask_train_23 = torch.load("mask_train_23_1M.pt")
    mask_train_final = torch.load("mask_train_final_1M.pt")
    mask_val_23 = torch.load("mask_val_23_1M.pt")
    mask_val_final = torch.load("mask_val_final_1M.pt")
    mask_test_23 = torch.load("mask_test_23_1M.pt")
    mask_test_final = torch.load("mask_test_final_1M.pt")

    # Rebuilding the DataLoaders.
    loader_train = batching(train_23, train_final, batch_size)
    loader_val = batching(val_23, val_final, batch_size)
    loader_test = batching(test_23, test_final, batch_size)

    loader_padding_train = batching(
        mask_train_23, mask_train_final, batch_size
    )
    loader_padding_val = batching(mask_val_23, mask_val_final, batch_size)
    loader_padding_test = batching(mask_test_23, mask_test_final, batch_size)

    # Extra information
    subset = train_23[0, 0, :]

    return (
        loader_train,
        loader_val,
        loader_test,
        loader_padding_train,
        loader_padding_val,
        loader_padding_test,
        subset,
    )

#######################################
#                MAIN                 #
#######################################

if __name__ == "__main__":
    load_and_save_tensor("events_1M.root")
