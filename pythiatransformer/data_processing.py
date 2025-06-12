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

def awkward_to_padded_tensor(
    data,
    features,
    truncate_energy=False,
    list_energy=None,
    eos_token=61,
    sos_token=62,
):
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
    if truncate_energy:
        # tronchiamo in modo che la somma delle energie sia il 50% di quella delle 23
        batch_size = len(event_particles)

        threshold = 0.5 * list_energy
        cum_energy = torch.cumsum(
            padded_tensor.squeeze(-1), dim=1
        )  # somma cumulativa dei pT
        # per ogni evento trovo indice fino a dove somma cumulativa < soglia
        keep_lengths = (cum_energy < threshold.unsqueeze(1)).sum(
            dim=1
        ) + 1 # per includere la particella che supera 50%
        keep_lengths = torch.clamp(keep_lengths, max=max_particles)
        
        new_max_len = keep_lengths.max().item()
        print(f"max len: {new_max_len}")

        padded_tensor_trunc = torch.zeros(
            (batch_size, new_max_len, num_features)
        )
        padding_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

        for i in range(batch_size):
            k = keep_lengths[
                i
            ].item()  # l'indice della particella dopo la quale tronchiamo
            padded_tensor_trunc[i, :k, :] = padded_tensor[i, :k, :]
            padding_mask[i, :k] = 0

    else:
        padding_mask = torch.tensor(
            [[0] * n + [1] * (max_particles - n) for n in event_particles],
            dtype=torch.bool,
        )
        total_energy = padded_tensor.sum(dim=1).squeeze(-1)

    if truncate_energy:
        return padded_tensor_trunc, padding_mask
    else:
        return padded_tensor, padding_mask, total_energy


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

def load_and_save_tensor(filename):

    logger.info("Beginning data_processing")

    with uproot.open(filename) as file:
        data_23 = file["tree_23"].arrays(library="ak")
        data_final = file["tree_final"].arrays(library="ak")

    logger.info("Opening of root file trees with uproot terminated")

    padded_tensor_23, padding_mask_23, energy_23 = awkward_to_padded_tensor(
        data_23, ["pT_23"]
    )
    padded_tensor_final, padding_mask_final = awkward_to_padded_tensor(
        data_final, ["pT_final"], truncate_energy=True, list_energy=energy_23
    )

    logger.info("Padded tensors created")

    for i in range(padded_tensor_23.size(0)):
        sum_23 = padded_tensor_23[i].sum()
        sum_final = padded_tensor_final[i].sum()
        if sum_final/sum_23 < 1:
            print(f"final/23: {sum_final/sum_23}")

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

    # Salvataggio dei tensori per ripristino futuro
    torch.save(train_23, "1_train_23_1M.pt")
    logger.info("Tensor train_23 saved")
    torch.save(train_final, "1_train_final_1M.pt")
    logger.info("Tensor train_final saved")
    torch.save(val_23, "1_val_23_1M.pt")
    logger.info("Tensor val_23 saved")
    torch.save(val_final, "1_val_final_1M.pt")
    logger.info("Tensor val_final saved")
    torch.save(test_23, "1_test_23_1M.pt")
    logger.info("Tensor test_23 saved")
    torch.save(test_final, "1_test_final_1M.pt")
    logger.info("Tensor test_final saved")
    torch.save(mask_train_23, "1_mask_train_23_1M.pt")
    logger.info("Tensor mask_train_23 saved")
    torch.save(mask_train_final, "1_mask_train_final_1M.pt")
    logger.info("Tensor mask_train_final saved")
    torch.save(mask_val_23, "1_mask_val_23_1M.pt")
    logger.info("Tensor mask_val_23 saved")
    torch.save(mask_val_final, "1_mask_val_final_1M.pt")
    logger.info("Tensor mask_val_final saved")
    torch.save(mask_test_23, "1_mask_test_23_1M.pt")
    logger.info("Tensor mask_test_23 saved")
    torch.save(mask_test_final, "1_mask_test_final_1M.pt")
    logger.info("Tensor mask_test_final saved")


def load_saved_dataloaders(batch_size):
    """Load tensors salvati da file .pt e ricrea i DataLoader."""

    # Caricamento tensori da file .pt
    train_23 = torch.load("1_train_23_1M.pt")
    train_final = torch.load("1_train_final_1M.pt")
    val_23 = torch.load("1_val_23_1M.pt")
    val_final = torch.load("1_val_final_1M.pt")
    test_23 = torch.load("1_test_23_1M.pt")
    test_final = torch.load("1_test_final_1M.pt")

    mask_train_23 = torch.load("1_mask_train_23_1M.pt")
    mask_train_final = torch.load("1_mask_train_final_1M.pt")
    mask_val_23 = torch.load("1_mask_val_23_1M.pt")
    mask_val_final = torch.load("1_mask_val_final_1M.pt")
    mask_test_23 = torch.load("1_mask_test_23_1M.pt")
    mask_test_final = torch.load("1_mask_test_final_1M.pt")

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
    load_and_save_tensor("events_1M.root")
    