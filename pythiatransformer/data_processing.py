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


# def standardize_features(data, features):
#     """Standardize features (mean=0, std=1) directly on Awkward Arrays.

#     Args:
#         data (ak.Array): Input Awkward Array.
#         features (list): List of features to standardize.

#     Returns:
#         tuple: (standardized data, list of means, list of stds)
#     """
#     means, stds = [], []
#     for feature in features:
#         mean = ak.mean(data[feature])
#         std = ak.std(data[feature])
#         data[feature] = (data[feature] - mean) / std
#         means.append(mean)
#         stds.append(std)
#     print(f"means: {means}, stds: {stds}")
#     return data, means, stds


def awkward_to_padded_tensor(data, features, truncate_energy = False, list_energy = None, eos_token=61, sos_token=62):
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
        cum_energy = torch.cumsum(padded_tensor.squeeze(-1), dim=1)  # somma cumulativa dei pT
        # per ogni evento trovo indice fino a dove somma cumulativa < soglia
        keep_lengths = (cum_energy < threshold.unsqueeze(1)).sum(dim=1) + 1 # +1 per includere la particella che supera 50%

        new_max_len = keep_lengths.max().item() + 2  # +2 per sos e eos token
        print(f"max len: {new_max_len}")

        padded_tensor_trunc = torch.zeros((batch_size, new_max_len, num_features))
        padding_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

        for i in range(batch_size):
            k = keep_lengths[i].item() # l'indice della particella dopo la quale tronchiamo
            if i < 100:
                print(f"k: {k}")
            padded_tensor_trunc[i, 0, 0] = sos_token
            padded_tensor_trunc[i, 1 : k + 1, :] = padded_tensor[i, :k, :]
            padded_tensor_trunc[i, k + 1, 0] = eos_token
            padding_mask[i, : k + 2] = 0

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


# def pdg_to_index(tensor, padding_mask):
#     dict_ids = {
#         21: 30,  # gluon
#         22: 31,  # photon
#         -12: 32,  # antineutrino e
#         12: 33,  # neutrino e
#         -14: 34,  # antineutrino mu
#         14: 35,  # neutrino mu
#         -16: 36,  # antineutrino tau
#         16: 37,  # neutrino tau
#         -11: 38,  # positron
#         11: 39,  # electron
#         -2: 40,  # antiup
#         2: 41,  # up
#         -1: 42,  # antidown
#         1: 43,  # down
#         -3: 44,  # antistrange
#         3: 45,  # strange
#         -13: 46,  # antimu
#         13: 47,  # mu
#         -211: 48,  # pi-
#         211: 49,  # pi +
#         -321: 50,  # K-
#         321: 51,  # K+
#         130: 52,  # K0 long
#         -2212: 53,  # antiproton
#         2212: 54,  # proton
#         -2112: 55,  # antineutron
#         2112: 56,  # neutron
#         -4: 57,  # anticharm
#         4: 58,  # charm
#         -5: 59,  # antibottom
#         5: 60,  # bottom
#     }
#     for pdg_id, index in dict_ids.items():
#         mask = tensor[:, :, 0] == pdg_id
#         tensor[:, :, 0][mask] = index
#     tensor[~padding_mask] = tensor[~padding_mask] - 29
#     tensor = tensor.long()
#     return tensor


def load_and_save_tensor(filename):

    logger.info("Beginning data_processing")

    with uproot.open(filename) as file:
        data_23 = file["tree_23"].arrays(library="ak")
        data_final = file["tree_final"].arrays(library="ak")

    logger.info("Opening of root file trees with uproot terminated")

    # data_23, mean_23, std_23 = standardize_features(
    #     data_23, ["e_23"]
    # )
    # data_final, mean_final, std_final = standardize_features(
    #     data_final, ["e_final"]
    # )
    # stats = np.array([mean_final, std_final])
    # np.savetxt("mean_std.txt", stats)

    # logger.info("Standardization and saving mean, std terminated")

    padded_tensor_23, padding_mask_23, energy_23 = awkward_to_padded_tensor(
        data_23, 
        ["pT_23"]
    )
    padded_tensor_final, padding_mask_final = awkward_to_padded_tensor(
        data_final,
        ["pT_final"],
        truncate_energy = True,
        list_energy = energy_23
    )

    logger.info("Padded tensors created")

    # drop_p_id = lambda t: t[:, :, :-2] # droppa pt e id
    # padded_tensor_23 = drop_p_id(padded_tensor_23)
    # padded_tensor_final = drop_p_id(padded_tensor_final)

    # logger.info("Dropping pT and id in tensor terminated")

    print("evento 0 per il 23:\n", padded_tensor_23[0, :, :])
    print("evento 0 per il finale:\n", padded_tensor_final[0, :, :])

    for i in range(100):
        sum_23 = padded_tensor_23[i].sum()
        sum_final = padded_tensor_final[i].sum() - 62 - 61
        # print(f"23: {sum_23}, final: {sum_final}")
        print(f"final/23: {sum_final/sum_23}")

    # padded_tensor_23 = pdg_to_index(padded_tensor_23, padding_mask_23)
    # padded_tensor_final = pdg_to_index(padded_tensor_final, padding_mask_final)

    # logger.info("Mapping PID into classes indices terminated")

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
    torch.save(train_23, "train_23_1M_7Gev.pt")
    logger.info("Tensor train_23 saved")
    torch.save(train_final, "train_final_1M_7Gev.pt")
    logger.info("Tensor train_final saved")
    torch.save(val_23, "val_23_1M_7Gev.pt")
    logger.info("Tensor val_23 saved")
    torch.save(val_final, "val_final_1M_7Gev.pt")
    logger.info("Tensor val_final saved")
    torch.save(test_23, "test_23_1M_7Gev.pt")
    logger.info("Tensor test_23 saved")
    torch.save(test_final, "test_final_1M_7Gev.pt")
    logger.info("Tensor test_final saved")
    torch.save(mask_train_23, "mask_train_23_1M_7Gev.pt")
    logger.info("Tensor mask_train_23 saved")
    torch.save(mask_train_final, "mask_train_final_1M_7Gev.pt")
    logger.info("Tensor mask_train_final saved")
    torch.save(mask_val_23, "mask_val_23_1M_7Gev.pt")
    logger.info("Tensor mask_val_23 saved")
    torch.save(mask_val_final, "mask_val_final_1M_7Gev.pt")
    logger.info("Tensor mask_val_final saved")
    torch.save(mask_test_23, "mask_test_23_1M_7Gev.pt")
    logger.info("Tensor mask_test_23 saved")
    torch.save(mask_test_final, "mask_test_final_1M_7Gev.pt")
    logger.info("Tensor mask_test_final saved")

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
    load_and_save_tensor("events_1M.root")

    # ===================== DEBUG: verifica EOS nei dati batchati ===============
    # inputs, targets = next(iter(loader_train))
    # _, targets_mask = next(iter(loader_padding_train))

    # id_pred = torch.argmax(targets[:, :, : len(dict_ids)], dim=-1)
    # eos_index = dict_ids[eos_token]
    # valid_mask = ~targets_mask
    # num_real = valid_mask.sum(dim=1)
    # last_index = num_real - 1
    # B = targets.size(0)
    # last_ids = id_pred[torch.arange(B), last_index]

    # num_eos = (last_ids == eos_index).sum().item()
    # print(f"\n DEBUG BATCH: {num_eos}/{B} eventi terminano con EOS")
    # assert torch.all(
    #     last_ids == eos_index
    # ), "Alcuni target batchati non terminano con EOS!"
    # print("Tutti i target nel batch terminano con EOS")


# ====================================================================================

# ROBA PER LO ONE HOT INUTILE ADESSO


#     def one_hot_encoding(tensor, dict_ids, num_classes, padding_token=0):
#     """Apply one-hot encoding to IDs in a tensor, handling EOS and padding.

#     Args:
#         tensor (torch.Tensor): Input tensor.
#         dict_ids (dict): Mapping of PDG IDs to one-hot indices.
#         num_classes (int): Total number of classes.
#         eos_token (int): EOS token.
#         padding_token (int): Token to treat as padding.

#     Returns:
#         torch.Tensor: One-hot-encoded tensor.
#     """
#     ids = tensor[:, :, 0].long()
#     one_hot = torch.zeros(tensor.size(0), tensor.size(1), num_classes)
#     for pdg_id, index in dict_ids.items():
#         one_hot[ids == pdg_id] = torch.nn.functional.one_hot(
#             torch.tensor(index), num_classes=num_classes
#         ).float()
#     one_hot[ids == padding_token] = 0
#     return one_hot


#     eos_token = -999
#     sos_token = -998
#     id_all = np.unique(np.concatenate([ak.flatten(data_23["id_23"]), ak.flatten(data_final["id_final"])]))
#     dict_ids = {int(pid): idx for idx, pid in enumerate(id_all)}
#     dict_ids[sos_token] = len(dict_ids)
#     dict_ids[eos_token] = len(dict_ids)
#     num_classes = len(dict_ids)

#     one_hot_23 = one_hot_encoding(padded_tensor_23, dict_ids, num_classes)
#     one_hot_final = one_hot_encoding(padded_tensor_final, dict_ids, num_classes)

#     padded_tensor_23 = torch.cat((one_hot_23, padded_tensor_23[:, :, 1:]), dim=-1)
#     padded_tensor_final = torch.cat((one_hot_final, padded_tensor_final[:, :, 1:]), dim=-1)
