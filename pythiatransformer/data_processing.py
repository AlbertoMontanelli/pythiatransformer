"""In this code the data from pythia_generator.py is processed.
First, the ROOT files are imported as awkward arrays.
Then, the features are standardized.
Last, the data is converted to Torch tensors and split
between training, validation and test sets.
"""
import awkward as ak
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import uproot


def standardize_features(data, features):
    """Standardize features (mean=0, std=1) directly on Awkward Arrays.

        Args:
            data (ak.Array): Input Awkward Array.
            feature (list): Features to standardize.

        Returns:
            data (ak.Array): Standardized Awkward Array.
    """
    means = []
    stds = []
    for feature in features:
        mean = ak.mean(data[feature])
        std = ak.std(data[feature])
        means.append(mean)
        stds.append(std)
        data[feature] = (data[feature] - mean) / std
    return data, means, stds

def awkward_to_padded_targets(data, features, eos_token=-999):
    """Convert Awkward Array to padded Torch tensor.

        Args:
            data (ak.Array): Input Awkward Array.
            feature_cols (list): List of feature columns to stack.
            eos_value (float): Value for the EOS token (default: 0).

        Returns:
            padded_tensor (torch.Tensor): Padded tensor of  shape:
                                          (num_events, max_particles,
                                          num_features).
            padding_mask (torch.Tensor): padding mask (0 for actual,
                                           1 for padding).
    """
    # Find max number of particles for all the events.
    event_particles = ak.num(data[features[0]], axis=1)  # Number of real particles per event
    max_particles = ak.max(event_particles)
    batch_size = len(event_particles)
    num_features = len(features)
    new_max_len = max_particles + 1
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
    base_tensor = torch.tensor(padded_array, dtype=torch.float32)
    
    # Sorting the padded tensor with ascending order with respect to pT
    indices = torch.argsort(base_tensor[:, :, -1], dim=1, descending=True) #shape: batch_size x nr_particelle
    # e salva numeri corrispondenti all'ordine della feature pT lungo la dim del nr di particelle
    # ex. ( [3, 4, 2, 1], batch 1 da 4 particelle
    #       [1, 4, 2, 3]) batch 2 da 4 particelle
    base_tensor_sorted = torch.gather(
        base_tensor, 
        dim=1, 
        index=indices.unsqueeze(-1).expand(-1, -1, num_features)
        # unsqueeze: aumenta di 1 la dim->shape: batch_size x nr particelle x 1
        # expand: espande la dimensione aggiunta fino al nr di features (duplicando i valori) 
        # lasciando le altre invariate.
        # index è un tensore della stessa forma di base_tensor, gather dice di prendere
        # gli elementi di base_tensor nell'ordine specificato dai valori degli elementi di index
    )
    # Initialize EOS tensor and padding mask (1 for padding, 0 for actual particles).
    padded_tensor = torch.zeros((batch_size, new_max_len, num_features))
    padding_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

    # Insert EOS token after last real particle in each event
    for i, true_particles in enumerate(event_particles):
        true_particles = true_particles.item()  # Length of real particles for event `i`
        
        # Copy real particles
        padded_tensor[i, :true_particles, :] = base_tensor_sorted[i, :true_particles, :]
        padding_mask[i, :true_particles] = 0  # Valid tokens

        # Inserisci token EOS: ID = eos_token, resto = 0
        eos_vector = torch.zeros(num_features)
        eos_vector[0] = eos_token
        padded_tensor[i, true_particles, :] = eos_vector
        padding_mask[i, true_particles] = 0

        # Remaining positions stay as padding (default values)


    return padded_tensor, padding_mask

def awkward_to_padded_inputs(data, features):
    """
    Convert Awkward array to padded tensor (no EOS).

    Args:
        data (ak.Array): Input Awkward Array.
        features (list): List of features to use.

    Returns:
        padded_tensor (Tensor): [B, N, F]
        padding_mask (BoolTensor): [B, N], 1 = padding
    """
    event_particles = ak.num(data[features[0]], axis=1)
    max_particles = ak.max(event_particles)
    batch_size = len(event_particles)
    num_features = len(features)

    padded_events = {
        feature: ak.fill_none(ak.pad_none(data[feature], target=max_particles, axis=1), 0)
        for feature in features
    }

    padded_arrays = [ak.to_numpy(padded_events[feature]) for feature in features]
    base_tensor = torch.tensor(np.stack(padded_arrays, axis=-1), dtype=torch.float32)

    # Ordina per pT se è l’ultima feature
    indices = torch.argsort(base_tensor[:, :, -1], dim=1, descending=True)
    padded_tensor = torch.gather(
        base_tensor,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, num_features)
    )

    padding_mask = torch.tensor(
        [[0] * n + [1] * (max_particles - n) for n in event_particles], dtype=torch.bool
    )

    return padded_tensor, padding_mask

def batching(input, target, shuffle = True, batch_size = 100):
    """This function prepares the data for training by splitting it
    into batches and shuffling the training data.

    Args:
        input (torch.Tensor): data.
        target (torch.Tensor): target.
        shuffle (bool): if True, shuffling along the rows.
        batch_size (int): size of the batches.
    Returns:
        loader (Iterator): An iterator for the data, with batching
                           and shuffling enabled.

    """
    seed = 1
    generator = torch.Generator() # creation of a new generator
    generator.manual_seed(seed)
    set = TensorDataset(input, target)

    loader = DataLoader(
        set,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers=2,  # Riduci il numero di worker in parallelo per diminuire il consumo di RAM
        pin_memory=True, # prealloca i batch nella memoria bloccata del CPU: il trasferimento 
                         # sulla GPU sarà più veloce
        generator = generator if shuffle else None
    )

    return loader

def one_hot_encoding(tensor, dict_ids, num_classes, eos_token=-999, padding_token=0):
    """One-hot-encoding of the ids, with EOS and padding handled correctly.

    Args:
        tensor (torch.Tensor): Input tensor.
        dict_ids (dict): Dictionary mapping PDG ids to integers.
        num_classes (int): Total number of IDs including EOS.
        eos_token (int): Special value used for EOS.
        padding_token (int): Special value used for padding.

    Returns:
        one_hot (torch.Tensor): One-hot-encoded tensor of shape (B, N, num_classes).
    """
    # Convert id from float to int
    tensor_ids = tensor[:, :, 0].long()

    # Inizializza tutto a zero
    one_hot = torch.zeros(
        tensor.shape[0], tensor.shape[1], num_classes, dtype=torch.float
    )

    # Codifica normale con il dizionario
    for pdg_id, index in dict_ids.items():
        mask = (tensor_ids == pdg_id)
        one_hot[mask] = torch.nn.functional.one_hot(
            torch.tensor(index), num_classes=num_classes
        ).float()

    # Padding token = tutto 0 (già zero per default)
    one_hot[tensor_ids == padding_token] = 0

    return one_hot


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

with uproot.open("events.root") as file:
    data_23 = file["tree_23"].arrays(library="ak")
    data_final = file["tree_final"].arrays(library="ak")

# pt_ordered = np.argsort(-np.sqrt(np.array(data_final["px_final"][0]**2) + np.array(data_final["py_final"][0]**2)))
# top2_indx = pt_ordered[:2]
# for i in top2_indx:
#     print(f"px particella target non std: {data_final['px_final'][0][i]}")

# Standardization.
data_23, mean_23, std_23 = standardize_features(
    data_23, 
    features=["px_23", "py_23", "pz_23"]
)
data_final, mean_final, std_final = standardize_features(
    data_final, 
    features=["px_final", "py_final", "pz_final"]
)
# Calculating pT.
data_23["pT_23"] = np.sqrt(
    data_23["px_23"]**2 + data_23["py_23"]**2
)
data_final["pT_final"] = np.sqrt(
    data_final["px_final"]**2 + data_final["py_final"]**2
)

# Padding.
padded_tensor_23, padding_mask_23 = awkward_to_padded_inputs(
    data_23,
    features=["id_23", "px_23", "py_23", "pz_23", "pT_23"]
)
padded_tensor_final, padding_mask_final = awkward_to_padded_targets(
    data_final,
    features=["id_final", "px_final", "py_final", "pz_final", "pT_final"]
)


# Dropping pT
padded_tensor_23 = padded_tensor_23[:, :, :-1]
padded_tensor_final = padded_tensor_final[:, :, :-1]

######################### EOS CHECKING ###########################################
eos_token = -999
id_channel = padded_tensor_final[:, :, 0]  # ID si trova sempre nella prima feature
# posizione EOS per ciascun evento = ultima particella valida (padding_mask = 0)
valid_mask = ~padding_mask_final
num_real = valid_mask.sum(dim=1)
last_valid_index = num_real - 1

# estrai l'ID all'ultima posizione valida per ciascun evento
eos_ids = id_channel[torch.arange(id_channel.size(0)), last_valid_index]

# verifica che siano tutti uguali a eos_token (-999)
assert torch.all(eos_ids == eos_token), "ERRORE: alcuni target non terminano con EOS!"

print("Verifica EOS: tutti gli eventi terminano correttamente con il token EOS.")

first_eos_vec = padded_tensor_final[0, last_valid_index[0]]
print("Vettore EOS del primo evento (prima del one-hot):")
print(first_eos_vec)

######################################################################################

# Finding unique ids for one_hot_encoding() function.
id_23 = np.unique(ak.flatten(data_23["id_23"]))
id_final = np.unique(ak.flatten(data_final["id_final"]))
m_final = np.unique(ak.flatten(data_final["m_final"]))
np.set_printoptions(threshold=np.inf)

# print("id_final", np.array(id_final))
# print("m_final", np.array(m_final))

np.set_printoptions(threshold=1000)
id_all = np.unique(np.concatenate([id_23, id_final]))

# One-hot dictionary
dict_ids = {pdg_id.item(): index for index, pdg_id in enumerate(id_all)}
eos_token = -999
eos_index = len(dict_ids)        # EOS dopo tutti gli ID noti
dict_ids[eos_token] = eos_index
num_classes = len(dict_ids)     # include EOS
print(f"dizionario one hot encoding: {dict_ids}")

# One-hot encoding per padded_tensor_23 e padded_tensor_final
one_hot_23 = one_hot_encoding(padded_tensor_23, dict_ids, num_classes, eos_token=eos_token)
one_hot_final = one_hot_encoding(padded_tensor_final, dict_ids, num_classes, eos_token=eos_token)


padded_tensor_final = torch.cat((one_hot_final, padded_tensor_final[:, :, 1:]), dim=-1)
padded_tensor_23 = torch.cat((one_hot_23, padded_tensor_23[:, :, 1:]), dim=-1)

print(f"padded tensor final shape after 1he: {padded_tensor_final.shape}")
print(f"padding mask final shape after 1he: {padding_mask_final.shape}")
print(f"padded tensor 23 shape after 1he: {padded_tensor_23.shape}")
print(f"padding mask 23 shape after 1he: {padding_mask_23.shape}")

#for i in range(padded_tensor_final.shape[1]):
#        print(f"event tensor final {padded_tensor_final[0, i, :]}")
#        print(f"event padding mask final {padding_mask_final[0, i]}")
#        print("\n")

# Splitting the data.
training_set_23, validation_set_23, test_set_23 = (
    train_val_test_split(padded_tensor_23)
)
padding_train_23, padding_val_23, padding_test_23 = (
    train_val_test_split(padding_mask_23)
)
training_set_final, validation_set_final, test_set_final = (
    train_val_test_split(padded_tensor_final)
)
padding_train_final, padding_val_final, padding_test_final = (
    train_val_test_split(padding_mask_final)
)

# Building the data loaders.
loader_train = batching(training_set_23, training_set_final)
loader_padding_train = batching(padding_train_23, padding_train_final)
loader_val = batching(validation_set_23, validation_set_final)
loader_padding_val = batching(padding_val_23, padding_val_final)
loader_test = batching(test_set_23, test_set_final)
loader_padding_test = batching(padding_test_23, padding_test_final)

subset = training_set_23[0, 0, :]
