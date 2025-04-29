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
    for feature in features:
        mean = ak.mean(data[feature])
        std = ak.std(data[feature])
        data[feature] = (data[feature] - mean) / std
    return data

def awkward_to_padded_tensor(data, features, isTarget, eos_value=-999):
    """Convert Awkward Array to padded Torch tensor.

        Args:
            data (ak.Array): Input Awkward Array.
            feature_cols (list): List of feature columns to stack.
            eos_value (float): Value for the EOS token (default: 0).

        Returns:
            padded_tensor (torch.Tensor): Padded tensor of  shape:
                                          (num_events, max_particles,
                                          num_features).
            attention_mask (torch.Tensor): Attention mask (0 for actual,
                                           1 for padding).
    """
    # Find max number of particles for all the events.
    event_particles = ak.num(data[features[0]], axis=1)  # Number of real particles per event
    max_particles = ak.max(event_particles)
    print(f"max number of particles: {max_particles}")
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
    
    # Sorting the padded tensor with ascending order with respect to pT
    indices_data = torch.argsort(padded_tensor[:, :, -1], dim=1, descending=True) #shape: batch_size x nr_particelle
    # e salva numeri corrispondenti all'ordine della feature pT lungo la dim del nr di particelle
    # ex. ( [3, 4, 2, 1], batch 1 da 4 particelle
    #       [1, 4, 2, 3]) batch 2 da 4 particelle
    padded_tensor_sorted = torch.gather(
        padded_tensor, 
        dim=1, 
        index=indices_data.unsqueeze(-1).expand(-1, -1, padded_tensor.size(-1))
        # unsqueeze: aumenta di 1 la dim->shape: batch_size x nr particelle x 1
        # expand: espande la dimensione aggiunta fino al nr di features (duplicando i valori) 
        # lasciando le altre invariate.
        # index è un tensore della stessa forma di padded_tensor, gather dice di prendere
        # gli elementi di padded_tensor nell'ordine specificato dai valori degli elementi di index
    )
    if isTarget:
        # Initialize EOS tensor and attention mask (1 for padding, 0 for actual particles).
        batch_size, max_len, num_features = padded_tensor_sorted.shape
        new_max_len = max_len + 1
        padded_tensor = torch.zeros((batch_size, new_max_len, num_features))
        attention_mask = torch.ones((batch_size, new_max_len), dtype=torch.bool)

        # Insert EOS token after last real particle in each event
        for i, true_particles in enumerate(event_particles):
            true_particles = true_particles.item()  # Length of real particles for event `i`
            
            # Copy real particles
            padded_tensor[i, :true_particles, :] = padded_tensor_sorted[i, :true_particles, :]
            attention_mask[i, :true_particles] = 0  # Valid tokens

            # Insert EOS token
            padded_tensor[i, true_particles, :] = eos_value  # EOS token (default: all zeros)
            attention_mask[i, true_particles] = 0  # Mark EOS as a valid token

            # Remaining positions stay as padding (default values)
    else:
        padded_tensor = padded_tensor_sorted.clone()
        attention_mask = torch.tensor(
        [[0] * num + [1] * (padded_array.shape[1] - num) for num in event_particles],
        dtype=torch.bool
        )


    return padded_tensor, attention_mask

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
        generator = generator if shuffle else None
    )

    return loader

def one_hot_encoding(tensor, dict_ids, num_classes):
    """One-hot-encoding of the ids.

        Args:
            tensor (torch.Tensor): input tensor.
            dict_ids (dict): dictionary having as keys the PDG ids and
                             as values the integers ranging from 0 to
                             the number of different ids.
            num_classes (int): maximum number of different ids.

        Returns:
            one_hot (torch.Tensor): one-hot-encoded tensor of the ids. 
    """
    # Convert id from float type to long int type.
    tensor_ids = tensor[:, :, 0].long()
    # Create a tensor_ids clone.
    indices = tensor_ids.clone()
    # When the content of tensor_ids is equal to the key of dict_ids
    # (pdg_id), the content of indices tensor is overwritten with the
    # value (index) corresponding to pdg_id in the dictionary.
    for pdg_id, index in dict_ids.items():
        indices[tensor_ids == pdg_id] = index
    # Create one-hot-encoding for the ids.
    one_hot = torch.nn.functional.one_hot(
        indices, num_classes=num_classes
    ).float()
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

# Standardization.
data_23 = standardize_features(
    data_23, 
    features=["px_23", "py_23", "pz_23"]
)
data_final = standardize_features(
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
padded_tensor_23, attention_mask_23 = awkward_to_padded_tensor(
    data_23,
    features=["id_23", "px_23", "py_23", "pz_23", "pT_23"],
    isTarget=False
)
padded_tensor_final, attention_mask_final = awkward_to_padded_tensor(
    data_final,
    features=["id_final", "px_final", "py_final", "pz_final", "pT_final"],
    isTarget=True
)

print(f"padded tensor final shape: {padded_tensor_final.shape}")
print(f"attention mask final shape: {attention_mask_final.shape}")
print(f"padded tensor 23 shape: {padded_tensor_23.shape}")
print(f"attention mask 23 shape: {attention_mask_23.shape}")

# Finding unique ids for one_hot_encoding() function.
id_23 = np.unique(ak.flatten(data_23["id_23"]))
id_final = np.unique(ak.flatten(data_final["id_final"]))
id_all = np.unique(np.concatenate([id_23, id_final]))

# One-hot dictionary
dict_ids = {pdg_id.item(): index for index, pdg_id in enumerate(id_all)}
padding_index = len(id_all)      # Ultima posizione per il padding
eos_index = len(id_all) + 1      # Posizione successiva per EOS

dict_ids[0] = padding_index      # Padding
dict_ids[-999] = eos_index         # EOS (usa -999 per esempio)

num_classes = len(id_all) + 2    # Aggiungi padding ed EOS
print(f"Dizionario aggiornato:\n{dict_ids}")

import torch.nn.functional as F

def one_hot_encoding_with_eos_and_padding(tensor, dict_ids, num_classes):
    """
    One-hot encoding con gestione speciale per EOS e Padding.

    Args:
        tensor (torch.Tensor): Tensor con ID da convertire.
        dict_ids (dict): Dizionario di mapping (ID -> indice).
        num_classes (int): Numero totale di classi.
        
    Returns:
        one_hot (torch.Tensor): Tensor one-hot encoded.
    """
    # 1. Mappa il tensor secondo il dizionario
    mapped_tensor = tensor.clone()  # Clona per sicurezza
    for value, index in dict_ids.items():
        mapped_tensor[tensor == value] = index

    # 2. One-hot encoding
    one_hot = F.one_hot(mapped_tensor, num_classes=num_classes).float()

    # 3. Gestione speciale per il padding (tutti 0)
    padding_index = dict_ids.get(0)  # Indice del padding
    if padding_index is not None:
        padding_mask = tensor == 0  # Trova i token di padding
        one_hot[padding_mask] = 0   # Imposta tutto a 0 per il padding

    return one_hot

# One-hot encoding per padded_tensor_23 e padded_tensor_final
one_hot_23 = one_hot_encoding_with_eos_and_padding(padded_tensor_23, dict_ids, num_classes)
one_hot_final = one_hot_encoding_with_eos_and_padding(padded_tensor_final, dict_ids, num_classes)

padded_tensor_final = torch.cat((one_hot_final, padded_tensor_final[:, :, 1:]), dim=-1)
padded_tensor_23 = torch.cat((one_hot_23, padded_tensor_23[:, :, 1:]), dim=-1)

print(f"padded tensor final shape after 1he: {padded_tensor_final.shape}")
print(f"attention mask final shape after 1he: {attention_mask_final.shape}")
print(f"padded tensor 23 shape after 1he: {padded_tensor_23.shape}")
print(f"attention mask 23 shape after 1he: {attention_mask_23.shape}")

print(f"event tensor 23 {padded_tensor_23[0, :, -1]}")
print(f"event attention mask 23 {attention_mask_23[0, :]}")
print(f"event tensor final {padded_tensor_final[0, :, -1]}")
print(f"event attention mask final {attention_mask_final[0, :]}")

torch.set_printoptions(threshold=torch.inf)
for i in range(padded_tensor_final.shape[1]):
        print(f"event tensor final {padded_tensor_final[0, i, :35]}")
        print(f"event attention mask final {attention_mask_final[0, i]}")
        print("\n")

# Splitting the data.
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

# Building the data loaders.
loader_train = batching(training_set_23, training_set_final)
loader_attention_train = batching(attention_train_23, attention_train_final)
loader_val = batching(validation_set_23, validation_set_final)
loader_attention_val = batching(attention_val_23, attention_val_final)
loader_test = batching(test_set_23, test_set_final)
loader_attention_test = batching(attention_test_23, attention_test_final)
