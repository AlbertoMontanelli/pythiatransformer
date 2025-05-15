"""Process data from pythia_generator.py using Awkward Arrays.

This script reads ROOT files, preprocesses the data (standardization
and normalization), and converts them into Torch tensors, split
into training, validation, and test sets.

"""

import awkward as ak
import numpy as np
import torch

# Creiamo un awkward array di esempio
events = ak.Array(
    [
        {
            "feature1": [1.0, 2.0],
            "feature2": [0.1, 0.2],
            "feature 3": [-11, -55],
            "feature 4": [11, 55],
        },  # Evento 1
        {
            "feature1": [3.0],
            "feature2": [0.3],
            "feature 3": [-45],
            "feature 4": [35],
        },  # Evento 2
        {
            "feature1": [1.0, 2.0, 5.0],
            "feature2": [0.4, 0.5, 0.6],
            "feature 3": [-88, -77, -66],
            "feature 4": [41, 69, 26],
        },  # Evento 3
    ]
)


def tensor_convertion(events, selected_features):
    # Calcolare il numero massimo di particelle
    max_particles = ak.max(ak.num(events[selected_features[0]]))

    # Applicare il padding su ciascuna feature e raccoglierle in un dizionario
    padded_events = {
        feature: ak.fill_none(
            ak.pad_none(events[feature], target=max_particles, axis=1), 0
        )
        for feature in selected_features
    }
    print(padded_events)
    # Convertire le feature in NumPy array
    padded_arrays = [
        ak.to_numpy(padded_events[feature]) for feature in selected_features
    ]
    print(padded_arrays)
    # Combinare le feature lungo l'asse finale (axis=-1)
    padded_array = np.stack(padded_arrays, axis=-1)

    # Convertire in tensore Torch
    tensor = torch.tensor(padded_array, dtype=torch.float32)

    # Compute attention mask (1 for padding, 0 for actual particles)
    attention_mask = ak.num(events[selected_features[0]], axis=1)
    attention_mask = torch.tensor(
        [
            [0] * num + [1] * (padded_array.shape[1] - num)
            for num in attention_mask
        ],
        dtype=torch.bool,
    )
    return tensor, attention_mask


selected_features = ["feature1", "feature2", "feature 3", "feature 4"]
tensor, attention_mask = tensor_convertion(events, selected_features)


print(tensor)
print(attention_mask.shape)
print(
    tensor.shape
)  # Output: torch.Size([num_events, max_particles, num_features])
