"""
data processing
"""
import numpy as np
import torch
import uproot

from pythia_generator import features_list

particle_list = ["23", "final"]

# Costruisci dinamicamente le chiavi per uproot
keys = {p: [f"{attr}_{p}" for attr in features_list] for p in particle_list}

# Apri il file ROOT
file = uproot.open("events.root")
tree = file["ParticleTree"]

# Leggi i dati per ciascuna particella
data = {p: tree.arrays(keys[p], library = "np") for p in particle_list}

# Prepara gli eventi in modo pythonico
inputs, outputs = [], []
for status23, final in zip(
    zip(*[data["23"][key] for key in keys["23"]]),
    zip(*[data["final"][key] for key in keys["final"]])
):
    # Crea tensori per input e output
    input_event = torch.tensor(np.column_stack(status23), dtype = torch.float32)
    output_event = torch.tensor(np.column_stack(final), dtype = torch.float32)
    
    inputs.append(input_event)
    outputs.append(output_event)

# Usa pad_sequence per gestire eventi di lunghezza variabile
inputs_tensor = torch.nn.utils.rnn.pad_sequence(inputs, batch_first = True)
outputs_tensor = torch.nn.utils.rnn.pad_sequence(outputs, batch_first = True)

print(f'tensore di input {inputs_tensor}')

# Crea un TensorDataset
dataset = torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)

# Salva il dataset
torch.save(dataset, "dataset.pt")
