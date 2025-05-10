import numpy as np
import torch
import torch.nn as nn
from transformer import ParticleTransformer
from main import build_model
from data_processing import loader_train
from data_processing import dict_ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Masses array; the index corresponds to the one-hot-encoding index.
masses = torch.tensor([
    0.93827, 0.93957, 0.49368, 0.13957, 0, 0, 0.10566, 0, 0.000511, 4.183, 1.273, 0.0935, 0.00216, 0.0047,
    0.0047, 0.00216, 0.0935, 1.273, 4.183, 0.000511, 0, 0.10566, 0, 0, 0, 0, 0.49761, 0.13957, 0.49368, 0.93957, 0.93827
    ], device=device)

transformer = build_model()
transformer.load_state_dict(torch.load("transformer_model_true.pt", map_location=device))
transformer.to(device)

outputs = []

transformer.eval()
with torch.no_grad():
    for (input, target), (input_padding_mask, target_padding_mask) in zip(transformer.train_data, transformer.train_data_pad_mask):

        input = input.to(device)
        target = target.to(device)
        input_padding_mask = input_padding_mask.to(device)
        target_padding_mask = target_padding_mask.to(device)

        target, target_padding_mask, attention_mask = transformer.de_padding(target, target_padding_mask)
        attention_mask = attention_mask.to(device)

        output = transformer.forward(input, target, input_padding_mask, target_padding_mask, attention_mask)
        outputs.append(output)


# Prepare output for fastjet.
for output in outputs:
    ids = output[:, :, :len(dict_ids)]
    index = torch.argmax(ids, dim=-1)
    mass = masses[index]
    output = output[:, :, len(dict_ids):]
    #mass = mass.unsqueeze(-1)
    #output = torch.cat((output, mass), dim=-1)
    energy = torch.sqrt(
        output[:,:,-1]**2 + output[:,:,-2]**2 + output[:,:,-3]**2 + mass**2
    )
    energy = energy.unsqueeze(-1)
    output = output[:, :, len(dict_ids):]
    output = torch.cat((output, energy), dim=-1)
    print(f"output: {output}")
    print(f"SHAPE: {output.shape}")


# Prepare target for fastjet
for i, batch in enumerate(loader_train):
    data, target = batch
    target = target.to(device)
    ids = target[:, :, :len(dict_ids)]
    index = torch.argmax(ids, dim=-1)
    mass = masses[index]
    target = target[:, :, len(dict_ids):]
    #mass = mass.unsqueeze(-1)
    #target = torch.cat((target, mass), dim=-1)
    energy = torch.sqrt(
        target[:,:,-1]**2 + target[:,:,-2]**2 + target[:,:,-3]**2 + mass**2
    )
    energy = energy.unsqueeze(-1)
    target = target[:, :, len(dict_ids):]
    target = torch.cat((target, energy), dim=-1)
    print(f"target: {target}")