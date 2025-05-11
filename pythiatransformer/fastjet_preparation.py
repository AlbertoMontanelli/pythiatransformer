import numpy as np
import torch
import torch.nn as nn

from main import build_model
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

def outputs_targets_tensor_list(model, device):
    """
    Esegue la forward su tutti i batch nel train set e restituisce
    due liste: outputs e targets, entrambi depaddati e pronte per fastjet.

    Args:
        model (ParticleTransformer): modello già caricato.
        device (torch.device): cpu o cuda.

    Returns:
        outputs_all (list[Tensor]): Lista di batch depaddati di output.
        targets_all (list[Tensor]): Lista di batch depaddati di target.
    """
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        for (input, target), (input_mask, target_mask) in zip(
            model.train_data, model.train_data_pad_mask
        ):
            input = input.to(device)
            target = target.to(device)
            input_mask = input_mask.to(device)
            target_mask = target_mask.to(device)

            target, target_mask, attn_mask = model.de_padding(target, target_mask)
            attn_mask = attn_mask.to(device)

            output = model.forward(input, target, input_mask, target_mask, attn_mask)

            outputs.append(output)
            targets.append(target)

    return outputs, targets


def tensor_convertion(batches, dict_ids, masses, device=None):
    """
    Converte una lista di batch (output o target) in una lista di tensori con [px, py, pz, E].

    Args:
        batches (list[Tensor]): lista di batch (ognuno [B, Nᵢ, F])
        dict_ids (dict): dizionario one-hot degli ID.
        masses (torch.Tensor): tensore delle masse ordinate come one-hot.
        device (torch.device, optional): se specificato, sposta i tensori sul device.

    Returns:
        list[Tensor]: lista di tensori [B, Nᵢ, 4]
    """
    result = []
    for batch in batches:
        if device:
            batch = batch.to(device)

        ids = batch[:, :, :len(dict_ids)]
        index = torch.argmax(ids, dim=-1)
        mass = masses[index]

        momentum = batch[:, :, len(dict_ids):len(dict_ids)+3]
        px, py, pz = momentum.unbind(-1)

        energy = torch.sqrt(px**2 + py**2 + pz**2 + mass**2).unsqueeze(-1)
        result.append(torch.cat([momentum, energy], dim=-1))

    return result

outputs_list, targets_list = outputs_targets_tensor_list(transformer, device)

outputs_fastjet = tensor_convertion(outputs_list, dict_ids, masses, device)
targets_fastjet = tensor_convertion(targets_list, dict_ids, masses, device)

if __name__== "__main__":
    # Stampa di controllo
    for (output,target) in zip(outputs_fastjet, targets_fastjet):
        print(f"output: {output[0, 0:2, :]}")
        print(f"target: {target[0, 0:2, :]}")
        break