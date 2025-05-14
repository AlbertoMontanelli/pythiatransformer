import torch

from data_processing import dict_ids
from data_processing import mean_final, std_final

def de_standardization(data, data_padding_mask, index, mean, std):
    for tensor, mask in zip(data, data_padding_mask):
        tensor[:, :, index][~mask] = tensor[:, :, index][~mask] * std + mean
    return data

def outputs_computing(model, device, data, data_pad_mask):
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
    outputs_mask = []
    targets = []
    targets_mask = []
    print("inizio")

    with torch.no_grad():
        for (input, target), (input_mask, target_mask) in zip(
            data, data_pad_mask
        ):
            print("batch")
            input = input.to(device)
            target = target.to(device)
            input_mask = input_mask.to(device)
            target_mask = target_mask.to(device)

            target, target_mask, attn_mask = model.de_padding(
                target, target_mask
            )
            attn_mask = attn_mask.to(device)

            output = model.forward(
                input, target, input_mask, target_mask, attn_mask
            )

            outputs.append(output)
            outputs_mask.append(target_mask)
            targets.append(target)
            targets_mask.append(target_mask)

    return outputs, outputs_mask, targets, targets_mask


def fastjet_tensor_preparing(batches, dict_ids, device=None):
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

    # Masses array; the index corresponds to the one-hot-encoding index.
    # The last one is refered to EOS.
    masses = torch.tensor(
        [0.93827, 0.93957, 0.49368, 0.13957, 0, 0, 0.10566, 0, 0.000511,
        4.183, 1.273, 0.0935, 0.00216, 0.0047, 0.0047, 0.00216, 0.0935,
        1.273, 4.183, 0.000511, 0, 0.10566, 0, 0, 0, 0, 0.49761, 0.13957,
        0.49368, 0.93957, 0.93827, 0], device=device
    )

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

if __name__== "__main__":

    from main import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = build_model()
    transformer.load_state_dict(
        torch.load("transformer_model_eos.pt", map_location=device)
    )
    transformer.to(device)

    outputs, outputs_mask, targets, targets_mask = outputs_computing(
        transformer, device,
        transformer.train_data, transformer.train_data_pad_mask
    )

    for (output,target) in zip(outputs, targets):
        #print(f"output: {output[0, 0:2, :]}")
        print(f"target!!!!: {target[0, 0:2, :]}")
        break

    print(f"media: {mean_final}")
    print(f"std: {std_final}")
    
    outputs = de_standardization(outputs, outputs_mask, -1, mean_final[2], std_final[2])

    outputs = de_standardization(outputs, outputs_mask, -2, mean_final[1], std_final[1])

    outputs = de_standardization(outputs, outputs_mask, -3, mean_final[0], std_final[0])

    targets = de_standardization(targets, targets_mask, -1, mean_final[2], std_final[2])

    targets = de_standardization(targets, targets_mask, -2, mean_final[1], std_final[1])

    targets = de_standardization(targets, targets_mask, -3, mean_final[0], std_final[0])
    
    for (output,target) in zip(outputs, targets):
        #print(f"output: {output[0, 0:2, :]}")
        print(f"target ricalcolati: {target[0, 0:2, :]}")
        break

    outputs_fastjet = fastjet_tensor_preparing(
        outputs, dict_ids, device
    )
    targets_fastjet = fastjet_tensor_preparing(
        targets, dict_ids, device
    )

    # Stampa di controllo
    for (output,target) in zip(outputs_fastjet, targets_fastjet):
        print(f"output: {output[0, 0:2, :]}")
        print(f"target: {target[0, 0:2, :]}")
        break
