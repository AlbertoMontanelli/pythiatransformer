from collections import defaultdict
import fastjet as fj
import numpy as np
import torch

from data_processing import dict_ids
from fastjet_preparation import outputs_targets_fastjet, fastjet_tensor
from main import build_model

def clustering(model, device, data, data_pad_mask):

    outputs, outputs_mask, targets, targets_mask = outputs_targets_fastjet(
        model, device,
        data, data_pad_mask
    )
    outputs_fastjet = fastjet_tensor(
        outputs, dict_ids, device
    )
    targets_fastjet = fastjet_tensor(
        targets, dict_ids, device
    )
    
    # Jet clustering algorithm
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clustered_outputs = []
    clustered_targets = []

    for output, target, output_mask, target_mask in zip(
        outputs_fastjet, targets_fastjet, outputs_mask, targets_mask
    ):
        batch_size = output.shape[0]

        for i in range(batch_size):  # loop su eventi nel batch
            particles_output = output[i]
            particles_target = target[i]

            pseudojets_output = [
                fj.PseudoJet(*particles_output[j].tolist())
                for j in range(particles_output.shape[0])
                if not output_mask[j]
            ]

            pseudojets_target = [
                fj.PseudoJet(*particles_target[j].tolist())
                for j in range(particles_target.shape[0])
                if not target_mask[j]
            ]

            clustered_outputs.append(fj.ClusterSequence(pseudojets_output, jet_def))
            clustered_targets.append(fj.ClusterSequence(pseudojets_target, jet_def))

    return clustered_outputs, clustered_targets

def compute_jet_differences(clustered_outputs, clustered_targets, n_jets=3):
    """
    Confronta i primi n_jets evento per evento tra output e target
    e calcola le differenze delle osservabili fisiche.

    Args:
        clustered_outputs (list[ClusterSequence])
        clustered_targets (list[ClusterSequence])
        n_jets (int): numero massimo di jet per evento da confrontare

    Returns:
        dict: dizionario con liste globali di differenze (ΔpT, Δeta, Δphi, Δmass, Δnconst)
    """
    diffs = defaultdict(list)

    for out_seq, tar_seq in zip(clustered_outputs, clustered_targets):
        jets_out = sorted(out_seq.inclusive_jets(), key=lambda j: -j.pt())
        jets_tar = sorted(tar_seq.inclusive_jets(), key=lambda j: -j.pt())

        # confronta i jet in ordine (i più energetici)
        for i in range(min(n_jets, len(jets_out), len(jets_tar))):
            jo = jets_out[i]
            jt = jets_tar[i]

            # differenze
            diffs["delta_pt"].append(jo.pt() - jt.pt())
            diffs["delta_eta"].append(jo.eta() - jt.eta())
            diffs["delta_phi"].append(wrap_delta_phi(jo.phi(), jt.phi()))
            diffs["delta_mass"].append(jo.m() - jt.m())
            diffs["delta_nconst"].append(len(jo.constituents()) - len(jt.constituents()))

    return diffs


def wrap_delta_phi(phi1, phi2):
    """
    Calcola Δphi in [-π, π]
    """
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi

import matplotlib.pyplot as plt

def plot_differences(differences, variable, bins=50):
    plt.hist(differences[f"delta_{variable}"], bins=bins, alpha=0.7, edgecolor='black')
    plt.title(f"Δ{variable} tra output e target")
    plt.xlabel(f"Δ{variable}")
    plt.ylabel("Conteggi")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    from main import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = build_model()
    transformer.load_state_dict(torch.load("transformer_model_true.pt", map_location=device))
    transformer.to(device)

    clustered_outputs, clustered_targets = clustering(
        transformer, device, transformer.train_data, transformer.train_data_pad_mask
    )

    diffs = compute_jet_differences(clustered_outputs, clustered_targets, n_jets=3)
    plot_differences(diffs, "pt")
    plot_differences(diffs, "mass")
    plot_differences(diffs, "eta")
    plot_differences(diffs, "phi")
