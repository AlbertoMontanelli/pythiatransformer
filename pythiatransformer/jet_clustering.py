import fastjet as fj

from fastjet_preparation import outputs_targets_fastjet, fastjet_tensor
from main import build_model

def clustering(model, device, outputs_pad_mask, targets_pad_mask):

    outputs, outputs_mask, targets, target_masks = outputs_targets_fastjet(
        transformer, device,
        transformer.train_data, transformer.train_data_pad_mask
    )
    outputs_fastjet = fastjet_tensor(
        outputs, dict_ids, masses, device
    )
    targets_fastjet = fastjet_tensor(
        targets, dict_ids, masses, device
    )
    
    # Jet clustering algorithm
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clustered_outputs = []
    clustered_targets = []

    for output, target, output_mask, target_mask in zip(
        outputs_fastjet, targets_fastjet, outputs_mask, target_masks
    ):
        batch_size = output.shape[0]
        # output_mask = ~output_mask
        # target_mask = ~target_mask

        for i in range(batch_size):  # loop su eventi nel batch
            particles_output = output[i]
            particles_target = target[i]

            pseudojets_output = [
                fj.PseudoJet(*particles_output[j].tolist())
                for j in range(particles_output.shape[0])
                if ~output_mask[j]
            ]

            pseudojets_target = [
                fj.PseudoJet(*particles_target[j].tolist())
                for j in range(particles_target.shape[0])
                if ~target_mask[j]
            ]

            clustered_outputs.append(fj.ClusterSequence(pseudojets_output, jet_def))
            clustered_targets.append(fj.ClusterSequence(pseudojets_target, jet_def))

    return clustered_outputs, clustered_targets

if __name__ == "__main__":

    from main import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = build_model()
    transformer.load_state_dict(torch.load("transformer_model_true.pt", map_location=device))
    transformer.to(device)
