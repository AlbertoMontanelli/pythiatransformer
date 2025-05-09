import fastjet as fj

from test_output_transformer import targets_pad_mask
from fastjet_preparation import output, target

N_EVENTS = output.shape[0]
N_PARTICLES = output.shape[1]

def clustering(data, padding_mask, n_events, n_particles):
    
    # Jet clustering algorithm
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clustered_sequences = []
    for i_event in range(n_events):
        pseudojets = []
        for i_particle in range(n_particles):
            if not padding_mask[i_event, i_particle]:
                pseudojet = fj.PseudoJet(
                    data[i_event, i_particle, 0],
                    data[i_event, i_particle, 1],
                    data[i_event, i_particle, 2],
                    data[i_event, i_particle, 3],
                )
                pseudojet.set_user_index(i_particle)
                pseudojets.append(pseudojet)
        # Cluster stable particles with FastJet
        clustered_sequence = fj.ClusterSequence(pseudojets, jet_def)
        clustered_sequences.append(clustered_sequence)
    return clustered_sequences

clustered_sequences_target = clustering(target, targets_pad_mask, N_EVENTS, N_PARTICLES)
clustered_sequences_target = clustering(output, targets_pad_mask, N_EVENTS, N_PARTICLES)
