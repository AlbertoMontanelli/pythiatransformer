import fastjet as fj
from pythia8 import Pythia

# Jet clustering algorithm
jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)

pythia = Pythia()
pythia.readString("Random:setSeed = on")
pythia.readString("Random:seed = 10")
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 100.")
pythia.init()

N_EVENTS = 10000

for i_event in range(N_EVENTS):
    if not pythia.next():
        continue

    # Create pseudojet inputs from stable particles
    particles = pythia.event
    #    pseudojets = [fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in particles if is_stable(p)]
    pseudojets = []
    for i, p in enumerate(particles):
        if p.isFinal():
            pseudojet = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pseudojet.set_user_index(i)
            pseudojets.append(pseudojet)

    # Cluster stable particles with FastJet
    clustered_sequence = fj.ClusterSequence(pseudojets, jet_def)