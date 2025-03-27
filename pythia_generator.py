"""
pythia generator
"""
from pythia8 import Pythia
from ROOT import TFile, TTree, std

N_EVENTS = 3000

# Pythia configuration.
pythia = Pythia()
pythia.readString("Beams:eCM = 13000.0")
pythia.readString("Top:qqbar2ttbar = on")
pythia.init()

features_list = ["id", "status", "px", "py", "pz", "e", "m"]
attributes_23 = {
    "_".join([feature, "23"]): std.vector('int')()
    if feature in ["id", "status"] else std.vector('float')()
    for feature in features_list
    }
attributes_final = {
    "_".join([feature, "final"]): std.vector('int')()
    if feature in ["id", "status"] else std.vector('float')()
    for feature in features_list
    }

root_file = TFile("events.root", "RECREATE")
tree = TTree("ParticleTree", "Tree of particles of status 23 and final state")
for key, vector in {**attributes_23, **attributes_final}.items():
    tree.Branch(key, vector)

# Accessor per le particelle status 23
# lambda è una funziona placeholder che accetta come argomento p e
# restituisce p.id() prima che abbia definito la funzione
# particle = pythia.event[i]
accessors_23 = {
    "_".join([feature, "23"]): lambda p, f = feature: getattr(p, f)()
    for feature in features_list
}
accessors_final = {
    "_".join([feature, "final"]): lambda p, f = feature: getattr(p, f)()
    for feature in features_list
}

for i_event in range(N_EVENTS):
    if not pythia.next():
        continue
    for vector in {**attributes_23, **attributes_final}.values():
        vector.clear()
    # Filtra particelle con status 23
    for i in range(pythia.event.size()):
        particle = pythia.event[i]
        if particle.status() == 23:
            for key, value in accessors_23.items():
                attributes_23[key].push_back(value(particle))
        if particle.isFinal():
            for key, value in accessors_final.items():
                attributes_final[key].push_back(value(particle))
    # Riempie il TTree. Ricorda che nel Tree ogni riga corrisponde ad
    # un evento di pythia. Per ogni riga avrò array colonna dinamici
    # contenenti le caratteristiche dell'evento:
    # nr particelle con status 23, nr particelle finali,
    # p di ciascuna particella con status 23, etc
    tree.Fill()

root_file.Write()
root_file.Close()
