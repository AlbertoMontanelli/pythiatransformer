from pythia8 import Pythia
from ROOT import TFile, TTree, std

# Configura Pythia8
pythia = Pythia()

# Imposta l'energia del centro di massa (13 TeV, LHC)
pythia.readString("Beams:eCM = 13000.0")
pythia.readString("Top:qqbar2ttbar = on")

# Inizializza Pythia
pythia.init()

# Configura il file ROOT e il TTree
root_file = TFile("events.root", "RECREATE")
tree = TTree("ParticleTree", "Tree of particles with status 23 and final state")

# Dizionario per le variabili delle particelle status 23
attributes_status23 = {
    "PID23": std.vector('int')(),
    "Status23": std.vector('int')(),
    "Px23": std.vector('float')(),
    "Py23": std.vector('float')(),
    "Pz23": std.vector('float')(),
    "E23": std.vector('float')(),
    "M23": std.vector('float')(),
    "Mother1_23": std.vector('int')(),
    "Mother2_23": std.vector('int')(),
    "Daughter1_23": std.vector('int')(),
    "Daughter2_23": std.vector('int')(),
}

# Dizionario per le variabili delle particelle finali
attributes_final = {
    "PID_final": std.vector('int')(),
    "Status_final": std.vector('int')(),
    "Px_final": std.vector('float')(),
    "Py_final": std.vector('float')(),
    "Pz_final": std.vector('float')(),
    "E_final": std.vector('float')(),
    "M_final": std.vector('float')(),
}

# Collega i dizionari al TTree
for key, vector in {**attributes_status23, **attributes_final}.items(): # ** serve a unire i due dizionari
    tree.Branch(key, vector)

# Accessor per le particelle status 23
# lambda è una funziona placeholder che accetta come argomento p e restituisce p.id() prima che abbia definito la funzione particle = pythia.event[i]
accessors_status23 = {
    "PID23": lambda p: p.id(),
    "Status23": lambda p: p.status(),
    "Px23": lambda p: p.px(),
    "Py23": lambda p: p.py(),
    "Pz23": lambda p: p.pz(),
    "E23": lambda p: p.e(),
    "M23": lambda p: p.m(),
    "Mother1_23": lambda p: p.mother1(),
    "Mother2_23": lambda p: p.mother2(),
    "Daughter1_23": lambda p: p.daughter1(),
    "Daughter2_23": lambda p: p.daughter2(),
}

# Accessor per le particelle finali
accessors_final = {
    "PID_final": lambda p: p.id(),
    "Status_final": lambda p: p.status(),
    "Px_final": lambda p: p.px(),
    "Py_final": lambda p: p.py(),
    "Pz_final": lambda p: p.pz(),
    "E_final": lambda p: p.e(),
    "M_final": lambda p: p.m(),
}


# Genera eventi
n_events = 1000
for i_event in range(n_events):
    if not pythia.next():
        continue

    # Resetta i vettori
    for vector in {**attributes_status23, **attributes_final}.values():
        vector.clear()

    # Filtra particelle con status 23
    for i in range(pythia.event.size()):
        particle = pythia.event[i]

        if particle.status() == 23:
            for key, accessor in accessors_status23.items():
                attributes_status23[key].push_back(accessor(particle))

        if particle.isFinal():
            for key, accessor in accessors_final.items():
                attributes_final[key].push_back(accessor(particle))

    # Riempie il TTree. Ricorda che nel Tree ogni riga corrisponde ad un evento di pythia. Per ogni riga avrò array colonna dinamici contenenti 
    # le caratteristiche dell'evento: nr particelle con status 23, nr particelle finali, p di ciascuna particella con status 23, etc
    tree.Fill()

# Salva il file ROOT
root_file.Write()
root_file.Close()


