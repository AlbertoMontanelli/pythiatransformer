from pythia8 import Pythia
import numpy as np
import pandas as pd
import uproot

"""
Adesso questo file genera eventi e li salva i dizionari (data_23 e data_final) della giusta shape
Quello che si ottiene è un dizionario del tipo:

{
    "id_final": [[1, 2], [3], [5]],
    "px_final": [[0.1, 0.2], [0.3], [0.8]]
    ...etc..
    }

in cui ho una lista lunga quanto gli eventi (evento = volta in cui ho trovato sia particelle 23 e sia succcessivamente ma nello stesso 
evento particelle finali) per ogni chiave. ogni lista ha ulteriori liste lunghe quanto il numero di particelle di stato 23 o finali
presenti in quello specifico evento

Infine si salvano i dizionari in due tree diversi di un file root. i tree mantengono questa shape, infatti al loro interno vengono
generati due diversi branch per ciascuna chiave: uno ha tante entries quanto gli eventi (e infatti i diversi branch con la n davanti
per entrambi i tree hanno le stesse entries), l'altro tante entries quante la somma di tutte le particelle per tutti gli eventi

Si possono fare degli unit test per controllare che gli if vengono rispettati e la coerenza delle lunghezza dei dizionari etc
Per esempio comparare final_counter per ogni evento con la lunghezza delle liste i data_final (vedi riga 73)

"""


def generate_events(output_file, n_events):
    # Dizionario per le particelle con status 23
    features = ["id", "status", "px", "py", "pz", "e", "m"]

    # Inizializza Pythia
    pythia = Pythia()
    pythia.readString("Beams:eCM = 13000.")
    pythia.readString("Top:qqbar2ttbar = on")
    pythia.init()

    # Liste per salvare i dati (per ogni evento una lista di particelle)
    data_23 = {key + "_23": [] for key in features}
    data_final = {key + "_final": [] for key in features}

    for _ in range(n_events):
        if not pythia.next():
            continue

        final_counter = 0
        # Flag per indicare se ci sono particelle con stato 23
        found_23 = False
        event_23 = {key + "_23": [] for key in features}
        event_final = {key + "_final": [] for key in features}

        for particle in pythia.event:
            # Particelle di stato 23
            if particle.status() == 23:
                found_23 = True
                for feature in features:
                    event_23[feature + "_23"].append(getattr(particle, feature)())

            # Particelle finali (solo se almeno una particella con stato 23 è stata trovata)
            if found_23 and particle.isFinal():
                final_counter += 1
                for feature in features:
                    event_final[feature + "_final"].append(getattr(particle, feature)())
        
        print(final_counter)
        # Salva i dati dell'evento solo se ci sono particelle di stato 23 e finali
        if found_23 and any(event_final[feature + "_final"] for feature in features):
            for feature in features:
                data_23[feature + "_23"].append(event_23[feature + "_23"])
                data_final[feature + "_final"].append(event_final[feature + "_final"])

    # Stampa le lunghezze per verificare
    for key, values in data_23.items():
        print(f"Chiave: {key}")
        for i, lista in enumerate(values):
            print(f"  Evento {i + 1}: Lunghezza = {len(lista)}")
    
    for key, values in data_final.items():
        print(f"Chiave: {key}")
        for i, lista in enumerate(values):
            print(f"  Evento {i + 1}: Lunghezza = {len(lista)}")
    
    # Convertire le liste in array NumPy di oggetti per mantenere la struttura variabile
    data_to_save_final = {key: np.array(value, dtype = object) for key, value in data_final.items()}
    data_to_save_23 = {key: np.array(value, dtype = object) for key, value in data_23.items()}

    # Creare e salvare il TTree in un file ROOT
    with uproot.recreate(output_file) as root_file:
        # Salva i dati in due TTree separati per le particelle di stato 23 e finali
        root_file["tree_final"] = data_to_save_final
        root_file["tree_23"] = data_to_save_23

if __name__ == "__main__":
    generate_events("events.root", n_events=100)



# ACCESSO ALLE ENTRIES DI CIASCUN BRANCH DEL TREE

"""
qua si potrebbero fare degli unit test spacchettando il file appena generato e controllando che i valori salvati, lunghezza, siano uguali, etc
"""

# Iterare su tutte le entry di un branch
with uproot.open("events.root") as root_file:
    tree = root_file["tree_final"]
    
    # questo un esempio con pandas, da installare con pip awkward-pandas
    df = tree.arrays(library = "pd")

    print(df)

    # qua un esempio per accedere alle singole entries (quindi ciascun evento, in cui ciascun evento ha un numero diverso di particelle) del branch 
    for i, entry in enumerate(tree["id_final"].array(library = "np")):
        print(f"Entry {i} (id_final):", entry)

    for i, entry in enumerate(tree["px_final"].array(library = "np")):
        print(f"Entry {i} (px_final):", entry)
    





