"""Generate events using Pythia and save particles with status 23 and
final stable particles into a ROOT file with two separate TTrees.
"""
from pathlib import Path

from loguru import logger
import numpy as np
import uproot

from pythia8 import Pythia
 

def setup_pythia() -> Pythia:
    """Configure and return a Pythia instance."""
    try:
        pythia = Pythia()
        pythia.readString("Beams:eCM = 13000.")
        pythia.readString("Top:qqbar2ttbar = on")
        pythia.init()
        return pythia
    except Exception as e:
        logger.exception("Failed to initialize Pythia.")
        raise

def initialize_data(features):
    """Initialize dictionary for each feature with an empty list."""
    return {f"{key}": [] for key in features}

def append_empty_event(data, features):
    """Append an empty list for a new event to each feature key."""
    for feature in features:
        data[f"{feature}"].append([])

def record_particle(particle, features, data, suffix):
    """Append particle data to the latest event list."""
    for feature in features:
        try:
            value = getattr(particle, feature)()
            data[f"{feature}{suffix}"][-1].append(value)
        except Exception as e:
            logger.warning(f"Failed to record feature '{feature}' for a particle: {e}")
            continue

def cleanup_event(data_23, data_final, features):
    """Discard the last event by removing the most recent sublist for
    each feature, if the event did not contain valid particles.
    """
    for feature in features:
        try:
            data_23[f"{feature}_23"].pop()
            data_final[f"{feature}_final"].pop()
        except IndexError:
            logger.warning(f"No event data found for feature '{feature}' — the mother list is empty")

def convert_to_numpy(data_dict):
    """Convert list of lists to numpy array with dtype=object."""
    try:
        return {
            key: np.array(value, dtype=object)
            for key, value in data_dict.items()
        }
    except Exception as e:
        logger.exception("Failed to convert data to numpy arrays.")
        raise

def save_to_root(output_file, data_to_save_23, data_to_save_final):
    """Save particle data to ROOT file using uproot."""
    try:
        with uproot.recreate(output_file) as root_file:
            root_file["tree_23"] = data_to_save_23
            root_file["tree_final"] = data_to_save_final
    except Exception as e:
        logger.exception("Failed to save data to ROOT file.")
        raise

def generate_events(output_file: str, n_events: int):
    """Generate ttbar events using Pythia and save particle data to a
    ROOT file. Stores status==23 particles and final-state particles
    in two TTrees. Each event is represented by a list of particles
    for each feature. Variable-length arrays are used to preserve
    per-event multiplicity.
    """
    output_file = Path(output_file)
    features = ["id", "status", "px", "py", "pz", "e", "m"]
    pythia = setup_pythia()

    data_23 = initialize_data([f"{key}_23" for key in features])
    data_final = initialize_data([f"{key}_final" for key in features])

    for i in range(n_events):
        try:
            if not pythia.next():
                logger.warning(f"Event {i} failed to generate.")
                continue

            found_23 = False
            found_final = False
            counter_23 = 0
            counter_final = 0

            append_empty_event(data_23, [f"{key}_23" for key in features])
            append_empty_event(data_final, [f"{key}_23" for key in features])

            for particle in pythia.event:
                if particle.status() == 23:
                    found_23 = True
                    counter_23 += 1
                    record_particle(particle, features, data_23, "_23")
                if found_23 and particle.isFinal():
                    found_final = True
                    counter_final += 1
                    record_particle(particle, features, data_final, "_final")

            if found_final:
                logger.info(f"Found {counter_23} 23-status particles and {counter_final} final particles for event {i}")
            else:
                cleanup_event(data_23, data_final, features)
                logger.info(f"Event {i} discarded: no status-23 or final particles found.")

        except Exception as e:
            logger.exception(f"Unexpected error during event {i}: {e}")

    
    save_to_root(
        output_file,
        convert_to_numpy(data_23),
        convert_to_numpy(data_final),
    )

if __name__ == "__main__":
    generate_events("events.root", n_events=100)
    

"""
    # Debug: print event sizes.
    for key, values in data_23.items():
        print(f"Key: {key}")
        for i, sublist in enumerate(values):
            print(f"  Event {i + 1}: Length = {len(sublist)}")

    for key, values in data_final.items():
        print(f"Key: {key}")
        for i, sublist in enumerate(values):
            print(f"  Event {i + 1}: Length = {len(sublist)}")
"""

