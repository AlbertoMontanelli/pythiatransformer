"""
Generate events using Pythia and save particles with status 23 and
final stable particles into a ROOT file with two separate TTrees.
"""

from pathlib import Path

import awkward as ak
import uproot
from loguru import logger
from pythia8 import Pythia


def setup_pythia(seed: int = 10, eCM: float = 13000., pTHatMin: float = 100.):
    """
    Configure and return a Pythia instance for HardQCD process with
    initialized random seed, center of mass energy and minimum pTHat.

    Args:
        seed (int): initialization of the seed for reproducibility;
        eCM (float): center of mass energy;
        pTHatMin (float): minimum pTHat.
    
    Returns:
        pythia (Pythia): initialized instance of the Pythia generator.
    """
    try:
        pythia = Pythia()
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
        pythia.readString(f"Beams:eCM = {eCM}")
        pythia.readString("HardQCD:all = on")
        pythia.readString(f"PhaseSpace:pTHatMin = {pTHatMin}.")
        pythia.init()
        return pythia
    except Exception as e:
        logger.exception("Failed to initialize Pythia.")
        raise


def initialize_data(features: list, suffix: str):
    """
    Initialize dictionary for each feature with an empty list.

    Args:
        features (list): list of strings of relevant features
                         (e.g. 'px', 'id', etc);
        suffix (str): suffix of the specific set of events
                      (e.g. '_23', '_final').

    Returns:
        dict: ordered dictionary linking features and set of events
              via the specific suffix.
    """
    if not all(isinstance(f, str) for f in features):
        raise TypeError(
            f"Parameter 'features' must be a list of strings."
        )
    return {f"{key}{suffix}": [] for key in features}


def append_empty_event(data: dict, features: list, suffix: str):
    """
    Append an empty list for a new event to each feature key.

    Args:
        data (dict): dictionary containing features per event.
        features (list): list of strings of relevant features
                         (e.g. 'px', 'id', etc);
        suffix (str): suffix of the specific set of events
                      (e.g. '_23', '_final').
    
    Returns:
        None
    """
    for feature in features:
        data[f"{feature}{suffix}"].append([])


def record_particle(particle, features: list, data: dict, suffix: str):
    """
    Append particle features to the latest event list.
    
    Args:
        particle: a Pythia8 particle object;
        features (list): list of features to record;
        data (dict): dictionary storing the event data;
        suffix (str): suffix of the specific set of events
                      (e.g. '_23', '_final').

    Returns:
        None
    """
    for feature in features:
        try:
            value = getattr(particle, feature)()
            data[f"{feature}{suffix}"][-1].append(value)
        except Exception as e:
            logger.warning(
                f"Failed to record feature '{feature}{suffix}'"
                f" for a particle: {e}"
            )
            continue


def cleanup_event(data: dict, features: list, suffix: str):
    """
    Discard the last event by removing the most recent sublist for
    each feature, if the event did not contain valid particles according
    to selected criteria.

    Args:
        data (dict): dictionary of the event data;
        features (list): list of particle features;
        suffix (str): suffix of the specific set of events
                      (e.g. '_23', '_final').

    Returns:
        None
    """
    for feature in features:
        try:
            data[f"{feature}{suffix}"].pop()
        except IndexError:
            logger.warning(
                f"No event data found for feature '{feature}{suffix}'"
                f" — the mother list is empty"
            )


def convert_to_awkward(data_dict: dict):
    """
    Convert list of lists to Awkward Array.

    Args:
        data_dict (dict): dictionary of the event data.

    Returns:
        ak.Array: data in form of Awkward Array.
    """
    try:
        return ak.Array(data_dict)
    except Exception as e:
        logger.exception("Failed to convert data to Awkward Array.")
        raise


def save_to_root(output_file: str, data_23: dict, data_final: dict):
    """
    Save particle data to ROOT file using uproot. Each array is stored
    in a different TTree.

    Args:
        output_file (str): the output ROOt file;
        data_23 (dict): data corresponding to status 23 particles;
        data_final (dict): data corresponding to final particles.

    Returns:
        None
    """
    try:
        with uproot.recreate(output_file) as root_file:
            root_file["tree_23"] = {
                key: data_23[key] for key in data_23.fields
            }
            root_file["tree_final"] = {
                key: data_final[key] for key in data_final.fields
            }
    except Exception as e:
        logger.exception("Failed to save data to ROOT file.")
        raise


def generate_events(output_file: str, n_events: int, seed: int = 10):
    """
    Generate events using Pythia and save particle data to a ROOT file.
    Store status==23 particles and final state particles in two TTrees.
    Each event is represented by a list of particles for each feature.
    Variable length arrays are used to preserve per-event multiplicity.

    Args:
        output_file (str): path to the output file;
        n_events (int): number of events;
        seed (int): initialization of the seed for reproducibility.

    Returns:
        None
    """
    output_file = Path(output_file)
    features = [
        "id",
        "status",
        "px",
        "py",
        "pz",
        "e",
        "m",
        "pT",
        "theta",
        "phi",
        "y",
        "eta",
    ]

    pythia = setup_pythia()

    data_23 = initialize_data(features, "_23")
    data_final = initialize_data(features, "_final")

    event = 0
    while event < n_events:
        try:
            if not pythia.next():
                logger.warning(f"Event {event} failed to generate.")
                continue

            found_23 = False
            found_final = False
            counter_23 = 0
            counter_final = 0

            append_empty_event(data_23, features, "_23")
            append_empty_event(data_final, features, "_final")

            for particle in pythia.event:
                if abs(particle.status()) == 23:
                    found_23 = True
                    counter_23 += 1
                    record_particle(particle, features, data_23, "_23")
                if found_23 and particle.isFinal():
                    found_final = True
                    counter_final += 1
                    record_particle(particle, features, data_final, "_final")

            if found_final:
                event += 1
            else:
                logger.info(
                    f"Event {event} discarded: no status-23 or final"
                    f" particles found."
                )
                cleanup_event(data_23, features, "_23")
                cleanup_event(data_final, features, "_final")

        except Exception as e:
            logger.exception(f"Unexpected error during event {event+1}: {e}")

    save_to_root(
        output_file,
        convert_to_awkward(data_23),
        convert_to_awkward(data_final),
    )


if __name__ == "__main__":
    for i in range(10):
        output = f"events_{i:02d}.root"
        seed = 10 + i
        generate_events(output, n_events=100000, seed=seed)
