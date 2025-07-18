"""
Generate events using Pythia and save particles with status 23 and
final stable particles into a ROOT file with two separate TTrees.
"""

from pathlib import Path

import awkward as ak
import uproot
from loguru import logger
from pythia8 import Pythia


def setup_pythia(seed = 10, e_cm = 13000., pt_hat_min = 100.):
    """
    Configure and return a Pythia instance for HardQCD process with
    initialized random seed, center of mass energy and minimum pTHat.

    Args:
        seed (int): initialization of the seed for reproducibility;
        e_cm (float): center of mass energy;
        pt_hat_min (float): minimum pTHat.
    
    Returns:
        pythia (Pythia): initialized instance of the Pythia generator.
    """
    if not isinstance(seed, int):
        raise TypeError(
            f"Parameter 'seed' must be of type 'int', "
            f"got '{type(seed)}' instead."
        )
    if not isinstance(e_cm, (int, float)):
        raise TypeError(
            f"Parameter 'e_cm' must be a positive number (int/float), "
            f"got '{type(e_cm)}' instead."
        )
    if e_cm <= 0:
        raise ValueError(
            f"Parameter 'e_cm' must be positive, "
            f"got {e_cm} instead."
        )
    if not isinstance(pt_hat_min, (int, float)) or pt_hat_min <= 0:
        raise TypeError(
            f"Parameter 'pTHatMin' must be a positive number (int/float), "
            f"got '{type(pt_hat_min)}' instead."
        )
    if pt_hat_min < 0:
        raise ValueError(
            f"Parameter 'pt_hat_min' must be non negative, "
            f"got {pt_hat_min} instead."
        )
    try:
        pythia = Pythia()
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
        pythia.readString(f"Beams:e_cm = {e_cm}")
        pythia.readString("HardQCD:all = on")
        pythia.readString(f"PhaseSpace:pt_hat_min = {pt_hat_min}.")
        pythia.init()
        return pythia
    except Exception as e:
        logger.exception("Failed to initialize Pythia.")
        raise


def initialize_data(features, suffix):
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
    if not isinstance(features, list):
        raise TypeError(
            f"Parameter 'features' must be of type 'list', "
            f"got '{type(features)}' instead."
        )
    if not isinstance(suffix, str):
        raise TypeError(
            f"Parameter 'suffix' must be of type 'str', "
            f"got '{type(suffix)}' instead."
        )
    if not all(isinstance(f, str) for f in features):
        raise TypeError(
            "Parameter 'features' must be a list of strings."
        )
    return {f"{key}{suffix}": [] for key in features}


def append_empty_event(data, features, suffix):
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


def record_particle(particle, features, data, suffix):
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


def cleanup_event(data, features, suffix):
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


def convert_to_awkward(data_dict):
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


def save_to_root(output_file, data_23, data_final):
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


def generate_events(output_file, n_events):
    """
    Generate events using Pythia and save particle data to a ROOT file.
    Store status==23 particles and final state particles in two TTrees.
    Each event is represented by a list of particles for each feature.
    Variable length arrays are used to preserve per-event multiplicity.

    Args:
        output_file (str): path to the output file;
        n_events (int): number of events;

    Returns:
        None
    """
    if not isinstance(n_events, int):
        raise TypeError(
            f"Parameter 'n_events' must be of type 'int', "
            f"got '{type(n_events)}' instead."
        )
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
        generate_events(output, n_events=100000)
