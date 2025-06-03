"""Generate events using Pythia and save particles with status 23 and
final stable particles into a ROOT file with two separate TTrees.
"""

from pathlib import Path

import awkward as ak
import uproot
from loguru import logger
from pythia8 import Pythia


def setup_pythia(seed: int = 10) -> Pythia:
    """Configure and return a Pythia instance."""
    try:
        pythia = Pythia()
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
        pythia.readString("Beams:eCM = 13000.")
        pythia.readString("HardQCD:all = on")
        pythia.readString("PhaseSpace:pTHatMin = 100.")
        pythia.init()
        return pythia
    except Exception as e:
        logger.exception("Failed to initialize Pythia.")
        raise


def initialize_data(features: list, suffix: str) -> dict:
    """Initialize dictionary for each feature with an empty list."""
    return {f"{key}{suffix}": [] for key in features}


def append_empty_event(data: dict, features: list, suffix: str):
    """Append an empty list for a new event to each feature key."""
    for feature in features:
        data[f"{feature}{suffix}"].append([])


def record_particle(particle, features: list, data: dict, suffix: str):
    """Append particle data to the latest event list."""
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
    """Discard the last event by removing the most recent sublist for
    each feature, if the event did not contain valid particles.
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
    """Convert list of lists to Awkward Array."""
    try:
        return ak.Array(data_dict)
    except Exception as e:
        logger.exception("Failed to convert data to Awkward Array.")
        raise


def save_to_root(output_file: str, data_23: dict, data_final: dict):
    """Save particle data to ROOT file using uproot."""
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
    """Generate ttbar events using Pythia and save particle data to a
    ROOT file. Stores status==23 particles and final-state particles
    in two TTrees. Each event is represented by a list of particles
    for each feature. Variable-length arrays are used to preserve
    per-event multiplicity.
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
                if found_23 and particle.isFinal() and (particle.pT()>10):
                    found_final = True
                    counter_final += 1
                    record_particle(particle, features, data_final, "_final")

            """
            if found_final:
                logger.info(
                    f"Found {counter_23} 23-status particles and"
                    f" {counter_final} final particles for event"
                    f" {event+1}.\n"
                )
            """
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
    generate_events("prova.root", n_events=1000)
"""
    for i in range(10):
        output = f"events_{i:02d}.root"
        seed = 10 + i
        generate_events(output, n_events=100000, seed=seed)
"""