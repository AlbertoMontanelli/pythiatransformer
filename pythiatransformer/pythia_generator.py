"""
Generate events using Pythia8.

Save particles with status 23 and final stable particles into a ROOT
file with two separate TTrees.
"""

import argparse
import gc
from pathlib import Path

import awkward as ak
import uproot
from loguru import logger
from pythia8 import Pythia


def _dir_path_finder(data):
    """Create and return the directory path for saving files."""
    base_dir = Path(__file__).resolve().parent
    if data:
        dir = base_dir / "data"
    else:
        dir = base_dir / "plots"
    dir.mkdir(exist_ok=True)
    return dir


FEATURES = [
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


def setup_pythia(seed=10, eCM=13000.0, pTHatMin=100.0):
    """
    Configure and return a Pythia instance for ``HardQCD`` process.

    Initialize the random seed, center of mass energy and minimum
    ``pTHat``.

    Parameters
    ----------
    seed: int
        initialization of the seed for reproducibility;
    eCM: float
        center of mass energy in GeV;
    pTHatMin: float
        minimum ``pTHat`` in GeV.

    Returns
    -------
    pythia: Pythia
        initialized instance of the Pythia generator.
    """
    if not isinstance(seed, int):
        raise TypeError(
            f"Parameter 'seed' must be of type 'int', "
            f"got '{type(seed)}' instead."
        )
    if not isinstance(eCM, (int, float)):
        raise TypeError(
            f"Parameter 'eCM' must be a positive number (int/float), "
            f"got '{type(eCM)}' instead."
        )
    if eCM <= 0:
        raise ValueError(
            f"Parameter 'eCM' must be positive, got {eCM} instead."
        )
    if not isinstance(pTHatMin, (int, float)) or pTHatMin <= 0:
        raise TypeError(
            f"Parameter 'pTHatMin' must be a positive number (int/float), "
            f"got '{type(pTHatMin)}' instead."
        )
    if pTHatMin < 0:
        raise ValueError(
            f"Parameter 'pTHatMin' must be non negative, "
            f"got {pTHatMin} instead."
        )
    try:
        pythia = Pythia()
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
        pythia.readString(f"Beams:eCM = {eCM}")
        pythia.readString("HardQCD:all = on")
        pythia.readString(f"PhaseSpace:pTHatMin = {pTHatMin}.")
        pythia.init()
        return pythia
    except Exception:
        logger.exception("Failed to initialize Pythia.")
        raise


def initialize_data(features, suffix):
    """
    Initialize dictionary for each feature with an empty list.

    Parameters
    ----------
    features: list[str]
        list of strings of relevant features (e.g. ``px``, ``id``,
        etc).
    suffix: str
        suffix of the specific set of events (e.g. ``_23``,
        ``_final``).

    Returns
    -------
    data_dict: dict
        ordered dictionary linking features and set of events via the
        the specific suffix.
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
        raise TypeError("Parameter 'features' must be a list of strings.")
    return {f"{key}{suffix}": [] for key in features}


def append_empty_event(data, features, suffix):
    """
    Append an empty list for a new event to each feature key.

    Parameters
    ----------
    data: dict
        dictionary containing features per event.
    features: list[str]
        list of strings of relevant features (e.g. ``px``, ``id``,
        etc);
    suffix: str
        suffix of the specific set of events (e.g. ``_23``,
        ``_final``).
    """
    for feature in features:
        data[f"{feature}{suffix}"].append([])


def record_particle(particle, features, data, suffix):
    """
    Append particle features to the latest event list.

    Parameters
    ----------
    particle: pythia8.Particle
        A single particle object from the Pythia8 event record.
    features: list[str]
        list of features to record;
    data: dict
        dictionary storing the event data;
    suffix: str
        suffix of the specific set of events (e.g. ``_23``,
        ``_final``).
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
    Remove the most recent sublist for each feature.

    Discard the last event if it did not contain valid particles
    according to selected criteria.

    Parameters
    ----------
    data: dict
        dictionary of the event data;
    features: list[str]
        list of particle features;
    suffix: str
        suffix of the specific set of events (e.g. ``_23``,
        ``_final``).
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

    The Awkward Array is structured as follows:

    - each key in ``data_dict`` becomes a ``field`` (branch) in the
      Awkward Array;
    - the outer dimension corresponds to ``events``: each element
      represents one event;
    - each event is stored as a ``record`` (i.e., a dictionary-like
      object) containing variable-length lists of particles for
      each field.

    The data is internally stored in a columnar format, but when
    printed it appears as a list of dictionaries:
    ::

        [
        {id_final: [...], px_final: [...], ...},   # event 0
        {id_final: [...], px_final: [...], ...},   # event 1
        ...
        ]

    Parameters
    ----------
    data_dict: dict
        dictionary of the event data.

    Returns
    -------
    data_akArray: ak.Array
        data in form of Awkward Array.
    """
    try:
        return ak.Array(data_dict)
    except Exception:
        logger.exception("Failed to convert data to Awkward Array.")
        raise


class RootChunkWriter:
    """
    Minimal abstraction over ``uproot`` to create and extend trees.

    The first call to ``RootChunckWriter.extend`` creates ``tree_23``
    and ``tree_final``.
    Subsequent calls append rows (events) to existing branches.
    """

    def __init__(
        self,
        output_file,
        _initialized=False,
    ):
        """

        Class constructor.

        Parameters
        ----------
        output_file : pathlib.Path
            Target ROOT file path.
        """
        self.output_file = output_file
        self._initialized = _initialized
        self._file = None
        self._tree_23 = None
        self._tree_final = None

    def _create(self, data_23_ak, data_final_ak):
        self._file = uproot.recreate(self.output_file)
        # Create the two TTrees.
        self._file["tree_23"] = {k: data_23_ak[k] for k in data_23_ak.fields}
        self._file["tree_final"] = {
            k: data_final_ak[k] for k in data_final_ak.fields
        }
        # Mantain handles to writable TTrees.
        self._tree_23 = self._file["tree_23"]
        self._tree_final = self._file["tree_final"]
        self._initialized = True

    def extend(self, data_23_ak, data_final_ak):
        """
        Create-or-append a chunk of events.

        Parameters
        ----------
        data_23_ak : ak.Array
            Status-23 particle chunk.
        data_final_ak : ak.Array
            Stable final-state particle chunk.
        """
        if not self._initialized:
            self._create(data_23_ak, data_final_ak)
            return
        # Append directly to TTrees handles.
        self._tree_23.extend({k: data_23_ak[k] for k in data_23_ak.fields})
        self._tree_final.extend(
            {k: data_final_ak[k] for k in data_final_ak.fields}
        )


def generate_events(
    seed,
    events,
    chunk_size,
    features=FEATURES,
):
    """
    Generate and store simulated events using Pythia8.

    For a given random seed, a specified number of events are generated
    with Pythia8 for the ``HardQCD`` process. For each event, status 23
    particles and final stable particles are stored into two separate
    TTrees within a single ROOT file.

    The generation proceeds in chunks to limit memory usage: after each
    chunk, events are converted into Awkward Arrays and flushed to
    disk.

    Parameters
    ----------
    seed : int
        Random seed used to initialize the Pythia8 generator for
        reproducibility.
    events : int
        Total number of events to generate. Must be an exact multiple
        of ``chunk_size``.
    chunk_size : int
        Number of events to generate per chunk before writing to disk.
    features : list[str]
        List of particle attributes to be stored.
    """
    if not events % chunk_size == 0:
        raise ValueError(
            "Number of total events must be a multiple of chunk size"
        )
    TOT_CHUNKS = int(events / chunk_size)

    # The output filename is identified by the nr of events generated.
    data_dir = _dir_path_finder(data=True)
    file_path = data_dir / f"events_{events}.root"
    writer = RootChunkWriter(Path(file_path))

    total = 0
    PARTICLE_STATUS_23 = 23

    logger.info(
        f"seed: {seed}, chunk size: {chunk_size}, total events: {events}"
    )
    pythia = setup_pythia(seed=seed)

    data_23 = initialize_data(features, "_23")
    data_final = initialize_data(features, "_final")

    ev_counter = 0
    chunk_counter = 0

    while ev_counter < events:
        if not pythia.next():
            continue

        found_23 = False
        found_final = False

        append_empty_event(data_23, features, "_23")
        append_empty_event(data_final, features, "_final")

        for particle in pythia.event:
            if abs(particle.status()) == PARTICLE_STATUS_23:
                found_23 = True
                record_particle(particle, features, data_23, "_23")
            if found_23 and particle.isFinal():
                found_final = True
                record_particle(particle, features, data_final, "_final")

        if found_final:
            ev_counter += 1
            total += 1
        else:
            cleanup_event(data_23, features, "_23")
            cleanup_event(data_final, features, "_final")

        if ev_counter > 0 and (
            ev_counter % chunk_size == 0 or ev_counter == events
        ):
            ak_23 = convert_to_awkward(data_23)
            ak_final = convert_to_awkward(data_final)
            writer.extend(ak_23, ak_final)
            chunk_counter += 1

            del ak_23, ak_final
            gc.collect()

            data_23 = initialize_data(features, "_23")
            data_final = initialize_data(features, "_final")

            logger.info(
                f"Chunk {chunk_counter}/{TOT_CHUNKS} flushed."
                f" Progress: {ev_counter}/{events}"
            )

    logger.info(
        f"Completed. Wrote {total}/{events} events to"
        f" data/events_{events}.root."
    )


def main():
    """
    Call ``pythia_transformer.generate_events`` with parser arguments.

    CLI Parameters
    --------------
    seed : int, optional, default=42
        Random seed used to initialize the Pythia8 generator for
        reproducibility.
    events : int, required
        Total number of events to generate. Must be an exact multiple
        of ``chunk_size``.
    chunk_size : int, optional, default=10000
        Number of events to generate per chunk before writing to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=int,
        required=True,
        help="number of events to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="number of events to generate per chunk",
    )
    args = parser.parse_args()

    generate_events(args.seed, args.events, args.chunk_size)


if __name__ == "__main__":
    main()
