import pytest
from pythiatransformer.pythia_generator import (
    setup_pythia,
    initialize_data,
    append_empty_event,
    record_particle,
    cleanup_event,
    convert_to_numpy,
    save_to_root,
    generate_events,
)
from pathlib import Path
import numpy as np
import uproot


def test_setup_pythia():
    """Test if Pythia is set up correctly."""
    try:
        pythia = setup_pythia()
        assert pythia is not None, "Pythia setup returned None"
    except Exception as e:
        pytest.fail(f"Pythia setup failed: {e}")


def test_initialize_data():
    """Test initialization of data dictionary."""
    features = ["id", "px", "py"]
    data = initialize_data(features)
    assert all(key in data for key in features)
    assert all(data[key] == [] for key in features)


def test_append_empty_event():
    """Test appending an empty event."""
    features = ["id", "px", "py"]
    data = initialize_data(features)
    append_empty_event(data, features)
    assert all(isinstance(data[key][-1], list) for key in features)


def test_record_particle():
    """Test recording a particle."""
    class DummyParticle:
        def id(self): return 11
        def px(self): return 0.1
        def py(self): return 0.2

    features = ["id", "px", "py"]
    data = initialize_data(features)
    append_empty_event(data, features)
    particle = DummyParticle()
    record_particle(particle, features, data, "")
    for feature in features:
        assert data[feature][-1][-1] == getattr(particle, feature)()


def test_cleanup_event():
    """Test cleaning up an invalid event."""
    features = ["id", "px", "py"]
    data_23 = initialize_data([f"{key}_23" for key in features])
    data_final = initialize_data([f"{key}_final" for key in features])
    append_empty_event(data_23, [f"{key}_23" for key in features])
    append_empty_event(data_final, [f"{key}_final" for key in features])
    cleanup_event(data_23, data_final, features)
    for key in data_23.keys():
        assert len(data_23[key]) == 0
    for key in data_final.keys():
        assert len(data_final[key]) == 0


def test_convert_to_numpy():
    """Test conversion to numpy arrays."""
    data = {"id": [[1, 2]], "px": [[0.1, 0.2]]}
    numpy_data = convert_to_numpy(data)
    for key in data.keys():
        assert isinstance(numpy_data[key], np.ndarray)


def test_save_to_root(tmp_path):
    """Test saving to a ROOT file."""
    output_file = tmp_path / "test.root"
    data_to_save_23 = {"px_23": [[1.0, 2.0]], "py_23": [[0.5, -0.5]]}
    data_to_save_final = {"px_final": [[1.0]], "py_final": [[0.5]]}

    try:
        save_to_root(output_file, data_to_save_23, data_to_save_final)
    except Exception as e:
        pytest.fail(f"Failed to save to ROOT: {e}")

    assert output_file.exists()

    with uproot.open(output_file) as root_file:
        assert "tree_23" in root_file, "tree_23 missing in ROOT file."
        assert "tree_final" in root_file, "tree_final missing in ROOT file."


@pytest.mark.parametrize("n_events", [10, 20])
def test_generate_events(tmp_path, n_events):
    """Test full event generation."""
    output_file = tmp_path / "events.root"
    generate_events(output_file, n_events)

    assert output_file.exists()

    with uproot.open(output_file) as root_file:
        assert "tree_23" in root_file, "tree_23 missing in ROOT file."
        assert "tree_final" in root_file, "tree_final missing in ROOT file."
