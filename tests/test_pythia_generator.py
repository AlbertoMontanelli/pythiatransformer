"""
Unit tests for ``pythiatransformer.pythia_generator`` module.

The tests verify:

- correct Pythia setup and basic error-free initialization;
- construction and mutation of the in-memory event dictionaries
  (``initialize_data``, ``append_empty_event``, ``record_particle``,
  ``cleanup_event``);
- conversion to Awkward arrays;
- the behavior of ``pythia_generator.generate_events`` using mocks for
  both Pythia configuration and ROOT writing.

All tests are written using Python ``unittest`` framework.
"""

import unittest
from unittest.mock import patch

import awkward as ak

from pythiatransformer.pythia_generator import (
    FEATURES,
    append_empty_event,
    cleanup_event,
    convert_to_awkward,
    generate_events,
    initialize_data,
    record_particle,
    setup_pythia,
)


class TestPythiaGenerator(unittest.TestCase):
    """Test case for ``pythiatransformer.pythia_generator`` module.

    The fixture prepares a small set of particle features and a dummy
    particle class that mimics the minimal Pythia8 surface used by the
    production code (``id()``, ``status()``, kinematics, and
    ``isFinal()``).
    """

    def setUp(self):
        """Create shared fixtures used by multiple tests.

        Initializes:

        - ``self.features``: a representative subset of particle
          fields;
        - toy dictionaries for ``_23`` and ``_final`` branches used to
          validate Awkward conversion and ROOT writing;
        - ``self.DummyParticle``: a light mock class replicating the
          methods accessed by ``pythiatransformer.pythia_generator``.
        """
        self.features = [
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
        self.toy_data_23 = {
            "id_23": [[2, 1], [2, 2, 4], [2]],
            "px_23": [[0.2, 1.0], [0.3, 0.4, 0.5], [2.1]],
        }
        self.toy_data_final = {
            "id_final": [[2, 1], [2, 2, 4], [2]],
            "px_final": [[0.2, 1.0], [0.3, 0.4, 0.5], [2.1]],
        }

        class DummyParticle:
            """Tiny stand-in for a Pythia8 particle."""

            def __init__(
                self, status, is_final, particle_id, *args, **kwargs
            ):
                """
                Class constructor.

                Parameters
                ----------
                status: int
                    Pythia status code.
                is_final: bool
                    Whether the particle is final state.
                particle_id: int
                    PDG ID (toy in these tests).
                """
                self._status = status
                self._isFinal = is_final
                self._id = particle_id

            def id(self):
                return self._id

            def status(self):
                return self._status

            def isFinal(self):
                return self._isFinal

            def px(self):
                return 0.1

            def py(self):
                return 0.2

            def pz(self):
                return 0.3

            def e(self):
                return 1.0

            def m(self):
                return 0.511

            def pT(self):
                return 0.22

            def theta(self):
                return 0.2

            def phi(self):
                return 0.3

            def y(self):
                return 0.1

            def eta(self):
                return 0.5

        # Save DummyParticle as attribute of test class
        self.DummyParticle = DummyParticle

    def test_setup_pythia(self):
        """
        Test ``pythia_generator.setup_pythia`` working.

        Ensure that the function returns an initialized object.

        The test only verifies that a non-None instance is returned
        without raising, not the full generator configuration.
        """
        seed = 1
        eCM = 1000
        pTHatMin = 1
        try:
            pythia = setup_pythia(seed, eCM, pTHatMin)
            self.assertIsNot(pythia, None, "Pythia setup returned None.")
        except Exception as e:
            self.fail(f"Pythia setup failed: {e}")

    def test_initialize_data(self):
        """Build the event dictionary skeleton with empty lists."""
        data = initialize_data(self.features, "")
        for key in self.features:
            self.assertIn(key, data, f"Key {key} is not in data")
            self.assertEqual(
                data[key], [], f"Key {key} does not map to an empty list"
            )

    def test_append_empty_event(self):
        """Append an empty sub-list for each feature."""
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        for key in self.features:
            self.assertTrue(
                isinstance(data[key][-1], list),
                f"The last element for key {key} is not a list",
            )
            self.assertEqual(
                len(data[key][-1]),
                0,
                f"The last element for key {key} is not an empty list",
            )

    def test_record_particle(self):
        """
        Write a particle feature into the current event slot.

        Verify that for every requested feature the value recorded in
        the nested list matches the one returned by the dummy particle
        method of the same name.
        """
        particle = self.DummyParticle(
            status=23, is_final=False, particle_id=1
        )
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        record_particle(particle, self.features, data, "")
        for feature in self.features:
            self.assertEqual(
                data[feature][-1][-1],
                getattr(particle, feature)(),
                f"The element {data[feature][-1][-1]}"
                f"does not match with {getattr(particle, feature)()}",
            )

    def test_cleanup_event(self):
        """Remove the latest (empty) event slot for all features."""
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        cleanup_event(data, self.features, "")
        for key in data.keys():
            self.assertEqual(len(data[key]), 0, "Cleaning unsuccessful")

    def test_convert_to_awkward(self):
        """
        Convert nested python lists to ``ak.Array`` class.

        Assert that:

        - awkward fields mirror the input dict keys and order;
        - values are preserved event-by-event;
        - sublists become Awkward arrays with consistent element types.
        """
        awkward_data = convert_to_awkward(self.toy_data_23)

        # Verify that the fields in the Awkward Array match the
        # dictionary keys.
        for ak_key, key in zip(awkward_data.fields, self.toy_data_23.keys()):
            self.assertEqual(
                ak_key,
                key,
                f"Field '{ak_key}' does not match the dictionary key '{key}'",
            )

        # Validate each key's data.
        for key in self.toy_data_23.keys():
            self.assertIsInstance(
                awkward_data[key], ak.Array, f"{key} is not an Awkward Array."
            )
            self.assertEqual(
                ak.to_list(awkward_data[key]),
                self.toy_data_23[key],
                f"Values for key {key} do not match the original input.",
            )
            for ak_sublist, sublist in zip(
                awkward_data[key], self.toy_data_23[key]
            ):
                self.assertTrue(
                    isinstance(ak_sublist, ak.Array),
                    f"The sublist {ak_sublist} is not an Awkward Array.",
                )
                self.assertTrue(
                    all(isinstance(x, type(sublist[0])) for x in sublist),
                    f"Element type in sublist of {key} differs"
                    f" from the original input.",
                )

    @patch("pythiatransformer.pythia_generator.Path.mkdir")
    @patch("pythiatransformer.pythia_generator.setup_pythia")
    def test_generate_events(self, mock_setup_pythia, mock_Path):
        """
        Integration test for ``pythia_generator.generate_events``.

        This test patches the two external boundaries:

        - patch #1: ``setup_pythia`` returns a mock `Pythia` object
          whose ``next()`` method yields two events. Each yield sets
          ``mock_pythia.event`` to a simple list of dummy particles;
        - patch #2: ``RootChunkWriter`` replaced inside the test with a
          minimal fake that records the last Awkward chunks it receives
          via ``extend``. This avoids any disk I/O and lets assert on
          the exact arrays produced by the driver.

        It then feeds two toy events and verifies that:

        - status-23 particle IDs are collected in order of appearance
          once a 23 is seen;
        - final-state particle IDs are collected after the first 23 is
          observed in the same event.
        """
        # # Mock Path.mkdir to avoid the creation of the directory.
        mock_Path.return_value = None

        # Mock Pythia instance returned by setup_pythia().
        mock_pythia = mock_setup_pythia()

        # Configurate the toy particles for each event.
        event_1 = [
            self.DummyParticle(23, False, 1),  # first 23
            self.DummyParticle(23, True, -1),
            self.DummyParticle(-23, False, 11),
            self.DummyParticle(0, True, -11),
            self.DummyParticle(0, True, -12),
        ]
        event_2 = [
            self.DummyParticle(0, True, 55),
            self.DummyParticle(23, True, 2),  # first 23
            self.DummyParticle(23, False, -2),
            self.DummyParticle(-23, False, -3),
            self.DummyParticle(1, True, 22),
            self.DummyParticle(0, False, -22),
        ]

        # Mock the event as a list of particles and update dynamically.
        def next_side_effect():
            # Return True for each event, then False to stop
            for particles in [event_1, event_2]:
                mock_pythia.event = particles
                yield True
            while True:
                yield False  # Stop after two events

        # Simulate each different event for each call given to original
        # pythia.next() method.
        mock_pythia.next.side_effect = next_side_effect()

        # Define dictionaries for status 23 and final particles.
        captured = {"ak23": None, "akF": None}

        class FakeWriter:
            """
            Minimal drop-in for ``RootChunkWriter`` used in tests.

            Working:

            - call ``extend(ak23, akF)``;
            - store those arrays into ``captured``;
            - don't save any ROOT file as the original function.
            """

            def __init__(
                self,
                output_file=None,
                _initialized=False,
            ):
                self._initialized = _initialized
                self.output_file = output_file

            def extend(self, ak23, akF):
                captured["ak23"] = ak23
                captured["akF"] = akF
                self._initialized = True

        # Patch RootChunkWriter to replace it with FakeWriter during
        # this block only.
        with patch(
            "pythiatransformer.pythia_generator.RootChunkWriter", FakeWriter
        ):
            # 2 events only (events_per_seed=2).
            generate_events(
                seed=42,
                events=2,
                chunk_size=2,
                features=FEATURES,
            )

        # Assert that captured data matches the expected data.
        ak23 = captured["ak23"]
        akF = captured["akF"]
        if ak23 is None or akF is None:
            self.fail("Writer didn't capture arrays")
        else:
            ids_23 = ak.to_list(ak23["id_23"])
            ids_F = ak.to_list(akF["id_final"])
        self.assertEqual(ids_23, [[1, -1, 11], [2, -2, -3]])
        self.assertEqual(ids_F, [[-1, -11, -12], [2, 22]])


if __name__ == "__main__":
    unittest.main()
