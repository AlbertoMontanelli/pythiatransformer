from pathlib import Path

import awkward as ak
import unittest
from unittest.mock import patch
import uproot
from loguru import logger

from pythiatransformer.pythia_generator import (
    setup_pythia,
    initialize_data,
    append_empty_event,
    record_particle,
    cleanup_event,
    convert_to_awkward,
    save_to_root,
    generate_events,
)


class PythiaTransformerTest(unittest.TestCase):

    def setUp(self):

        self.features = ["id", "status", "px", "py", "pz", "e", "m", "pT", "theta", "phi", "y"]
        self.toy_data_23 = {
            "id_23": [[2, 1], [2, 2, 4], [2]],
            "px_23": [[0.2, 1.0], [0.3, 0.4, 0.5], [2.1]],
        }
        self.toy_data_final = {
            "id_final": [[2, 1], [2, 2, 4], [2]],
            "px_final": [[0.2, 1.0], [0.3, 0.4, 0.5], [2.1]],
        }

        class DummyParticle(unittest.mock.Mock):
            """
            Toy class to define a particle simulating particle class
            of Pythia.
            """

            def __init__(self, status, is_final, particle_id, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._id = particle_id
                self._status = status
                self._isFinal = is_final

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
        # Save DummyParticle as attribute of test class
        self.DummyParticle = DummyParticle

    def test_setup_pythia(self):
        seed=1
        eCM=1000
        pTHatMin=1
        """
        Test if Pythia is set up correctly.
        """
        try:
            pythia = setup_pythia(seed, eCM, pTHatMin)
            self.assertIsNot(pythia, None, "Pythia setup returned None.")
        except Exception as e:
            self.fail(f"Pythia setup failed: {e}")

    def test_initialize_data(self):
        """
        Test initialization of data dictionary.
        """
        data = initialize_data(self.features, "")
        for key in self.features:
            self.assertIn(key, data, f"Key {key} is not in data")
            self.assertEqual(
                data[key], [], f"Key {key} does not map to an empty list"
            )

    def test_append_empty_event(self):
        """
        Test appending an empty sublist to the dictionary.
        """
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
        Test recording a particle.
        """
        particle = self.DummyParticle(status=23, is_final=False, particle_id=1)
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
        """
        Test cleaning up a sublist appended to the dictionary.
        """
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        cleanup_event(data, self.features, "")
        for key in data.keys():
            self.assertEqual(len(data[key]), 0, f"Cleaning unsuccessful")

    def test_convert_to_awkward(self):
        """
        Test the conversion of a dictionary of lists of lists
        to an Awkward Array.
        """
        awkward_data = convert_to_awkward(self.toy_data_23)

        # Verify that the fields in the Awkward Array match
        # the dictionary keys.
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
                    all(
                        type(x) == type(sublist[0])
                        or isinstance(x, type(sublist[0]))
                        for x in sublist
                    ),
                    f"Element type in sublist of {key} differs"
                    f" from the original input.",
                )

    def test_save_to_root(self):
        """
        Test saving to a ROOT file.
        """
        output_file = Path("tests/toy_testing_save.root")
        save_to_root(
            output_file,
            convert_to_awkward(self.toy_data_23),
            convert_to_awkward(self.toy_data_final),
        )
        self.assertTrue(
            output_file.exists(), f"File {output_file} does not exist."
        )
        with uproot.open(output_file) as root_file:
            self.assertIn(
                "tree_23", root_file, "tree_23 missing in ROOT file."
            )
            self.assertIn(
                "tree_final", root_file, "tree_final missing in ROOT file."
            )

    """
    @patch is a decorator given by unittest.mock library, which allows to 
    substitute (mock) specific objects or functions of the code that needs
    to be tested. This is useful in order to simulate behaviours or to isolate
    dependencies during testing.

    How does @patch work?
    When using @patch('module.object'), you are telling Python to substitute
    said object during testing.
    The mock is passed as a parameter of the testing function.
    """

    @patch("pythiatransformer.pythia_generator.setup_pythia")
    @patch("pythiatransformer.pythia_generator.save_to_root")
    def test_generate_events(self, mock_save_to_root, mock_setup_pythia):
        """
        Test to verify the recordings of multiple particles for
        event of generate_events function in pythia_generator (without
        saving data in a ROOT file) mocking the configuration of
        setup_pythia and save_to_root function.
        """
        # Mock the Pythia object.
        mock_pythia = mock_setup_pythia.return_value

        # Configurate the toy particles for each event.
        event_1 = [
            self.DummyParticle(23, False, 1),
            self.DummyParticle(23, True, -1),
            self.DummyParticle(-23, False, 11),
            self.DummyParticle(0, True, -11),
            self.DummyParticle(0, True, -12),
        ]
        event_2 = [
            self.DummyParticle(0, True, 55),
            self.DummyParticle(23, True, 2),
            self.DummyParticle(23, False, -2),
            self.DummyParticle(-23, False, -3),
            self.DummyParticle(1, True, 22),
            self.DummyParticle(0, False, -22),
        ]

        # Mock the event as a list of particles and update dynamically.
        def side_effect_for_event():
            for particles in [event_1, event_2]:
                mock_pythia.event = particles
                yield True

        # Simulate each different event for each call.
        mock_pythia.next.side_effect = side_effect_for_event()

        # Define expected results for data_23 and data_final.
        expected_data_23 = {"id_23": [[1, -1, 11], [2, -2, -3]]}
        expected_data_final = {"id_final": [[-1, -11, -12], [2, 22]]}

        # Define dictionaries for status 23 and final particles.
        data_23 = {}
        data_final = {}

        # Mock function to simulate saving data to a ROOT file.
        def mock_save(file_name, data_23_input, data_final_input):
            """
            Mocking function to substituite save_to_root function.
            Fill non locally the dictionaries defined before with the
            values of the toy particles that will be generated for each
            event. Don't store any data in a ROOT file as the original
            save_to_root function did.
            """
            nonlocal data_23, data_final
            # Convert Awkward Arrays to Python lists before storing
            data_23 = {
                key: ak.to_list(data_23_input[key])
                for key in data_23_input.fields
            }
            data_final = {
                key: ak.to_list(data_final_input[key])
                for key in data_final_input.fields
            }

        mock_save_to_root.side_effect = mock_save

        # Run the function to generate events. It won't saved any file
        # because the original save_to_root is mocked.
        generate_events("dummy_output.root", 2)

        # Assert that captured data matches the expected data.
        self.assertEqual(data_23["id_23"], expected_data_23["id_23"])
        self.assertEqual(
            data_final["id_final"], expected_data_final["id_final"]
        )


if __name__ == "__main__":
    unittest.main()
