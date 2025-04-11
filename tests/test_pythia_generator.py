from pathlib import Path

import awkward as ak
import unittest
import uproot

from pythiatransformer.pythia_generator import (
    setup_pythia,
    initialize_data,
    append_empty_event,
    record_particle,
    cleanup_event,
    convert_to_awkward,
    save_to_root
)


class PythiaTransformerTest(unittest.TestCase):

    def setUp(self):
        self.features = ["id", "status", "px", "py", "pz", "e", "m"]
        self.toy_data_23 = {
            "id_23": [[2, 1], [2, 2, 4], [2]],
            "px_23": [[0.2, 1.], [0.3, 0.4, 0.5], [2.1]],
        }
        self.toy_data_final = {
            "id_final": [[2, 1], [2, 2, 4], [2]],
            "px_final": [[0.2, 1.], [0.3, 0.4, 0.5], [2.1]],
        }

    def test_setup_pythia(self):
        """Test if Pythia is set up correctly."""
        try:
            pythia = setup_pythia()
            self.assertIsNot(pythia, None, "Pythia setup returned None") 
        except Exception as e:
            self.fail(f"Pythia setup failed: {e}")

    def test_initialize_data(self):
        """Test initialization of data dictionary."""
        data = initialize_data(self.features, "")
        for key in self.features:
            self.assertIn(key, data, f"Key {key} is not in data")
            self.assertEqual(
                data[key], [],
                f"Key {key} does not map to an empty list"
            )

    def test_append_empty_event(self):
        """Test appending an empty sublist to the dictionary."""
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        for key in self.features:
            self.assertTrue(
                isinstance(data[key][-1], list),
                f"The last element for key {key} is not a list"
            )
            self.assertEqual(
                len(data[key][-1]), 0,
                f"The last element for key {key} is not an empty list"
            )

    def test_record_particle(self):
        """Test recording a particle."""
        class DummyParticle:
            def id(self): return 11
            def status(self): return 23
            def px(self): return 0.1
            def py(self): return 0.2
            def pz(self): return 0.3
            def e(self): return 5
            def m(self): 0.511

        particle = DummyParticle()
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        record_particle(particle, self.features, data, "")
        for feature in self.features:
            self.assertEqual(
                data[feature][-1][-1], getattr(particle, feature)(),
                f"The element {data[feature][-1][-1]}"
                f"does not match with {getattr(particle, feature)()}"
            )

    def test_cleanup_event(self):
        """Test cleaning up a sublist appended to the dictionary."""
        data = initialize_data(self.features, "")
        append_empty_event(data, self.features, "")
        cleanup_event(data, self.features, "")
        for key in data.keys():
            self.assertEqual(len(data[key]), 0, f"Cleaning unsuccessful")

    def test_convert_to_awkward(self):
        """Test the conversion of a dictionary of lists of lists
        to an Awkward Array."""
        awkward_data = convert_to_awkward(self.toy_data_23)
        
        # Verify that the fields in the Awkward Array match
        #  the dictionary keys.
        for ak_key, key in zip(awkward_data.fields, self.toy_data_23.keys()):
            self.assertEqual(
                ak_key, key,
                f"Field '{ak_key}' does not match the dictionary key '{key}'"
            )

        # Validate each key's data.
        for key in self.toy_data_23.keys():
            self.assertIsInstance(
                awkward_data[key], ak.Array,
                f"{key} is not an Awkward Array."
            )
            self.assertEqual(
                ak.to_list(awkward_data[key]), self.toy_data_23[key],
                f"Values for key {key} do not match the original input."
            )
            for ak_sublist, sublist in zip(
                awkward_data[key], self.toy_data_23[key]
            ):
                self.assertTrue(
                    isinstance(ak_sublist, ak.Array),
                    f"The sublist {ak_sublist} is not an Awkward Array."
                )
                self.assertTrue(
                    all(type(x) == type(sublist[0]) or
                    isinstance(x, type(sublist[0])) for x in sublist),
                    f"Element type in sublist of {key} differs"
                    f" from the original input."
                )

    def test_save_to_root(self):
        """Test saving to a ROOT file."""
        output_file = Path("tests/toy_testing_save.root")
        save_to_root(
            output_file, 
            convert_to_awkward(self.toy_data_23),
            convert_to_awkward(self.toy_data_final)
        )
        self.assertTrue(
            output_file.exists(), 
            f"File {output_file} does not exist."
        )
        with uproot.open(output_file) as root_file:
            self.assertIn("tree_23", root_file, "tree_23 missing in ROOT file.")
            self.assertIn("tree_final", root_file, "tree_final missing in ROOT file.")


if __name__ == '__main__':
    unittest.main()