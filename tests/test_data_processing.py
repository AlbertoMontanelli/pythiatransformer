import torch
import unittest
import awkward as ak
from loguru import logger

from pythiatransformer.data_processing import awkward_to_padded_targets

class TestEOSInsertion(unittest.TestCase):
    def setUp(self):
        # Use real Awkward Array for compatibility
        self.dummy_data = ak.Array({
            "id_final": [[11, -11], [13, -13, 22]],
            "px_final": [[1.0, -1.0], [0.5, -0.5, 0.0]],
            "py_final": [[0.0, 0.0], [1.0, -1.0, 0.0]],
            "pz_final": [[0.5, -0.5], [1.5, -1.5, 0.0]],
            "pT_final": [[1.0, 1.0], [1.12, 1.12, 0.0]]
        })
        self.features = ["id_final", "px_final", "py_final", "pz_final", "pT_final"]
        self.eos_token = -999

    def test_eos_is_inserted_correctly(self):
        padded_tensor, padding_mask = awkward_to_padded_targets(
            self.dummy_data, self.features, eos_token=self.eos_token
        )

        id_channel = padded_tensor[:, :, 0]
        last_valid_index = padding_mask.sum(dim=1) - 1
        eos_ids = id_channel[torch.arange(id_channel.size(0)), last_valid_index]

        for i, eos_id in enumerate(eos_ids):
            logger.info(f"EOS ID for event {i}: {eos_id.item()}")
            self.assertEqual(
                eos_id.item(), self.eos_token,
                f"EOS token not found at the last valid position of event {i}"
            )

    def test_eos_vector_content(self):
        padded_tensor, padding_mask = awkward_to_padded_targets(
            self.dummy_data, self.features, eos_token=self.eos_token
        )

        eos_index = padding_mask.sum(dim=1)[0].item() - 1
        eos_vector = padded_tensor[0, eos_index]

        logger.info(f"EOS vector for first event: {eos_vector}")
        self.assertEqual(eos_vector[0].item(), self.eos_token)
        self.assertTrue(torch.all(eos_vector[1:] == 0), "EOS vector should have zero px/py/pz")

if __name__ == "__main__":
    unittest.main()


# class TestTrainValTestSplit(unittest.TestCase):
#     def setUp(self):
#         self.tensor = torch.tensor([
#             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
#             [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
#             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
#             [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
#             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
#             [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
#             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
#             [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
#             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
#             [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]]
#         ])

#     def test_output_lenght(self):
#         train, val, test = train_val_test_split(self.tensor)
#         self.assertIsInstance(train, torch.Tensor)
#         self.assertEqual(len(train)+len(val)+len(test), len(self.tensor))