"""
Unit tests for ``pythiatransformer.data_processing`` module.

The tests include:

- check on the funcionality of Exceptions;
- check ``padded_tensor``, ``padding_mask`` and ``tot_pt`` if
  ``truncate_pT = 0``;
- check ``padded_tensor``, ``padding_mask`` if ``truncate_pT = 1``,
  check that the second dataset is actually truncated at 50% pT;
- check length of batches;
- check splitting in training, validation, and test sets.

All tests are written using Python ``unittest`` framework.
"""

import math
import unittest

import awkward as ak
import torch

from pythiatransformer.data_processing import (
    awkward_to_padded_tensor,
    batching,
    train_val_test_split,
)


class TestDataProcessing(unittest.TestCase):
    """Test case for ``pythiatransformer.data_processing`` module."""

    def test_awkward_to_padded_tensor_invalid(self):
        """Test invalid types of parameters of original function."""
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(data=[1, 2, 3], features=["pT_23"])
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_23": [[1]]}), features="pT_23"
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_23": [[1]]}), features=[1, 2, 3]
            )
        with self.assertRaises(KeyError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_23": [[1]]}), features=["E"]
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_final": [[1]]}),
                features=["pT_final"],
                list_pt=[],
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_final": [[1]]}),
                features=["pT_final"],
                list_pt=3,
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_final": [[1]]}),
                features=["pT_final"],
                list_pt=[3, 4],
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT_final": [[1]]}),
                features=["pT_final"],
                list_pt=["3"],
            )

    def test_awkward_to_padded_tensor_no_trunc(self):
        """Test original function if ``list_pt is None``."""
        data = ak.Array({"pT_23": [[1.0, 2.0, 3.0], [4.0]]})
        tensor, mask, tot_pt = awkward_to_padded_tensor(data, ["pT_23"])
        assert isinstance(tot_pt, torch.Tensor)
        self.assertEqual(
            tensor.shape,
            (2, 3, 1),
            "Data tensor does not have the expected shape",
        )
        self.assertEqual(
            mask.shape, (2, 3), "Mask tensor does not have the expected shape"
        )
        self.assertEqual(
            mask.dtype,
            torch.bool,
            "Mask tensor does not have torch.bool type",
        )
        self.assertFalse(
            mask[0, 0],
            "Padding when corresponding particle is a true particle",
        )  # check true particle.
        self.assertTrue(
            mask[1, 2],
            "No padding when corresponding particle is not a true particle",
        )  # check padding
        self.assertTrue(
            torch.allclose(tot_pt, torch.Tensor([6.0, 4.0])),
            "Error in summing pT per event",
        )

    def test_awkward_to_padded_tensor_trunc(self):
        """Test original function if ``list_pt is not None``."""
        data_23 = ak.Array({"pT_23": [[1.0, 2.0, 3.0], [4.0]]})
        data_f = ak.Array(
            {"pT_final": [[2.0, 2.0, 3.0], [1.0, 1.0, 1.0, 2.0]]}
        )
        _, _, tot_pt_23 = awkward_to_padded_tensor(data_23, ["pT_23"])
        tensor_f, _, _ = awkward_to_padded_tensor(
            data_f,
            ["pT_final"],
            tot_pt_23,
        )
        assert isinstance(tot_pt_23, torch.Tensor)
        self.assertEqual(
            tensor_f.shape[0],
            tot_pt_23.shape[0],
            "Target tensor does not have same number of events as "
            "events for which pT has been computed",
        )
        sum_pt_f = tensor_f.squeeze(-1).sum(dim=1)
        self.assertTrue(
            torch.all(sum_pt_f >= 0.5 * tot_pt_23),
            "Sum of pT of targets is not always at least 50perc of pT"
            " of corresponding data",
        )

    def test_batching_invalid(self):
        """Test invalid types of parameters of original function."""
        invalid_3 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        valid_3 = torch.arange(6).view(3, 2)
        valid_2 = torch.arange(6).view(2, 3)
        with self.assertRaises(TypeError):
            batching(invalid_3, valid_3, batch_size=1)
        with self.assertRaises(TypeError):
            batching(valid_3, invalid_3, batch_size=1)
        with self.assertRaises(ValueError):
            batching(valid_3, valid_2, batch_size=1)
        with self.assertRaises(TypeError):
            batching(valid_2, valid_2, batch_size=1.5)
        with self.assertRaises(ValueError):
            batching(valid_2, valid_2, batch_size=5)
        with self.assertRaises(ValueError):
            batching(valid_2, valid_2, batch_size=-4)

    def test_batching_len(self):
        """Test length of batches.

        Verify the behaviour when ``batch_size`` does not divide
        perfectly the dataset.
        """
        inputs = torch.arange(20).view(10, 2)
        targets = torch.arange(21, 61).view(10, 4)
        batch_size = 3
        batches = batching(inputs, targets, batch_size, False)
        self.assertEqual(
            len(batches),
            math.ceil(len(inputs) / batch_size),
            "Batch length is not what expected",
        )

    def test_train_val_test_split_invalid(self):
        """Test invalid types of parameters of original function."""
        valid = torch.arange(10).view(5, 2)
        with self.assertRaises(TypeError):
            train_val_test_split([1, 2, 3, 4], 0.5, 0.25, 0.25)
        with self.assertRaises(ValueError):
            train_val_test_split(valid, 0.5, 0.3, 0.1)
        with self.assertRaises(ValueError):
            train_val_test_split(valid, -0.2, 0.8, 0.4)

    def test_train_val_test_split(self):
        """Test splitting the dataset."""
        tensor = torch.arange(10)
        train, val, test = train_val_test_split(tensor, 0.6, 0.3, 0.1)
        expected_train = torch.arange(6)
        expected_val = torch.arange(6, 9)
        expected_test = torch.Tensor([9])
        self.assertTrue(
            torch.equal(train, expected_train),
            "Training set size is not what expected",
        )
        self.assertTrue(
            torch.equal(val, expected_val),
            "Validation set size is not what expected",
        )
        self.assertTrue(
            torch.equal(test, expected_test),
            "Test set size is not what expected",
        )


if __name__ == "__main__":
    unittest.main()
