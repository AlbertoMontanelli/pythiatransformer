import awkward as ak
import numpy as np
import torch
import unittest

from pythiatransformer.data_processing import (
    awkward_to_padded_tensor,
    batching,
    train_val_test_split
)


class TestDataProcessing(unittest.TestCase):
    """
    Contain unit tests to verify the functionality of the code
    data_processing.py.
    Using unittest framework to run the tests.
    The tests include:
    - Check on the funcionality of raised Exceptions
    - Check padded_tensor, padding_mask and tot_pt if truncate_pT = 0
    - Check padded_tensor, padding_mask if truncate_pT = 1, check that
    the second dataset is actually truncated at 50% pT
    - 
    """
    def test_awkward_to_padded_tensor_invalid_types(self):
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=[1, 2, 3], features=["pT"]
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT"
            )
        with self.assertRaises(TypeError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=[1, 2, 3]
            )
        with self.assertRaises(KeyError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="E"
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT", truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT", list_pt=[], truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT", list_pt=3, truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT", list_pt=[3, 4], truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features="pT", list_pt=["3"], truncate_pt=True
            )
    

    def test_awkward_to_padded_tensor_no_trunc(self):
        data = ak.Array({"pT": [[1., 2., 3.], [4.]]})
        tensor, mask, tot_pT = awkward_to_padded_tensor(
            data, ["pT"]
        )
        self.assertEqual(tensor.shape, (2, 3, 1))
        self.assertTrue(
           torch.all(tensor[:-1] >= tensor[1:])
        )
        self.assertEqual(mask.shape, (2, 3))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask[0, 0]) # check true particle.
        self.assertTrue(mask[1, 2]) # check padding.
        self.assertTrue(torch.allclose(tot_pT, torch.Tensor([6., 4.])))

    def test_awkward_to_padded_tensor_trunc(self):
        data_23 = ak.Array({"pT": [[1., 2., 3.], [4.]]})
        data_f = ak.Array({"pT": [[2., 2., 3.], [1., 1., 1., 2.]]})
        tensor_23, mask_23, tot_pT_23 = awkward_to_padded_tensor(
            data, ["pT"]
        )
        tensor_f, mask_f = awkward_to_padded_tensor(
            data_f, ["pT"], tot_pT_23, True
        )
        self.assertEqual(tensor_f.shape, (2, 4, 1))
        self.assertEqual(mask_f.shape, (2, 4))
        self.assertFalse(mask_f[0, 0]) # check true particle.
        self.assertTrue(mask_f[0, 3]) # check padding.
        self.assertEqual(tensor_f.shape[0], tot_pT_23.shape[0])
        sum_pT_f = tensor_f.squeeze(-1).sum(dim=1)
        self.assertTrue(torch.allclose(
            torch.all((sum_pT_f > 0.5*tot_pT_23) & (sum_pT_f < 0.7*tot_pT_23))
        ))
