import awkward as ak
import math
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
    - Check on the funcionality of Exceptions
    - Check padded_tensor, padding_mask and tot_pt if truncate_pT = 0
    - Check padded_tensor, padding_mask if truncate_pT = 1, check that
    the second dataset is actually truncated at 50% pT
    - Check length and alignment of batches
    - Check splitting in training, validation, and test set
    """
    def test_awkward_to_padded_tensor_invalid(self):
        """
        Test invalid types of parameters of original function.
        """
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
                data=ak.Array({"pT": [[1]]}), features=["E"]
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=["pT"], truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=["pT"], list_pt=[], truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=["pT"], list_pt=3, truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=["pT"], list_pt=[3, 4], truncate_pt=True
            )
        with self.assertRaises(ValueError):
            awkward_to_padded_tensor(
                data=ak.Array({"pT": [[1]]}), features=["pT"], list_pt=["3"], truncate_pt=True
            )
    

    def test_awkward_to_padded_tensor_no_trunc(self):
        """
        Test original function if truncate_pt=False.
        """
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
        """
        Test original function if truncate_pt=True.
        """
        data_23 = ak.Array({"pT": [[1., 2., 3.], [4.]]})
        data_f = ak.Array({"pT": [[2., 2., 3.], [1., 1., 1., 2.]]})
        tensor_23, mask_23, tot_pT_23 = awkward_to_padded_tensor(
            data_23, ["pT"]
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

    def test_batching_invalid(self):
        """
        Test invalid types of parameters of original function.
        """
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
        """
        Test length of batches when batch_size does not
        divide perfectly the dataset.
        """
        inputs = torch.arange(20).view(10, 2)
        targets = torch.arange(21, 61).view(10, 4)
        batch_size = 3
        batches = batching(inputs, targets, batch_size, False)
        self.assertEqual(len(batches), math.ceil(len(inputs)/batch_size))

    def test_batching_shuffle_alignment(self):
        """
        Test alignment of input and target if shuffle=True.
        """        
        inputs = torch.arange(20).view(10, 2)
        targets = torch.arange(10)
        batch_size = 5
        batches = batching(inputs, targets, batch_size, True)

        shuffled_inputs = torch.cat([bb[0] for bb in batches], dim=0)
        shuffled_targets = torch.cat([bb[1] for bb in batches], dim=0)

        for ii in range(len(inputs)):
            input_row = shuffled_inputs[ii]
            # Since inputs = [[0,1], [2,3], [4,5],...] the expected
            # target is the first number of the row // 2.
            expected_target = (input_row[0] // 2).item()
            self.assertEqual(expected_target, shuffled_targets[ii].item())

    def test_train_val_test_split_invalid(self):
        """
        Test invalid types of parameters of original function.
        """
        valid = torch.arange(10).view(5,2)
        with self.assertRaises(TypeError):
            train_val_test_split([1, 2, 3, 4], 0.5, 0.25, 0.25)
        with self.assertRaises(ValueError):
            train_val_test_split(valid, 0.5, 0.3, 0.1)
        with self.assertRaises(ValueError):
            train_val_test_split(valid, -0.2, 0.8, 0.4)
        with self.assertRaises(ValueError):
            train_val_test_split(valid, 0.5, 0.4, 0.1)

    def test_train_val_test_split(self):
        """
        Test splitting the dataset.
        """
        tensor = torch.arange(10)
        train, val, test = train_val_test_split(
            tensor, 0.6, 0.3, 0.1
        )
        expected_train = torch.arange(6)
        expected_val = torch.arange(6, 9)
        expected_test = torch.Tensor([9])
        self.assertTrue(torch.Equal(train, expected_train))
        self.assertTrue(torch.Equal(val, expected_val))
        self.assertTrue(torch.Equal(test, expected_test))


if __name__ == "__main__":
    unittest.main()
