import numpy as np
import pandas as pd
import torch
import unittest

from pythiatransformer.data_processing import (
    preprocess_dataframe,
    dataframe_to_padded_tensor,
    train_val_test_split,
)


class TestPreprocessDataframe(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "nid": [2],
                "id": ["[11, -12]"],
                "status": ["[23, 23]"],
                "px": ["[2.0, -4.0]"],
                "py": ["[-3.0, 6.0]"],
                "pz": ["[2.0, -2.0]"],
                "e": ["[4.2, 7.5]"],
                "m": ["[0.511, 0.511]"],
            }
        )

    def test_output_shape(self):
        df_stand = preprocess_dataframe(self.df)
        self.assertEqual(df_stand.shape[0], 1)
        self.assertIn("id", df_stand.columns)


class TestTrainValTestSplit(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.tensor(
            [
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                [[-1, -2, -3, -4, -5, -6], [-7, -8, -9, -10, -11, -12]],
            ]
        )

    def test_output_lenght(self):
        train, val, test = train_val_test_split(self.tensor)
        self.assertIsInstance(train, torch.Tensor)
        self.assertEqual(len(train) + len(val) + len(test), len(self.tensor))
