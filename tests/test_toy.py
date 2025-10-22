"""
Unit tests for ``pythiatransformer.toy.toy_model`` module.

-``toy_model.ToyDataset`` class:
    Ensure that the class behave as expected.
    Tested behaviours include:

    - correct dataset length based on initialization parameter;
    - expected shapes and dimensions of the returned sample tensors;
    - consistency between the mask and the reported sequence length;
    - approximate equality between the scalar input and the sum of the
      valid elements in the padded target sequence.

-``toy_model.ToyTransformer`` class:
    Verify the correct behaviour and output shapes of the different
    class methods.
    Tested behaviours include:

    - ``ToyTransformer.forward_teacher`` returns outputs with correct
      shapes;
    - the model can perform a training step without numerical issues;
    - the generate method outputs sequences with valid dimensions
      respecting the configured ``max_len``.

All tests are written using Python ``unittest`` framework.
"""

import unittest

import torch

from pythiatransformer.toy.toy_model import ToyDataset, ToyTransformer


class TestToyDataset(unittest.TestCase):
    """Test case for ``toy_model.ToyDataset`` class."""

    def test_dataset_length(self):
        """
        Test dataset length.

        Assure that the dataset returns the correct number of samples
        when initialized with ``n_samples=100``.
        """
        dataset = ToyDataset(n_samples=100)
        self.assertEqual(len(dataset), 100)

    def test_shapes(self):
        """
        Test shapes of returned tensors.

        Assure that the shapes of the returned tensors from a sample
        in ``toy_model.ToyDataset`` class are as expected.
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        # Retrieves the first sample from the dataset.
        x, y_pad, mask, _ = dataset[0]
        self.assertEqual(x.ndim, 0)
        self.assertEqual(y_pad.shape, (5,))
        self.assertEqual(mask.shape, (5,))

    def test_mask_sum_matches_length(self):
        """
        Test consistency between mask and length.

        Verify that the length value matches the sum of the mask
        (i.e. the number of valid tokens).
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        _, _, mask, length = dataset[0]
        self.assertEqual(length, mask.sum().item())

    def test_sample_sum(self):
        """
        Test sum of target sequence matches input scalar.

        Checks that the sum of the valid elements in the target
        sequence (``y_pad``) is approximately equal to the scalar input
        ``x``. The equality is checked up to 3 decimal places
        (``places=3``), meaning the absolute difference between the two
        values should be less than 0.0005, allowing a reasonable
        numerical tolerance due to floating-point computations.
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        x, y_pad, _, length = dataset[0]
        self.assertAlmostEqual(
            y_pad[:length].sum().item(), x.item(), places=3
        )


class TestToyTransformer(unittest.TestCase):
    """Test case for ``toy_model.ToyTransformer`` class."""

    def setUp(self):
        """
        Initialize an instance of the model before each test.

        After initialization of an istance of the model, the setup
        method puts the model in evaluation mode before each test.
        """
        self.model = ToyTransformer(d_model=32, nhead=4, max_len=5)
        self.model.eval()

    def test_forward_shapes(self):
        """Test output shapes of forward_teacher method."""
        b, t = 2, 5
        x = torch.rand(b)
        y = torch.rand(b, t)
        mask = torch.ones(b, t, dtype=torch.bool)
        y_hat, stop_logits = self.model.forward_teacher(x, y, mask)
        self.assertEqual(y_hat.shape, (b, t + 1))
        self.assertEqual(stop_logits.shape, (b, t + 1))

    def test_training_step(self):
        """
        Test the model behaviour during training.

        Ensure that the model can perform a typical training step
        without numerical errors or crashes, and that the gradients
        are computed correctly.
        """
        model = ToyTransformer(d_model=32, nhead=4, max_len=5)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        b, t = 2, 5
        x = torch.rand(b)
        y = torch.rand(b, t)
        mask = torch.ones(b, t, dtype=torch.bool)
        # Ignore SOS for simplicity.
        y_hat, _ = model.forward_teacher(x, y, mask)
        loss = criterion(y_hat[:, :t], y)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(loss).any())

    def test_generate_shape(self):
        """
        Test the generate method.

        Check that:

        - the output ``y_seq`` is a two-dimensional tensor;
        - the batch size dimension in the output matches the input
          batch size;
        - the generated sequence length is less than or equal to
          ``max_len``, which is the maximum sequence length set in
          the model.
        """
        x = torch.rand(5)
        y_seq = self.model.generate(x)
        self.assertEqual(y_seq.ndim, 2)
        self.assertEqual(y_seq.shape[0], 5)
        self.assertLessEqual(y_seq.shape[1], self.model.max_len)


if __name__ == "__main__":
    unittest.main()
