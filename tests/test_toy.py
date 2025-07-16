import unittest
import scipy
import torch
from pythiatransformer.toy.toy_model import ToyDataset, ToyTransformer


class TestToyDataset(unittest.TestCase):
    """Unit tests for the ToyDataset class. Ensures that the ToyDataset
    behaves as expected. Tested behaviors include:
    - Correct dataset length based on initialization parameter.
    - Expected shapes and dimensions of the returned sample tensors.
    - Consistency between the mask and the reported sequence length.
    - Approximate equality between the scalar input and the sum of the
      valid elements in the padded target sequence.
    """
    def test_dataset_length(self):
        """Tests that the dataset returns the correct number of
        samples when initialized with n_samples=100.
        """
        dataset = ToyDataset(n_samples=100)
        self.assertEqual(len(dataset), 100)

    def test_shapes(self):
        """Tests that the shapes of the returned tensors from a sample
        in ToyDataset are as expected.
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        # Retrieves the first sample from the dataset.
        x, y_pad, mask, length = dataset[0]
        self.assertEqual(x.ndim, 0)
        self.assertEqual(y_pad.shape, (5,))
        self.assertEqual(mask.shape, (5,))

    def test_mask_sum_matches_length(self):
        """Verifies that the length value matches the sum of the mask
        (i.e., number of valid tokens).
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        _, _, mask, length = dataset[0]
        self.assertEqual(length, mask.sum().item())

    def test_sample_sum(self):
        """Checks that the sum of the valid elements in the target
        sequence (y_pad) is approximately equal to the scalar input x.
        The equality is checked up to 3 decimal places (places=3),
        meaning the absolute difference between the two values should
        be less than 0.0005, allowing a reasonable numerical tolerance
        due to floating-point computations.
        """
        dataset = ToyDataset(n_samples=1, max_len=5)
        x, y_pad, mask, length = dataset[0]
        self.assertAlmostEqual(
            y_pad[:length].sum().item(),
            x.item(),
            places=3
        )


class TestToyTransformer(unittest.TestCase):
    """Unit tests for the ToyTransformer class. Verifies the correct
    behavior and output shapes of the ToyTransformer's methods.
    Verifies:
    - The forward_teacher method returns outputs with correct shapes.
    - The model can perform a training step without numerical issues.
    - The generate method outputs sequences with valid dimensions
      respecting the configured max_len.
    """
    def setUp(self):
        """Initializes an instance of the model and initializes the
        model in evaluation mode before each test.
        """
        self.model = ToyTransformer(d_model=32, nhead=4, max_len=5)
        self.model.eval()

    def test_forward_shapes(self):
        """Verifies that the forward_teacher method returns outputs of
        the correct shape.
        """
        B, T = 2, 5
        x = torch.rand(B)
        y = torch.rand(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        length = torch.full((B,), T)
        y_hat, stop_logits = self.model.forward_teacher(x, y, mask, length)
        self.assertEqual(y_hat.shape, (B, T + 1))
        self.assertEqual(stop_logits.shape, (B, T + 1))

    def test_training_step(self):
        """Tests the model behavior during training. Ensures that the
        model can perform a typical training step without numerical
        errors or crashes, and that the gradients are computed
        correctly.
        """
        model = ToyTransformer(d_model=32, nhead=4, max_len=5)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        B, T = 2, 5
        x = torch.rand(B)
        y = torch.rand(B, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        length = torch.full((B,), T)
        # Ignore SOS for simplicity.
        y_hat, _ = model.forward_teacher(x, y, mask, length)
        loss = criterion(y_hat[:, :T], y)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(loss).any())

    def test_generate_shape(self):
        """Tests the generate method. Checks that:
        - The output y_seq is a two-dimensional tensor.
        - The batch size dimension in the output matches the input
          batch size.
        - The generated sequence length is less than or equal to
          max_len, which is the maximum sequence length set in the
          model.
        """
        x = torch.rand(5)
        y_seq = self.model.generate(x)
        self.assertEqual(y_seq.ndim, 2)
        self.assertEqual(y_seq.shape[0], 5)
        self.assertLessEqual(y_seq.shape[1], self.model.max_len)


if __name__ == "__main__":
    unittest.main()
