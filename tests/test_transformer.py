"""
Unit tests for ``pythiatransformer.transformer`` modue.

Contains unit tests to verify the functionality of
``transformer.ParticleTransformer`` class.
It uses the unittest framework to run the tests.
The tests include:

- checking that all the tensors used during the training are on the
  same device as the model;
- ensuring the output does not contain ``NaN`` or ``Inf``;
- ensuring the projection produces the correct shape;
- checks that the model's weights are updated after some epochs of
  training;
- verifies that the model can complete training and testing over 20
  epochs without issues;

All tests are written using Python ``unittest`` framework.
The features dimension is representing `pT` so it is fixed to ``1``.
"""

import unittest

import matplotlib.pyplot as plt
import torch
import torch.optim as optimizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pythiatransformer.transformer import ParticleTransformer


class TestParticleTransformer(unittest.TestCase):
    """Test case for ``transformer.ParticleTransformer`` class."""

    def setUp(self):
        """
        Initialize an instance of the model before each test.

        This function initializes the variables required to create an
        instance of ``transformer.ParticleTransformer`` class.
        This method is automatically called before each test to set up
        a consistent test environment.
        """
        seed = 1
        torch.Generator().manual_seed(seed)

        self.batch_size = 5

        # Initialization input tensors and their padding mask
        self.input_train = torch.rand(20, 2, 1)
        self.pad_mask_input_train = torch.zeros(20, 2, dtype=torch.bool)
        self.input_test = torch.rand(20, 2, 1)
        self.pad_mask_input_test = torch.zeros(20, 2, dtype=torch.bool)
        self.input_val = torch.rand(20, 2, 1)
        self.pad_mask_input_val = torch.zeros(20, 2, dtype=torch.bool)

        # Initialization target tensors and their padding mask
        self.target_train, self.pad_mask_target_train = self.target_creation(
            20, 15, 1
        )
        self.target_test, self.pad_mask_target_test = self.target_creation(
            20, 15, 1
        )
        self.target_val, self.pad_mask_target_val = self.target_creation(
            20, 15, 1
        )

        self.train_data = self.data_processing(
            self.input_train, self.target_train
        )
        self.val_data = self.data_processing(
            self.input_val, self.target_val, False
        )
        self.test_data = self.data_processing(
            self.input_test, self.target_test, False
        )

        self.train_data_pad_mask = self.data_processing(
            self.pad_mask_input_train, self.pad_mask_target_train
        )
        self.val_data_pad_mask = self.data_processing(
            self.pad_mask_input_val, self.pad_mask_target_val, False
        )
        self.test_data_pad_mask = self.data_processing(
            self.pad_mask_input_test, self.pad_mask_target_test, False
        )

        self.num_heads = 4
        self.num_encoder_layers = 1
        self.num_decoder_layers = 1
        self.num_units = 12
        self.dropout = 0.1
        self.activation = nn.ReLU()

        self.transformer = ParticleTransformer(
            self.train_data,
            self.val_data,
            self.test_data,
            self.train_data_pad_mask,
            self.val_data_pad_mask,
            self.test_data_pad_mask,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.num_units,
            self.dropout,
            self.activation,
        )

        self.loss_func = nn.MSELoss()
        self.optim = optimizer.Adam(self.transformer.parameters(), lr=1e-3)

    def target_creation(self, num_event, max_num_particles, num_features):
        """
        Generate padded targets and corresponding padding mask.

        Create padded target sequences and corresponding padding
        masks for a given number of events.
        Each event consists of a random number of particles (between 1
        and ``max_num_particles``). The sequences are padded with
        zeros to match ``max_num_particles``, and a padding mask is
        generated to indicate which elements are padding.

        Parameters
        ----------
        num_event: int
            Number of events to generate.
        max_num_particles: int
            Maximum number of particles.
        num_features: int
            Number of features for each particle.

        Returns
        -------
        target: torch.Tensor
            Target particles tensor.
        padding_mask: torch.Tensor
            Corresponding padding mask.
        """
        target = []
        padding_mask = []
        for _ in range(num_event):
            target_len = int(
                torch.randint(1, max_num_particles + 1, (1,)).item()
            )
            one_event_target = torch.rand(target_len, num_features)
            padded_target = nn.functional.pad(
                one_event_target,
                (0, 0, 0, max_num_particles - target_len),
                "constant",
                0,
            )
            target.append(padded_target)
            mask = torch.cat(
                [
                    torch.zeros(target_len, dtype=torch.bool),
                    torch.ones(
                        max_num_particles - target_len, dtype=torch.bool
                    ),
                ]
            )
            padding_mask.append(mask)
        target = torch.stack(target)
        padding_mask = torch.stack(padding_mask)

        return target, padding_mask

    def data_processing(self, input, target, shuffle=True):
        """
        Process the data for training.

        Prepare the data for training by splitting it into batches
        and shuffling the training data.

        Parameters
        ----------
        input: torch.Tensor
            Input tensor.
        target: torch.Tensor
            Target tensor.
        shuffle: bool
            True if the data are to be shuffled. Default is ``True``.

        Returns
        -------
        loader: Iterator
            An iterator for the data, with batching and shuffling
            enabled.
        """
        seed = 1
        generator = torch.Generator()
        generator.manual_seed(seed)
        set = TensorDataset(input, target)

        loader = DataLoader(
            set,
            self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )
        return loader

    def test_device_consistency(self):
        """
        Ensure device consistency across model and data.

        Verify that all tensors used in training (inputs, targets
        and their corresponding padding masks) are on the same device
        as the model to ensure device consistency during training and
        evaluation.
        """
        device = next(self.transformer.parameters()).device

        for (input, target), (input_pad, target_pad) in zip(
            self.train_data, self.train_data_pad_mask
        ):
            self.assertEqual(
                input.device,
                device,
                "Input tensor is not on the correct device",
            )
            self.assertEqual(
                target.device,
                device,
                "Target tensor is not on the correct device",
            )
            self.assertEqual(
                input_pad.device,
                device,
                "Input padding mask is not on the correct device",
            )
            self.assertEqual(
                target_pad.device,
                device,
                "Target padding mask is not on the correct device",
            )

    def test_forward_output_nans_infs(self):
        """Ensure no ``NaN`` or ``Inf`` in model outputs."""
        for (input, target), (input_padding_mask, target_padding_mask) in zip(
            self.train_data, self.train_data_pad_mask
        ):
            output, _ = self.transformer.forward(
                input, target, input_padding_mask, target_padding_mask
            )
            self.assertFalse(
                torch.isnan(output).any(), "Output contains NaN values"
            )
            self.assertFalse(
                torch.isinf(output).any(), "Output contains Inf values"
            )

    def test_projection_layer(self):
        """
        Test the functionality of the projection layers.

        Ensure that ``ParticleTransformer.input_projection`` method
        transforms the data from the input feature space to the hidden
        representation space, and that the ``particle_head`` correctly
        maps it back to the original feature dimension.
        """
        input_proj = self.transformer.input_projection(self.input_train)
        output_proj = self.transformer.particle_head(input_proj)
        self.assertEqual(input_proj.shape, (20, 2, self.num_units))
        self.assertEqual(output_proj.shape, (20, 2, 1))

    def test_weights_change_after_training(self):
        """Test that model weights are updated after training."""
        initial_weights = (
            self.transformer.input_projection.weight.detach().clone()
        )
        self.transformer.train_val(20, self.optim)
        final_weights = self.transformer.input_projection.weight.detach()
        self.assertFalse(
            torch.allclose(initial_weights, final_weights),
            "Weights did not change after training",
        )

    def plot_losses(self, train_loss, val_loss):
        """
        Plot the training and validation loss curves.

        Parameters
        ----------
        train_loss: list
            Training loss values per epoch.
        val_loss: list
            Validation loss values per epoch.
        """
        plt.figure()
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_training(self):
        """
        Test the full training and evaluation process.

        Run the full training, validation, and testing process for
        the model. It first trains the model for 20 epochs, using the
        specified loss function and optimizer.
        Then, it evaluates the model on the test data after training.
        """
        num_epoch = 20
        train_loss, val_loss = self.transformer.train_val(
            num_epoch, self.optim
        )
        self.plot_losses(train_loss, val_loss)


if __name__ == "__main__":
    unittest.main()
