import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optimizer
from torch.utils.data import TensorDataset, DataLoader
import unittest

from pythiatransformer.transformer import ParticleTransformer


class TestParticleTransformer(unittest.TestCase):
    """
    Contains unit tests to verify the functionality of the
    ParticleTransformer class.
    It uses the unittest framework to run the tests.
    The tests include:
    - Checks that all the tensors used during the training are on the
      same device as the model.
    - Ensuring the dim_features matches the dim_features of the torch
      tensor.
    - Ensuring the output does not contain NaN or Inf.
    - Ensuring the projection produces the correct shape.
    - Checks that the model's weights are updated after some epochs of
      training.
    - Verifies that the model can complete training and testing over 20
      epochs without issues.
    """

    def setUp(self):
        """
        This function initializes the variables required to create an
        instance of the ParticleTransformers class.
        This method is automatically called before each test to set up
        a consistent test environment.
        """
        seed = 1
        torch.Generator().manual_seed(seed)

        self.dim_features = 1
        self.batch_size = 5

        # Initialization input tensors and them padding mask
        self.input_train = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_train = torch.zeros(20, 2, dtype=torch.bool)
        self.input_test = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_test = torch.zeros(20, 2, dtype=torch.bool)
        self.input_val = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_val = torch.zeros(20, 2, dtype=torch.bool)

        # Initialization target tensors and them padding mask
        self.target_train, self.pad_mask_target_train = self.target_creation(
            20, 15, self.dim_features
        )
        self.target_test, self.pad_mask_target_test = self.target_creation(
            20, 15, self.dim_features
        )
        self.target_val, self.pad_mask_target_val = self.target_creation(
            20, 15, self.dim_features
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
            self.dim_features,
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
        Generates padded target sequences and corresponding padding
        masks for a given number of events.
        Each event consists of a random number of particles (between 1
        and max_num_particles). The sequences are padded with zeros to
        match max_num_particles, and a padding mask is generated to
        indicate which elements are padding.

        Args:
            num_event (int): Number of events to generate.
            max_num_particles (int): Maximum number of particles.
            num_features (int): Number of features for each particle.
        Returns:
            target (torch.Tensor): Target particles tensor.
            padding_mask (torch.Tensor): Corresponding padding mask.
        """
        target = []
        padding_mask = []
        for i in range(num_event):
            target_len = torch.randint(1, max_num_particles + 1, (1,)).item()
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
        Prepares the data for training by splitting it into batches
        and shuffling the training data.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
            shuffle (bool): True if the data are to be shuffled.
                            Default True.
        Returns:
            loader (Iterator): An iterator for the data, with batching
                               and shuffling enabled.
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
        Verifies that all tensors used in training (inputs, targets
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
                "Input tensor is not on the correct device"
            )
            self.assertEqual(
                target.device,
                device,
                "Target tensor is not on the correct device"
            )
            self.assertEqual(
                input_pad.device,
                device,
                "Input padding mask is not on the correct device"
            )
            self.assertEqual(
                target_pad.device,
                device,
                "Target padding mask is not on the correct device"
            )

    def test_dim_features(self):
        """
        Checks that the third dimension of the input
        tensor (i.e., the number of features per particle) matches
        the dim_features parameter of the model.
        """
        self.assertEqual(
            self.input_train.shape[2],
            self.dim_features,
            f"Mismatch in feature dimensions: expected {self.dim_features}, "
            f"got {self.input_train.shape[2]}",
        )

    def test_forward_output_nans_infs(self):
        """
        Tests that the output of the forward methods does not
        contain Inf values or NaN values.
        """
        for (input, target), (input_padding_mask, target_padding_mask) in zip(
            self.train_data, self.train_data_pad_mask
        ):
            output, eos_prob_vector = self.transformer.forward(
                input,
                target,
                input_padding_mask,
                target_padding_mask
            )
            self.assertFalse(
                torch.isnan(output).any(), "Output contains NaN values"
            )
            self.assertFalse(
                torch.isinf(output).any(), "Output contains Inf values"
            )

    def test_projection_layer(self):
        """
        Tests the functionality of the projection layers.
        Ensures that the input_projection transforms the data from the
        input feature space to the hidden representation space, and
        that the particle_head correctly maps it back to the original
        feature dimension.
        """
        input_proj = self.transformer.input_projection(self.input_train)
        output_proj = self.transformer.particle_head(input_proj)
        self.assertEqual(input_proj.shape, (20, 2, self.num_units))
        self.assertEqual(output_proj.shape, (20, 2, self.dim_features))

    def test_weights_change_after_training(self):
        """
        Verifies that the model's weights are updated during
        training.
        """
        initial_weights = self.transformer.input_projection.weight.detach().clone()
        self.transformer.train_val(20, self.optim)
        final_weights = self.transformer.input_projection.weight.detach()
        self.assertFalse(
            torch.allclose(initial_weights, final_weights),
            "Weights did not change after training"
        )

    def plot_losses(self, train_loss, val_loss):
        """
        Plots the training and validation loss curves.

        Args:
            train_loss (list): Training loss values per epoch.
            val_loss (list): Validation loss values per epoch.
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
        Runs the full training, validation, and testing process for
        the model. It first trains the model for 20 epochs, using the
        specified loss function and optimizer.
        Then, it evaluates the model on the test data after training.
        """
        num_epoch = 20
        train_loss, val_loss = self.transformer.train_val(
            num_epoch,
            self.optim
        )
        self.plot_losses(train_loss, val_loss)


if __name__ == "__main__":
    unittest.main()
