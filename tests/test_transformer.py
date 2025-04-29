import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optimizer
from torch.utils.data import TensorDataset, DataLoader
import unittest

from pythiatransformer.transformer import ParticleTransformer


class TestParticleTransformer(unittest.TestCase):
    """This class contains unit tests to verify the functionality of the
    ParticleTransformer class.
    It uses the unittest framework to run the tests.
    The tests include:
    - Ensuring the dim_features matches the dim_features of the torch
      tensor.
    - Ensuring the output shape of the forward method matches the input
      shape.
    - Ensuring the output does not contain NaN or Inf.
    - Ensuring the projection produces the correct shape.
    - Verifies that the model can complete training and testing over 20
      epochs without issues.
    """
    def setUp(self):
        """This function initializes the variables required to create an
        instance of the ParticleTransformers class.
        This method is automatically called before each test to set up
        a consistent test environment.
        """
        seed = 1
        torch.Generator().manual_seed(seed)

        self.dim_features = 4
        self.batch_size = 5

        # Initialization input tensors and them padding mask
        self.input_train = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_train = torch.zeros(20, 2, dtype = torch.bool)
        self.input_test = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_test = torch.zeros(20, 2, dtype = torch.bool)
        self.input_val = torch.rand(20, 2, self.dim_features)
        self.pad_mask_input_val = torch.zeros(20, 2, dtype = torch.bool)

        # Initialization target tensors and them padding mask
        self.target_train, self.pad_mask_target_train = self.target_creation(
            20,
            15,
            self.dim_features
        )
        self.target_test, self.pad_mask_target_test = self.target_creation(
            20,
            15,
            self.dim_features
        )
        self.target_val, self.pad_mask_target_val = self.target_creation(
            20,
            15,
            self.dim_features
        )

        self.train_data = self.data_processing(self.input_train, self.target_train)
        self.val_data = self.data_processing(self.input_val, self.target_val, False)
        self.test_data = self.data_processing(self.input_test, self.target_test, False)

        self.train_data_pad_mask = self.data_processing(self.pad_mask_input_train, self.pad_mask_target_train)
        self.val_data_pad_mask = self.data_processing(self.pad_mask_input_val, self.pad_mask_target_val, False)
        self.test_data_pad_mask = self.data_processing(self.pad_mask_input_test, self.pad_mask_target_test, False)

        print(f"train data: {self.train_data}")
        print(f"padding_mask: {self.train_data_pad_mask}")


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
            self.activation
        )

        self.loss_func = nn.MSELoss()
        self.optim = optimizer.Adam(self.transformer.parameters(), lr=1e-3)
    
    def target_creation(self, num_event, max_num_particles, num_features):
        """
        """
        target = []
        padding_mask = []
        for i in range (num_event):
            target_len = torch.randint(1, max_num_particles + 1, (1,)).item()
            one_event_target = torch.rand(target_len, num_features)
            padded_target = nn.functional.pad(one_event_target, (0, 0, 0, max_num_particles - target_len), "constant", 0)
            target.append(padded_target)
            mask = torch.cat([
                torch.zeros(target_len, dtype=torch.bool),
                torch.ones(max_num_particles - target_len, dtype=torch.bool)
            ])
            padding_mask.append(mask)
            # from list to torch.tensor
        target = torch.stack(target)
        padding_mask = torch.stack(padding_mask)

        return target, padding_mask
    
    def data_processing(self, input, target, shuffle = True):
        """This function prepares the data for training by splitting it
        into batches and shuffling the training data.

        Args:
            shuffle (bool):
        Returns:
            loader (Iterator): An iterator for the training
                                        data, with batching and
                                        shuffling enabled.
            loader (Iterator): An iterator for the test data, with
                                    batching and shuffling enabled.

        """
        seed = 1
        generator = torch.Generator() # creation of a new generator
        generator.manual_seed(seed)
        set = TensorDataset(input, target)

        loader = DataLoader(
            set,
            self.batch_size,
            shuffle = shuffle,
            generator = generator if shuffle else None
        )

        return loader

    def test_dim_features(self):
        """This function checks that the third dimension of the input
        tensor (i.e., the number of features per particle) matches
        the dim_features parameter of the model.
        """
        self.assertEqual(
            self.input_train.shape[2],
            self.dim_features,
            f"Mismatch in feature dimensions: expected {self.dim_features}, "
            f"got {self.input_train.shape[2]}"
        )

    def test_forward_output_shape(self):
        """This function tests that the forward method of the model
        produces an output tensor with the correct shape. The shape of
        the output tensor must match with the shape of the input tensor.
        """
        for (input, target), (input_padding_mask, target_padding_mask) in zip(self.train_data, self.train_data_pad_mask):
            target, target_padding_mask, attention_mask = self.transformer.de_padding(target, target_padding_mask)
            output = self.transformer.forward(
                input,
                target,
                input_padding_mask,
                target_padding_mask,
                attention_mask
            )
            self.assertEqual(output.shape, target.shape)

    def test_forward_output_nans_infs(self):
        """This function tests that the putput of the forward methods does
        not contain Inf values or NaN values.
        """
        for (input, target), (input_padding_mask, target_padding_mask) in zip(self.train_data, self.train_data_pad_mask):
            target, target_padding_mask, attention_mask = self.transformer.de_padding(target, target_padding_mask)
            output = self.transformer.forward(
                input,
                target,
                input_padding_mask,
                target_padding_mask,
                attention_mask
            )
            self.assertFalse(
                torch.isnan(output).any(),
                "Output contains NaN values"
            )
            self.assertFalse(
                torch.isinf(output).any(),
                "Output contains Inf values"
            )

    def test_projection_layer(self):
        """This function tests the functionality of the projection layers.
        Ensures that the input projection transforms the data from the
        input feature space to the hidden representation space, and
        that the output projection correctly maps it back to the
        original feature dimension.
        """
        input_proj= self.transformer.input_projection(self.input_train)
        output_proj= self.transformer.output_projection(input_proj)
        self.assertEqual(input_proj.shape, (100, 2, self.num_units))
        self.assertEqual(output_proj.shape, (100, 2, self.dim_features))
    
    def plot_losses(self, train_loss, val_loss):
        """
        """
        plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_training(self):
        """This function runs the full training, validation, and
        testing process for the model. It first trains the model for 20
        epochs, using the specified loss function and optimizer.
        Then, it evaluates the model on the test data after training.
        """
        num_epoch = 100
        train_loss, val_loss = self.transformer.train_val(num_epoch, self.loss_func, self.optim)
        self.plot_losses(train_loss, val_loss)


if __name__ == '__main__':
    unittest.main()
