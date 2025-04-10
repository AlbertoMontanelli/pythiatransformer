import logging
import torch
import torch.nn as nn
import torch.optim as optimizer
import unittest

from pythiatransformer.transformer import ParticleTransformer

# logging set up
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')


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
    """
    def setUp(self):
        """This function initializes the variables required to create an
        instance of the ParticleTransformers class.
        This method is automatically called before each test to set up
        a consistent test environment.
        """
        self.input_train = torch.rand(10, 2, 6)
        self.input_test = torch.rand(10, 2, 6)
        self.input_val = torch.rand(10, 2, 6)
        self.target_train = torch.rand(10, 12, 6)
        self.target_test = torch.rand(10, 12, 6)
        self.target_val = torch.rand(10, 12, 6)

        self.dim_features = 6
        self.num_heads = 4
        self.num_encoder_layers = 1
        self.num_decoder_layers = 1
        self.num_units = 12
        self.dropout = 0.1
        self.batch_size = 2
        self.transformer = ParticleTransformer(
            self.input_train,
            self.input_val,
            self.input_test,
            self.target_train,
            self.target_val,
            self.target_test,
            self.dim_features,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.num_units,
            self.dropout,
            self.batch_size
        )

        self.loss_func = nn.MSELoss()
        self.optim = optimizer.Adam(self.transformer.parameters(), lr=1e-3)
       

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
        output = self.transformer.forward(self.input_train, self.target_train)
        self.assertEqual(output.shape, self.target_train.shape)

    def test_forward_output_nans_infs(self):
        """This function tests that the putput of the forward methods does
        not contain Inf values or NaN values.
        """
        output = self.transformer.forward(self.input_train, self.target_train)
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
        self.assertEqual(input_proj.shape, (10, 2, self.num_units))
        self.assertEqual(output_proj.shape, (10, 2, self.dim_features))
    
    def test_training(self):
        """
        """
        self.transformer.train_val(20, self.loss_func, self.optim)
        #self.transformer.test(self.loss_func)


if __name__ == '__main__':
    unittest.main()
