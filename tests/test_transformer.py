import logging
import torch
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
        self.dim_features = 4
        self.num_heads = 4
        self.num_encoder_layers = 1
        self.num_decoder_layers = 1
        self.num_units = 12
        self.dropout = 0.1
        self.transformer = ParticleTransformer(
            self.dim_features,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.num_units,
            self.dropout
        )

        self.input = torch.tensor(
            [
                # event 1
                [
                    # 3 particles for each event
                    # 4 features for each particles
                    [1.0, 2.0, 3.0, 5.0],
                    [4.0, 5.0, 6.0, 3.0],
                    [7.0, 8.0, 9.0, 2.0],
                ],
                # event 2
                [
                    [2.0, 3.0, 4.0, 3.0],
                    [5.0, 6.0, 7.0, 5.0],
                    [8.0, 9.0, 10.0, 8.0],
                ]
            ], dtype=torch.float32
        ) # (batch_size=2, sequence_length=3, dim_features=4)

    def test_dim_features(self):
        """This function checks that the third dimension of the input
        tensor (i.e., the number of features per particle) matches
        the dim_features parameter of the model.
        """
        self.assertEqual(
            self.input.shape[2],
            self.dim_features,
            f"Mismatch in feature dimensions: expected {self.dim_features}, "
            f"got {self.input.shape[2]}"
        )

    def test_forward_output_shape(self):
        """This function tests that the forward method of the model
        produces an output tensor with the correct shape. The shape of
        the output tensor must match with the shape of the input tensor.
        """
        output = self.transformer.forward(self.input, self.input)
        self.assertEqual(output.shape, self.input.shape)

    def test_forward_output_nans_infs(self):
        """This function tests that the putput of the forward methods does
        not contain Inf values or NaN values.
        """
        output = self.transformer.forward(self.input, self.input)
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
        input_proj= self.transformer.input_projection(self.input)
        output_proj= self.transformer.output_projection(input_proj)
        self.assertEqual(input_proj.shape, (2, 3, self.num_units))
        self.assertEqual(output_proj.shape, (2, 3, self.dim_features))

if __name__ == '__main__':
    unittest.main()
