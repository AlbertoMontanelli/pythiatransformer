"""
Transformer class.
"""
import argparse
import logging
import torch.nn as nn
import torch.optim as optimizer

#from data_processing import inputs_tensor, outputs_tensor


class ParticleTransformer(nn.Module):
    '''
    Transformer taking in input particles having status 23
    (i.e. outgoing particles of the hardest subprocess)
    and as target the final particles of the event.
    '''
    def __init__(self,
                 dim_features,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 num_units,
                 dropout,
                 activation = nn.ReLU()
                ):
        '''
        Args: 
            dim_features (int): number of features of each particle
                                (px, py, pz, E, M, ID).
            num_heads (int): heads number of the attention system.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            num_units (int): number of units of each hidden layer.
            dropout (float): probability of each neuron to be
                             switched off.
            activation (string): activation function of encoder
                                 and/or decoder layers.
        '''

        super(ParticleTransformer, self).__init__()
        self.dim_features = dim_features
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_units = num_units
        self.dropout = dropout
        self.activation = activation

        logging.info(
            f"Initialized ParticleTransformer with "
            f"dim_features={dim_features}, "
            f"num_heads={num_heads}, "
            f"num_encoder_layers={num_encoder_layers}, "
            f"num_decoder_layers={num_decoder_layers}, "
            f"num_units={num_units}, "
            f"dropout={dropout}, activation={activation}"
        )

        self.build_projection_layer()
        self.initialize_transformer()

    def build_projection_layer(self):
        '''
        This function transforms input and output data into a
        representation more suitable for a Transformers. It utilizes an
        nn.Linear layer, which applies a linear transformation.
        To obtain a more abstract representation of the data, the
        number of hidden units is chosen to be greater than the number
        of input features. Subsequently, a linear trasformation is
        applied to restore the data to its original dimensions.
        '''
        self.input_projection = nn.Linear(self.dim_features, self.num_units)
        self.output_projection = nn.Linear(self.num_units, self.dim_features)
        logging.debug("Projection layers (input/output) created.")

    def initialize_transformer(self):
        '''
        This function initializes the transformer with the specified
        configuration parameters. 
        '''
        self.transformer = nn.Transformer(
            d_model = self.num_units,
            nhead = self.num_heads,
            num_encoder_layers = self.num_encoder_layers,
            num_decoder_layers = self.num_decoder_layers,
            dim_feedforward = 4 * self.num_units,
            dropout = self.dropout,
            activation = self.activation,
            batch_first=True
        )

        logging.debug(
            f"Transformer initialized with "
            f"d_model={self.num_units}, nhead={self.num_heads}, "
            f"num_encoder_layers={self.num_encoder_layers}, "
            f"num_decoder_layers={self.num_decoder_layers}"
        )

    def forward(self, source, target):
        '''
        The aim of this function is computed the output of the model by
        projecting the input and the target into an hidden
        representation space, processing them through a Transformer, 
        and projecting the result back to the original feature space.

        Args:
            source (torch.Tensor): Input tensor representing status 23
                                   particles.
            target (torch.Tensor): Input tensor representing stable
                                   particles.
        
        Returns:
            output (torch.Tensor): The output tensor after processing
                                   through the model.
        '''
        source = self.input_projection(source)
        target = self.input_projection(target)
        output = self.transformer(source, target)
        output = self.output_projection(output)
        return output
    