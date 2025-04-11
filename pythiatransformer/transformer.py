"""
Transformer class.
"""
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import TensorDataset, DataLoader

# from data_processing import input_train, input_test, input_val
# from data_processing import target_train, target_test, target_val

class ParticleTransformer(nn.Module):
    """Transformer taking in input particles having status 23
    (i.e. outgoing particles of the hardest subprocess)
    and as target the final particles of the event.
    """
    def __init__(
        self,
        input_train,
        input_val,
        input_test,
        target_train,
        target_val,
        target_test,
        dim_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        num_units,
        dropout,
        batch_size,
        activation
    ):
        """
        Args:  
            input_train (torch.Tensor): Input tensor representing
                                        status 23 particles used during
                                        the model training.
            input_val (torch.Tensor): Input tensor representing status
                                      23 particles used during the
                                      model validation.
            input_test (torch.Tensor): Input tensor representing status
                                       23 particles used during the
                                       model test.
            target_train (torch.Tensor): Target tensor representing
                                         stable particles used during
                                         the model training.
            target_val (torch.Tensor): Target tensor representing
                                       stable particles used during the
                                       model validation.
            target_test (torch.Tensor): Target tensor representing
                                        stable particles used during
                                        the model test.
            dim_features (int): number of features of each particle
                                (px, py, pz, E, M, ID).
            num_heads (int): heads number of the attention system.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            num_units (int): number of units of each hidden layer.
            dropout (float): probability of each neuron to be
                             switched off.
            batch_size (int): 
            activation (nn.function): activation function of encoder
                                      and/or decoder layers.
        """
        # da aggiungere le varie exception per controllare i tipi
        super(ParticleTransformer, self).__init__()
        self.input_train = input_train
        self.input_val = input_val
        self.input_test = input_test
        self.target_train = target_train
        self.target_val = target_val
        self.target_test = target_test
        self.dim_features = dim_features
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_units = num_units
        self.dropout = dropout
        self.batch_size = batch_size
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
        self.train_data, self.val_data, self.test_data = self.data_processing()

    def build_projection_layer(self):
        """This function transforms input and output data into a
        representation more suitable for a Transformers. It utilizes an
        nn.Linear layer, which applies a linear transformation.
        To obtain a more abstract representation of the data, the
        number of hidden units is chosen to be greater than the number
        of input features. Subsequently, a linear trasformation is
        applied to restore the data to its original dimensions.
        """
        self.input_projection = nn.Linear(self.dim_features, self.num_units)
        self.output_projection = nn.Linear(self.num_units, self.dim_features)
        logging.debug("Projection layers (input/output) created.")

    def initialize_transformer(self):
        """This function initializes the transformer with the specified
        configuration parameters. 
        """
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

    def data_processing(self):
        """This function prepares the data for training by splitting it
        into batches and shuffling the training data.

        Returns:
            training_loader (Iterator): An iterator for the training
                                        data, with batching and
                                        shuffling enabled.
            validation_loader (Iterator): An iterator for the
                                          validation data, with
                                          batching and shuffling
                                          enabled.
            test_loader (Iterator): An iterator for the test data, with
                                    batching and shuffling enabled.
        """
        training_set = TensorDataset(self.input_train, self.target_train)
        validation_set = TensorDataset(self.input_val, self.target_val)
        test_set = TensorDataset(self.input_test, self.target_test)

        training_loader = DataLoader(
            training_set,
            self.batch_size,
            shuffle = True
        )
        validation_loader = DataLoader(validation_set, self.batch_size)
        test_loader = DataLoader(test_set, self.batch_size)

        return training_loader, validation_loader, test_loader

    def forward(self, input, target):
        """The aim of this function is computed the output of the model by
        projecting the input and the target into an hidden
        representation space, processing them through a Transformer, 
        and projecting the result back to the original feature space.

        Args:
            input (torch.Tensor): Input tensor representing status 23
                                  particles.
            target (torch.Tensor): Input tensor representing stable
                                   particles.
        
        Returns:
            output (torch.Tensor): The output tensor after processing
                                   through the model.
        """
        input = self.input_projection(input)
        target = self.input_projection(target)
        output = self.transformer(input, target)
        output = self.output_projection(output)
        return output
    
    def train_one_epoch(self, epoch, loss_func, optim):
        """This function trains the model for one epoch. It iterates
        through the training data, computes the model's output and the
        loss, performs backpropagation and updates the model parameters
        provided optimizer.

        Args:
            epoch (int): current epoch.
            loss_func (nn.Module): Loss function used to compute the
                                   loss.
            optim (torch.optim.optimizer): Optimizer used for updating
                                           model parameters.
        """
        self.train()
        loss_epoch = 0
        for inputs, targets in self.train_data:
            optim.zero_grad()
            outputs = self.forward(inputs, targets)
            loss = loss_func(outputs, targets)
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        logging.info(f"Loss at epoch {epoch + 1}: {loss_epoch}")

    def val_one_epoch(self, epoch, loss_func):
        """This function validates the model for one epoch. It iterates
        through the validation data, computes the model's output and
        the loss.

        Args:
            epoch (int): current epoch.
            loss_func (nn.Module): Loss function used to compute the
                                   loss.
        """
        self.eval()
        loss_epoch = 0
        with torch.no_grad():
            for inputs, targets in self.val_data:
                outputs = self.forward(inputs, targets)
                loss = loss_func(outputs, targets)
                loss_epoch += loss.item()
        logging.info(f"validation loss at epoch {epoch + 1}: {loss_epoch}")

    def train_val(self, num_epochs, loss_func, optim):
        """This function trains and validates the model for the given
        number of epochs.
        Args:
            num_epochs (int): Number of total epochs.
            loss_func (nn.Module): Loss function used to compute the
                                   loss.
            optim (torch.optim.optimizer): Optimizer used for updating
                                           model parameters.
        """
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch, loss_func, optim)
            self.val_one_epoch(epoch, loss_func)

    def test(self, loss_func):
        """This function computes the total loss of the model on the
        test data without updating the model's parameters.
        Args:
            loss_func (nn.Module): Loss function used to compute the
                                   loss. 
        """
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_data:
                outputs = self(inputs, targets)
                loss = loss_func(outputs, targets)
                total_loss += loss.item()
        logging.info(f"Test loss: {total_loss}")
