"""
Transformer class.
"""
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
        super(ParticleTransformer, self).__init__()
        self.input_train = input_train
        self.input_val = input_val
        self.input_test = input_test
        self.target_train = target_train
        self.target_val = target_val
        self.target_test = target_test

        self.dim_features = dim_features
        if not isinstance(dim_features, int):
            raise TypeError(
                f"The number of features must be int, got {type(dim_features)}"
            )

        self.num_heads = num_heads
        if not isinstance(num_heads, int):
            raise TypeError(
                f"The number of heads must be int, got {type(num_heads)}"
            )

        self.num_encoder_layers = num_encoder_layers
        if not isinstance(num_encoder_layers, int):
            raise TypeError(
                f"The number of encoder layers must be int, "
                f"got {type(num_encoder_layers)}"
            )

        self.num_decoder_layers = num_decoder_layers
        if not isinstance(num_decoder_layers, int):
            raise TypeError(
                f"The number of decoder layers must be int, "
                f"got {type(num_decoder_layers)}"
            )

        self.num_units = num_units
        if not isinstance(num_units, int):
            raise TypeError(
                f"The number of unit must be int, got {type(num_units)}"
            )

        self.dropout = dropout
        if not isinstance(dropout, float):
            raise TypeError(f"dropout must be float, got {type(dropout)}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        self.batch_size = batch_size
        if not isinstance(batch_size, int):
            raise TypeError(
                f"Batch size must be int, got {type(batch_size)}"
            )
        if not (batch_size <= input_train.shape[0]):
            raise ValueError(
                f"Batch size must be smaller than the input dataset size."
            )

        self.activation = activation

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
        logger.info("Projection layers input/output created.")

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
            batch_first = True
        )

        logger.info(
            f"Transformer initialized with "
            f"number of hidden units: {self.num_units}, "
            f"number of heads: {self.num_heads}, "
            f"number of encoder layers: {self.num_encoder_layers}, "
            f"number of decoder layers: {self.num_decoder_layers}, "
            f"number of feedforward units: {4 * self.num_units}, "
            f"dropout probability: {self.dropout}, "
            f"activation function: {self.activation}."
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

        logger.info("Data preprocessed.")

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
        Returns:
            loss_epoch (float):
        """
        self.train()
        loss_epoch = 0
        for inputs, targets in self.train_data:
            optim.zero_grad()
            outputs = self.forward(inputs, targets)
            loss = loss_func(outputs, targets)
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Loss is not finite at epoch {epoch + 1}"
                )
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        logger.debug(f"Loss at epoch {epoch + 1}: {loss_epoch}")
        return loss_epoch

    def val_one_epoch(self, epoch, loss_func, val):
        """This function validates the model for one epoch. It iterates
        through the validation data, computes the model's output and
        the loss.

        Args:
            epoch (int): current epoch.
            loss_func (nn.Module): Loss function used to compute the
                                   loss.
            val (bool): Default True. Set to False when using the test
                        set
        Returns:
            loss_epoch (float):
        """
        if val:
            data_loader = self.val_data
        else: 
            data_loader = self.test_data
        self.eval()
        loss_epoch = 0
        with torch.no_grad(): # Compute only the loss value
            for inputs, targets in data_loader:
                outputs = self.forward(inputs, targets)
                loss = loss_func(outputs, targets)
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Loss is not finite at epoch {epoch + 1}"
                    )
                loss_epoch += loss.item()
        if val:
            logger.debug(f"Validation loss at epoch {epoch + 1}: {loss_epoch}")
        else:
            logger.debug(f"Test loss at epoch {epoch + 1}: {loss_epoch}")
        return loss_epoch

    def train_val(self, num_epochs, loss_func, optim, val = True, patient = 10):
        """This function trains and validates the model for the given
        number of epochs.
        Args:
            num_epochs (int): Number of total epochs.
            loss_func (nn.Module): Loss function used to compute the
                                   loss.
            optim (torch.optim.optimizer): Optimizer used for updating
                                           model parameters.
            patient (int):
        Returns:
            train_loss (list):
            val_loss (list):
        """
        if not isinstance(num_epochs, int):
            raise TypeError(
                f"The number of epoch must be int, got {type(num_epochs)}"
            )
        if not isinstance(patient, int):
            raise TypeError(
                f"The patien must be int, got {type(patient)}"
            )
        if not (patient < num_epochs):
            raise ValueError(
                f"Patient must be smaller than the number of epochs."
            )
        train_loss = []
        val_loss = []
        counter = 0
        logger.info("Training started!")
        for epoch in range(num_epochs):
            train_loss.append(self.train_one_epoch(epoch, loss_func, optim))
            val_loss.append(self.val_one_epoch(epoch, loss_func, val))
            # if epoch > patient and val_loss[-1] > val_loss[-2] and train_loss[-1] < train_loss[-2]:
            #     logger.warning(f"Possible overfitting at epoch {epoch + 1}.")
            if epoch > num_epochs/10:
                stop = self.early_stopping(val_loss, epoch)
                if stop:
                    counter += 1
                else:
                    counter = 0
                if counter >= patient:
                    logger.warning(f"Early stopping at epoch {epoch + 1}.")
                    break
        logger.info("Training completed!")
        return train_loss, val_loss

    def early_stopping(self, val_loss, current_epoch):
        """
        Args:
            val_loss (list):
            current_epoch (int):
        Returns:
            stop (bool):
        """
        stop = False
        if val_loss[current_epoch-1] < val_loss[current_epoch]:
            stop = True
        return stop
