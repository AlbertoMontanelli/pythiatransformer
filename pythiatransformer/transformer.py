"""
A custum transformer model designed to predict the final stable
particles starting from the status 23 particles.
"""

import gc

from loguru import logger
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def log_peak_memory(epoch=None):
    """
    Print the maximum GPU memory allocated during the current epoch and
    reset the counter.
    """
    peak_MB = torch.cuda.max_memory_allocated() / 1024**2
    prefix = f"[Epoch: {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix} Max memory allocated: {peak_MB:.2f} MB")
    torch.cuda.reset_peak_memory_stats()


def log_gpu_memory(epoch=None):
    """
    Print a clear summary of GPU memory usage.
    """
    alloc_MB = torch.cuda.memory_allocated() / 1024**2
    reserved_MB = torch.cuda.memory_reserved() / 1024**2
    stats = torch.cuda.memory_stats()
    total_alloc_MB = stats["allocation.all.allocated"] / 1024**2

    prefix = f"[Epoca {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix}Memory allocated: {alloc_MB:.2f} MB")
    print(f"{prefix}Memory reserved: {reserved_MB:.2f} MB")
    print(
        f"{prefix}Total memory allocated througth epochs:"
        f" {total_alloc_MB:.2f} MB"
    )

def _check_type(var, name, t):
    if not isinstance(var, t):
        raise TypeError(f"{name} must be of type {t.__name__}, got {type(var).__name__}")

import matplotlib.pyplot as plt


class ParticleTransformer(nn.Module):
    """Transformer taking in input particles having status 23
    (i.e. outgoing particles of the hardest subprocess) and as target
    the final particles of the event.
    """

    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        train_data_pad_mask,
        val_data_pad_mask,
        test_data_pad_mask,
        dim_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        num_units,
        dropout,
        activation,
    ):
        """
        Args:
            train_data (DataLoader): Dataloader containing the training
                                     input-target pairs.
            val_data (DataLoader): Dataloader containing the validation
                                   input-target pairs.
            test_data (DataLoader): Dataloader containing the test
                                    input-target pairs.
            train_data_pad_mask (DataLoader): Dataloader containing the
                                              padding masks for the
                                              training set used to mask
                                              padded values during
                                              attention computations.
            val_data_pad_mask (DataLoader): Dataloader containing the
                                            padding masks for the
                                            validation set.
            test_data_pad_mask (DataLoader): Dataloader containing the
                                             padding masks for the test
                                             set.
            dim_features (int): Number of features of each particle.
            num_heads (int): Heads number of the attention system.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            num_units (int): Number of units of each hidden layer. 
                             To obtain a more abstract representation
                             of the data, this number is chosen to be
                             greater than the number of input features.
            dropout (float): Probability of each neuron to be
                             switched off.
            activation (nn.function): Activation function of encoder
                                      and/or decoder layers.
        """
        super(ParticleTransformer, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_pad_mask = train_data_pad_mask
        self.val_data_pad_mask = val_data_pad_mask
        self.test_data_pad_mask = test_data_pad_mask
        self.num_encoder_layers = num_encoder_layers

        self.dim_features = dim_features
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_units = num_units
        self.dropout = dropout
        self.activation = activation

        _check_type(dim_features, "dim_features", int)
        _check_type(num_heads, "num_heads", int)
        _check_type(num_encoder_layers, "num_encoder_layers", int)
        _check_type(num_decoder_layers, "num_decoder_layers", int)
        _check_type(num_units, "num_units", int)
        _check_type(dropout, "dropout", float)

        if not (num_units % num_heads == 0):
            raise ValueError(
                "Number of units must be a multiple of number of heads."
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("Dropout must be between 0.0 and 1.0")

        # Loss functions used during the training
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.build_projection_layer()
        self.initialize_transformer()

        self.device = next(self.parameters()).device
        logger.info(f"Model initialized on device: {self.device}")

    def build_projection_layer(self):
        """Initializes the layers responible for projecting input
        features to the model's hidden dimentionality and outputs back
        to the original feature space.
        """
        self.input_projection = nn.Linear(self.dim_features, self.num_units)
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.num_units))
        self.particle_head = nn.Linear(self.num_units, self.dim_features)
        # The EOS head returns one value per feature.
        self.eos_head = nn.Linear(self.num_units, self.dim_features)
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
            batch_first = True,
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

    def forward(self, input, target, enc_input_mask, dec_input_mask):
        """Computes the output of the model by projecting both input
        and target into an hidden representation space, processing them
        through a Transformer, and projecting the result back to the
        original feature space.

        Args:
            input (torch.Tensor): Input tensor representing status 23
                                  particles.
            target (torch.Tensor): Input tensor representing stable
                                   particles.
            enc_input_mask (torch.Tensor): Padding mask corresponding
                                           to input.
            dec_input_mask (torch.Tensor): Padding mask corresponding
                                           to input.
        Returns:
            pred (torch.Tensor): The output tensor after processing
                                 through the model.
            eos_prob_vector (torch.Tensor): Tensor representing the
                                            end-of-sequence probabilities
                                            for each step. 
        """
        batch_size = input.size(0)
        # Input projected
        enc_input = self.input_projection(input)
        # SOS projected
        sos = self.sos_token.expand(batch_size, -1, -1)
        # Decoder input without sos
        dec_input = self.input_projection(target)
        # Decoder input with sos
        dec_input = torch.cat([sos, dec_input], dim=1)
        # Encoder and decoder input padding mask
        sos_mask = torch.zeros(batch_size, 1).to(self.device)
        dec_input_mask = torch.cat([sos_mask, dec_input_mask], dim=1)

        attention_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_input.size(1)
        ).to(self.device)
        output = self.transformer(
            src=enc_input,
            tgt=dec_input,
            tgt_mask=attention_mask,
            src_key_padding_mask=enc_input_mask,
            tgt_key_padding_mask=dec_input_mask,
        )
        pred = self.particle_head(output).squeeze(-1)
        eos_prob_vector = self.eos_head(output).squeeze(-1)
        return pred, eos_prob_vector

    def train_one_epoch(self, epoch, optim):
        """This function trains the model for one epoch. It iterates
        through the training data, computes the model's output and the
        loss, performs backpropagation and updates the model parameters
        provided optimizer.

        Args:
            epoch (int): current epoch.
            optim (torch.optim.optimizer): Optimizer used for updating
                                           model parameters.
        Returns:
            loss_epoch (float): Loss computed in the current epoch.
        """
        self.train()
        loss_epoch = 0

        for (enc_input, dec_input), (
            enc_input_padding_mask,
            dec_input_padding_mask,
        ) in zip(self.train_data, self.train_data_pad_mask):

            enc_input = enc_input.to(self.device)
            dec_input = dec_input.to(self.device)
            enc_input_padding_mask = enc_input_padding_mask.to(self.device)
            dec_input_padding_mask = dec_input_padding_mask.to(self.device)

            inverse_dec_input_padding_mask = ~dec_input_padding_mask
            eos_tensor = torch.zeros(
                dec_input.size(0), dec_input.size(1) + 1, dec_input.size(2)
            ).to(self.device)
            for event in range(dec_input.size(0)):
                len_event = inverse_dec_input_padding_mask[event].sum()
                eos_tensor[event, len_event] = 1

            optim.zero_grad()
            output, eos_prob_vector = self.forward(
                enc_input,
                dec_input,
                enc_input_padding_mask,
                dec_input_padding_mask,
            )
            mse = self.mse_loss(
                output[:, :-1] * inverse_dec_input_padding_mask.float(),
                dec_input.squeeze(-1) * inverse_dec_input_padding_mask.float(),
            )
            bce = self.bce_loss(eos_prob_vector, eos_tensor.squeeze(-1))
            loss = mse + bce

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite at epoch {epoch + 1}")

            loss.backward()
            optim.step()
            loss_epoch += loss.item()

            del (
                enc_input,
                dec_input,
                enc_input_padding_mask,
                dec_input_padding_mask,
                output,
                eos_prob_vector,
                eos_tensor,
                loss,
                mse,
                bce,
                inverse_dec_input_padding_mask,
            )
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        loss_epoch = loss_epoch / len(self.train_data)
        logger.debug(f"Training loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        return loss_epoch

    def val_one_epoch(self, epoch):
        """This function validates the model for one epoch. It iterates
        through the validation data, computes the model's output and
        the loss.

        Args:
            epoch (int): current epoch.
        Returns:
            loss_epoch (float): Loss computed in the current epoch.
        """
        self.eval()
        loss_epoch = 0
        with torch.no_grad():
            for (enc_input, dec_input), (
                enc_input_padding_mask,
                dec_input_padding_mask,
            ) in zip(self.val_data, self.val_data_pad_mask):

                enc_input = enc_input.to(self.device)
                dec_input = dec_input.to(self.device)
                enc_input_padding_mask = enc_input_padding_mask.to(self.device)
                dec_input_padding_mask = dec_input_padding_mask.to(self.device)

                inverse_dec_input_padding_mask = ~dec_input_padding_mask

                eos_tensor = torch.zeros(
                    dec_input.size(0), dec_input.size(1) + 1, dec_input.size(2)
                ).to(self.device)
                for event in range(dec_input.size(0)):
                    len_event = inverse_dec_input_padding_mask[event].sum()
                    eos_tensor[event, len_event] = 1

                output, eos_prob_vector = self.forward(
                    enc_input,
                    dec_input,
                    enc_input_padding_mask,
                    dec_input_padding_mask,
                )

                mse = self.mse_loss(
                    output[:, :-1] * inverse_dec_input_padding_mask.float(),
                    dec_input.squeeze(-1) * inverse_dec_input_padding_mask.float(),
                )

                bce = self.bce_loss(eos_prob_vector, eos_tensor.squeeze(-1))
                loss = mse + bce

                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Loss is not finite at epoch {epoch + 1}"
                    )
                loss_epoch += loss.item()

                del (
                    enc_input,
                    dec_input,
                    enc_input_padding_mask,
                    dec_input_padding_mask,
                    output,
                    eos_prob_vector,
                    eos_tensor,
                    loss,
                    mse,
                    bce,
                    inverse_dec_input_padding_mask,
                )
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        loss_epoch = loss_epoch / len(self.val_data)
        
        logger.debug(
            f"Validation loss at epoch {epoch + 1}: {loss_epoch:.4f}"
        )
        return loss_epoch

    def train_val(
        self,
        num_epochs,
        optim,
        patient_early=10,
    ):
        """Trains and validates the model for the given number of
        epochs, using train_one_epoch and loss_one_epoch. Implements
        early stopping if validation loss does not improve.

        Args:
            num_epochs (int): Number of total epochs.
            optim (torch.optim.optimizer): Optimizer used for updating
                                           model parameters.
            patient (int): Number of consecutive epochs to wait for an
                           improvement in validation loss before
                           stopping early. Default 10.
        Returns:
            train_loss (list): Training loss recorded at each epoch.
            val_loss (list): Validation loss recorded at each epoch.
        """
        _check_type(num_epochs, "num_epochs", int)
        _check_type(patient_early, "patient_early", int)

        if not (patient_early < num_epochs):
            raise ValueError(
                f"Patient must be smaller than the number of epochs."
            )

        train_loss = []
        val_loss = []

        counter_earlystop = 0

        logger.info("Training started!")
        for epoch in range(num_epochs):
            torch.cuda.reset_peak_memory_stats()
            train_loss_epoch = self.train_one_epoch(epoch, optim)
            val_loss_epoch = self.val_one_epoch(epoch)
            train_loss.append(train_loss_epoch)
            val_loss.append(val_loss_epoch)

            if epoch >= int(num_epochs / 10):
                stop_early = self.early_stopping(val_loss, epoch)
                if stop_early:
                    counter_earlystop += 1
                else:
                    counter_earlystop = 0
                if counter_earlystop >= patient_early:
                    logger.warning(
                        f"Overfitting at epoch {epoch + 1 - patient_early}."
                    )
                    train_loss = train_loss[:epoch + 1 - patient_early]
                    val_loss = val_loss[:epoch + 1 - patient_early]
                    break
                
            torch.cuda.empty_cache()
            log_gpu_memory(epoch)
            log_peak_memory(epoch)

        logger.info("Training completed!")
        return train_loss, val_loss

    def early_stopping(self, val_losses, current_epoch):
        """Checks whether the validation loss has increased or remained
        the same compared to the previous epoch.

        Args:
            val_losses (list): List containing validation losses for
                               each epoch before the current.
            current_epoch (int): Index of the current epoch.
        Returns:
            stop (bool): True if validation loss has no improved, False
                         otherwise.
        """
        if val_losses[current_epoch - 1] <= val_losses[current_epoch]:
            stop = True
        else:
            stop = False
        return stop

    def generate_targets(self, stop_threshold=0.5):
        """Performs autoregressive generation of stable particles'
        energy using the test set. Computes the difference between
        targets and predictions in order to plots an histogram.

        Args:
            stop_threshold (float): Threshold for the EOS probability
                                    to stop generation. Default is 0.5.
        Returns:
            diff (list): List containing the difference between target
                         and prediction
        """
        diff = []
        nbatch = 0
        for (input, target), (
            input_padding_mask,
            target_padding_mask,
        ) in zip(self.test_data, self.test_data_pad_mask):
            nbatch = nbatch + 1
            input = input.to(self.device)
            target = target.to(self.device)
            input_padding_mask = input_padding_mask.to(self.device)
            target_padding_mask = target_padding_mask.to(self.device)
            batch_size = input.size(0)
            max_len = target.size(1)

            enc_input = self.input_projection(input)
            dec_input = [
                self.sos_token.clone().squeeze(0) for _ in range(batch_size)
            ]
            generated = torch.zeros(batch_size, max_len)

            for event in range(batch_size):
                for t in range(max_len):
                    attention_mask = (
                        nn.Transformer.generate_square_subsequent_mask(t + 1).to(
                            self.device
                        )
                    )
                    output = self.transformer(
                        src=enc_input[event],
                        tgt=dec_input[event],
                        tgt_mask=attention_mask,
                    )
                    last_token = output[-1, :]
                    proj_token = self.particle_head(last_token)
                    eos = torch.sigmoid(self.eos_head(last_token))
                    generated[event, t] = proj_token
                    next_input = self.input_projection(proj_token).unsqueeze(0)
                    dec_input[event] = torch.cat(
                        [dec_input[event], next_input], dim=0
                    )
                    if eos > stop_threshold:
                        break
            # Compute the difference from target to predicted
            for i in range(batch_size):
                target_sum = target[i].sum().item()
                pred_sum = generated[i].sum().item()
                diff.append(target_sum - pred_sum)
            logger.info(f"Generated batch n: {nbatch}")

        return diff
