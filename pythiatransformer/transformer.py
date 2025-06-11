"""
Transformer class.
"""

import gc

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pad_sequence


def log_peak_memory(epoch=None):
    """
    Stampa il picco massimo di memoria GPU allocata durante l'epoca corrente
    e resetta il contatore.
    """
    peak_MB = torch.cuda.max_memory_allocated() / 1024**2
    prefix = f"[Epoca {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix} Picco memoria allocata: {peak_MB:.2f} MB")
    torch.cuda.reset_peak_memory_stats()


def log_gpu_memory(epoch=None):
    """
    Stampa un riepilogo chiaro della memoria GPU.
    """
    alloc_MB = torch.cuda.memory_allocated() / 1024**2
    reserved_MB = torch.cuda.memory_reserved() / 1024**2
    stats = torch.cuda.memory_stats()
    total_alloc_MB = stats["allocation.all.allocated"] / 1024**2

    prefix = f"[Epoca {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix}Memoria allocata:   {alloc_MB:.2f} MB")
    print(f"{prefix}Memoria riservata:   {reserved_MB:.2f} MB")
    print(f"{prefix}Totale allocata (storico): {total_alloc_MB:.2f} MB")


class ParticleTransformer(nn.Module):
    """Transformer taking in input particles having status 23
    (i.e. outgoing particles of the hardest subprocess)
    and as target the final particles of the event.
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
        num_classes,
        dropout,
        activation,
    ):
        """
        Args:
            train_data (DataLoader):
            val_data (DataLoader):
            test_data (DataLoader):
            attention_train_data (DataLoader):
            attention_val_data (DataLoader):
            attention_test_data (DataLoader):
            dim_features (int): number of features of each particle
                                (px, py, pz, E, M, ID).
            num_heads (int): heads number of the attention system.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            num_units (int): number of units of each hidden layer.
            dropout (float): probability of each neuron to be
                             switched off.
            activation (nn.function): activation function of encoder
                                      and/or decoder layers.
        """
        super(ParticleTransformer, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_pad_mask = train_data_pad_mask
        self.val_data_pad_mask = val_data_pad_mask
        self.test_data_pad_mask = test_data_pad_mask

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

        self.num_classes = num_classes
        if not isinstance(num_classes, int):
            raise TypeError(
                f"The number of unit must be int, got {type(num_classes)}"
            )

        self.dropout = dropout
        if not isinstance(dropout, float):
            raise TypeError(f"dropout must be float, got {type(dropout)}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        if not (num_units % num_heads == 0):
            raise ValueError(
                "Number of units must be a multiple of number of heads."
            )

        self.activation = activation

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.build_projection_layer()
        self.initialize_transformer()

        self.device = next(self.parameters()).device
        print(self.device)

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

        # 1 sta per gli eventi, poi diventa lungo tanto quanto è il batch size
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.num_units))

        self.particle_head = nn.Linear(self.num_units, self.dim_features)

        # se poi ho più di una features ho tante teste di eos quante le features?
        self.eos_head = nn.Linear(self.num_units, self.dim_features)
        logger.info("Projection layers input/output created.")

    def initialize_transformer(self):
        """This function initializes the transformer with the specified
        configuration parameters.
        """
        self.transformer = nn.Transformer(
            d_model=self.num_units,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=4 * self.num_units,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
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

    def de_padding(self, input, padding_mask):
        """Function that eliminate extra padding for each batch"""
        non_pad_mask = ~padding_mask  # inverto la matrice di padding
        num_particles = non_pad_mask.sum(
            dim=1
        )  # tensore di lunghezza batch_size, con valori il numero di particelle vere
        max_len_tensor = num_particles.max()
        max_len = max_len_tensor.item()
        input = input[:, :max_len, :]
        padding_mask = padding_mask[:, :max_len]
        return input, padding_mask

    def forward(self, input, target, enc_input_mask, dec_input_mask):
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
        batch_size = input.size(0)
        # print(f"shape encoder input non proiettato {input.shape}")
        enc_input = self.input_projection(input)
        # print(f"shape encoder input proiettato {enc_input.shape}")
        sos = self.sos_token.expand(batch_size, -1, -1)
        # print(f"sos shape proiettata {sos.shape}")
        dec_input = self.input_projection(target)
        # print(f"decoder input senza sos {dec_input.shape}")
        dec_input = torch.cat([sos, dec_input], dim=1)
        # print(f"decoder input con sos {dec_input.shape}")
        sos_mask = torch.zeros(batch_size, 1).to(self.device)
        dec_input_mask = torch.cat([sos_mask, dec_input_mask], dim=1)
        # print(f"encoder input mask {enc_input_mask.shape}")
        # print(f"decoder input mask con sos {dec_input_mask.shape}")
        attention_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_input.size(1)
        ).to(self.device)
        # input = input.squeeze(-2)
        # target = target.squeeze(-2)
        output = self.transformer(
            src=enc_input,
            tgt=dec_input,
            tgt_mask=attention_mask,
            src_key_padding_mask=enc_input_mask,
            tgt_key_padding_mask=dec_input_mask,
        )
        # print(f"output shape: {output.shape}")
        pred = self.particle_head(output).squeeze(-1)
        # print(f"pred shape: {pred.shape}")
        eos_prob_vector = self.eos_head(output).squeeze(-1)
        # print(f"shape eos {eos_prob_vector.shape}")
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
            loss_epoch (float):
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
            """target, target_padding_mask = self.de_padding(
                target, target_padding_mask
            )"""

            """
            # Lista per decoder_input e mask
            decoder_input_list = []
            decoder_input_mask_list = []

            # Per ogni evento nel batch, rimuoviamo l'ultimo elemento dall'input del
            # decoder e il primo dal target per la loss
            for event in range(target.shape[0]):
                event_target = target[event]
                event_mask = target_padding_mask[event]

                decoder_input_event = event_target[:-1]
                decoder_input_event_mask = event_mask[:-1]
                decoder_input_list.append(decoder_input_event)
                decoder_input_mask_list.append(decoder_input_event_mask)

            decoder_input = torch.stack(decoder_input_list, dim=0)
            decoder_input_padding_mask = torch.stack(
                decoder_input_mask_list, dim=0
            )

            target_4_loss = target[:, 1:, :]
            """
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
            # print(f"output shape: {output[:, 1:].shape}, mask {inverse_dec_input_padding_mask.float().shape}, dec input {dec_input.shape}")
            mse = self.mse_loss(
                output[:, 1:] * inverse_dec_input_padding_mask.float(),
                dec_input.squeeze(-1),
            )

            # print(f"eos_prob {eos_prob_vector.shape}, eos_tensor: {eos_tensor.shape}")
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

        gc.collect()  # forcing garbage collector
        torch.cuda.empty_cache()

        loss_epoch = loss_epoch / len(self.train_data)
        logger.debug(f"Training loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        return loss_epoch

    def val_one_epoch(self, epoch, val):
        """This function validates the model for one epoch. It iterates
        through the validation data, computes the model's output and
        the loss.

        Args:
            epoch (int): current epoch.
            val (bool): Default True. Set to False when using the test
                        set
        Returns:
            loss_epoch (float):
        """
        if val:
            data_loader = self.val_data
            mask_loader = self.val_data_pad_mask
        else:
            data_loader = self.test_data
            mask_loader = self.test_data_pad_mask

        self.eval()
        loss_epoch = 0
        with torch.no_grad():
            for (enc_input, dec_input), (
                enc_input_padding_mask,
                dec_input_padding_mask,
            ) in zip(data_loader, mask_loader):

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
                    output[:, 1:] * inverse_dec_input_padding_mask.float(),
                    dec_input.squeeze(-1),
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

        gc.collect()  # forcing garbage collector
        torch.cuda.empty_cache()

        loss_epoch = loss_epoch / len(self.train_data)
        if val:
            logger.debug(
                f"Validation loss at epoch {epoch + 1}: {loss_epoch:.4f}"
            )
        else:
            logger.debug(f"Test loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        return loss_epoch

    def train_val(
        self,
        num_epochs,
        optim,
        val=True,
        patient_early=10,
    ):
        """This function trains and validates the model for the given
        number of epochs.
        Args:
            num_epochs (int): Number of total epochs.
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
        if not isinstance(patient_early, int):
            raise TypeError(
                f"The patien must be int, got {type(patient_early)}"
            )
        if not (patient_early < num_epochs):
            raise ValueError(
                f"Patient must be smaller than the number of epochs."
            )

        train_loss = []
        val_loss = []

        # early_stop initialization
        counter_earlystop = 0

        logger.info("Training started!")
        for epoch in range(num_epochs):
            torch.cuda.reset_peak_memory_stats()
            train_loss_epoch = self.train_one_epoch(epoch, optim)
            val_loss_epoch = self.val_one_epoch(epoch, val)
            train_loss.append(train_loss_epoch)
            val_loss.append(val_loss_epoch)

            if epoch >= int(num_epochs / 100):
                # early_stopping check
                stop_early = self.early_stopping(val_loss, epoch)
                if stop_early:
                    counter_earlystop += 1
                else:
                    counter_earlystop = 0
                logger.info(f"stop early: {stop_early}")
                if counter_earlystop >= patient_early:
                    logger.warning(
                        f"Overfitting at epoch {epoch + 1 - patient_early}."
                    )
                    break

            torch.cuda.empty_cache()
            log_gpu_memory(epoch)
            log_peak_memory(epoch)

        logger.info("Training completed!")
        return train_loss, val_loss

    def early_stopping(self, val_losses, current_epoch):
        """
        Args:
            val_losses (list):
            current_epoch (int):
        Returns:
            stop (bool):
        """
        if val_losses[current_epoch - 1] <= val_losses[current_epoch]:
            stop = True
        else:
            stop = False
        return stop

    def generate_targets(self, input, stop_threshold=0.5):
        """ """
        batch_size = input.size(0)
        max_len = input.size(1)
        # print(f"shape encoder input non proiettato {input.shape}")
        enc_input = self.input_projection(input)
        print(f"shape encoder input proiettato {enc_input.shape}")
        dec_input = self.sos_token.expand(batch_size, 1, self.num_units)
        print(f"shape dec input con sos iniziale e basta: {dec_input.shape}")
        generated = []
        for t in range(max_len):
            print(f"particella {t}")
            attention_mask = nn.Transformer.generate_square_subsequent_mask(
                t + 1
            ).to(self.device)
            print(
                f"attention mask shape al passo t={t}: {attention_mask.shape}"
            )
            output = self.transformer(
                src=enc_input, tgt=dec_input, tgt_mask=attention_mask
            )
            last_token = output[:, -1, :]
            print(f"last token shape: {last_token.shape}")
            proj_token = self.particle_head(last_token).squeeze(-1)
            print(f"proj_token shape: {proj_token.shape}")
            eos = torch.sigmoid(self.eos_head(last_token)).squeeze(-1)
            print(f"eos shape: {eos.shape}")
            generated.append(proj_token)
            next_input = self.input_projection(
                proj_token.unsqueeze(-1)
            ).unsqueeze(1)
            print(f"next input shape: {next_input.shape}")
            dec_input = torch.cat([dec_input, next_input], dim=1)
            print(f"dec input shape al passo t={t}: {dec_input.shape}")
            if (eos > stop_threshold).all():
                break
        generated_sequence = torch.stack(generated, dim=1)
        return generated_sequence
