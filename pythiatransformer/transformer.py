"""
Transformer class.
"""

import gc

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from data_processing import dict_ids

# from torch.nn import Transformer
# from torch.utils.data import TensorDataset, DataLoader


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

        self.dropout = dropout
        if not isinstance(dropout, float):
            raise TypeError(f"dropout must be float, got {type(dropout)}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        # self.batch_size = batch_size
        # if not isinstance(batch_size, int):
        #     raise TypeError(
        #         f"Batch size must be int, got {type(batch_size)}"
        #     )
        # if not (batch_size <= input_train.shape[0]):
        #     raise ValueError(
        #         f"Batch size must be smaller than the input dataset size."
        #     )

        if not (num_units % num_heads == 0):
            raise ValueError(
                "Number of units must be a multiple of number of heads."
            )

        self.activation = activation

        self.build_projection_layer()
        self.initialize_transformer()
        # logger.info("Data preprocessing...")
        # self.train_data = self.data_processing(input_train, target_train)
        # self.val_data = self.data_processing(input_val, target_val, False)
        # self.test_data = self.data_processing(input_test, target_test, False)
        # self.attention_train_data = self.data_processing(
        #     attention_input_train,
        #     attention_target_train
        # )
        # self.attention_val_data = self.data_processing(
        #     attention_input_val,
        #     attention_target_val,
        #     False
        # )
        # self.attention_test_data = self.data_processing(
        #     attention_input_test,
        #     attention_target_test,
        #     False
        # )
        # logger.info("Data preprocessed.")

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

    # def data_processing(self, input, target, shuffle = True):
    #     """This function prepares the data for training by splitting it
    #     into batches and shuffling the training data.

    #     Args:
    #         shuffle (bool):
    #     Returns:
    #         loader (Iterator): An iterator for the training
    #                                     data, with batching and
    #                                     shuffling enabled.
    #         loader (Iterator): An iterator for the test data, with
    #                                 batching and shuffling enabled.

    #     """
    #     seed = 1
    #     generator = torch.Generator() # creation of a new generator
    #     generator.manual_seed(seed)
    #     set = TensorDataset(input, target)

    #     loader = DataLoader(
    #         set,
    #         self.batch_size,
    #         shuffle = shuffle,
    #         generator = generator if shuffle else None
    #     )

    #     return loader

    def de_padding(self, input, padding_mask):
        """ """
        # print("Train Tensor:")
        # print(input)

        # print("Padding Mask:")
        # print(padding_mask)

        non_pad_mask = ~padding_mask  # inverto la matrice di padding
        num_particles = non_pad_mask.sum(
            dim=1
        )  # tensore di lunghezza batch_size, con valori il numero di particelle vere
        max_len_tensor = num_particles.max()
        # print(f"max_len = {max_len_tensor}")
        max_len = max_len_tensor.item()
        input = input[:, :max_len, :]
        padding_mask = padding_mask[:, :max_len]

        # print("Train Tensor:")
        # print(input)

        # print("Padding Mask:")
        # print(padding_mask)

        # print("att Mask:")
        # print(att_mask)
        return input, padding_mask

    def forward(self, input, target, input_mask, target_mask, attention_mask):
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
        output = self.transformer(
            src=input,
            tgt=target,
            tgt_mask=attention_mask,
            src_key_padding_mask=input_mask,
            tgt_key_padding_mask=target_mask,
        )
        output = self.output_projection(output)
        return output

    def mixed_loss(self, output, target, mask, alpha=30.0, beta=2.5):
        """
        Combina CrossEntropy per ID e MSE per px,py,pz, con:
        - rinforzo EOS corretti (alpha),
        - penalità EOS predetti in posizioni sbagliate (beta),
        evitando allocazioni inutili per EOS assenti.
        """
        device = output.device

        # Separazione componenti
        output_id = output[:, :, : len(dict_ids)]
        target_id = target[:, :, : len(dict_ids)]
        output_p = output[:, :, len(dict_ids) :]
        target_p = target[:, :, len(dict_ids) :]

        # Indici
        target_index = torch.argmax(target_id, dim=-1)
        pred_index = torch.argmax(output_id, dim=-1)

        ce = nn.CrossEntropyLoss(reduction="none")
        ce_loss = ce(output_id.transpose(1, 2), target_index)

        mask = mask.to(device)
        target_index = target_index.to(device)
        pred_index = pred_index.to(device)

        eos_index = dict_ids[-999]  # EOS token index
        eos_true_mask = target_index == eos_index
        eos_pred_mask = pred_index == eos_index
        padding_mask = mask
        valid_mask = (~padding_mask) & (~eos_true_mask)

        # MSE sui momenti (no EOS, no padding)
        mse_loss = nn.functional.mse_loss(
            output_p[valid_mask], target_p[valid_mask]
        )

        # CE solo su token fisici (no pad, no EOS)
        loss_ce_valid = ce_loss[valid_mask].mean()

        # Loss totale iniziale
        total_loss = loss_ce_valid + mse_loss

        # Rinforza EOS corretti se presenti
        eos_correct_mask = eos_true_mask & eos_pred_mask & (~padding_mask)
        if eos_correct_mask.any():
            eos_loss = ce_loss[eos_correct_mask].mean()
            total_loss += alpha * eos_loss

        # Penalizza EOS fuori posto se presenti
        eos_wrong_mask = eos_pred_mask & (~eos_true_mask) & (~padding_mask)
        if eos_wrong_mask.any():
            extra_eos_penalty = ce_loss[eos_wrong_mask].mean()
            total_loss += beta * extra_eos_penalty

        return total_loss, eos_loss.item()

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
        ce_eos_epoch = 0

        for (input, target), (input_padding_mask, target_padding_mask) in zip(
            self.train_data, self.train_data_pad_mask
        ):
            device = next(self.parameters()).device
            input = input.to(device)
            target = target.to(device)
            input_padding_mask = input_padding_mask.to(device)
            target_padding_mask = target_padding_mask.to(device)
            target, target_padding_mask = self.de_padding(
                target, target_padding_mask
            )

            # Lista per decoder_input e mask
            decoder_input_list = []
            decoder_input_mask_list = []

            # Per ogni evento nel batch
            for event in range(target.shape[0]):
                # Trova l'indice dell'EOS → ultima particella valida prima del padding
                eos_idx = (~target_padding_mask[event]).sum().item() - 1
                # Rimuovi l'EOS dal target: [0:eos_idx] + [eos_idx+1:]
                event_target = target[event]
                event_mask = target_padding_mask[event]

                event_input = torch.cat(
                    [event_target[:eos_idx], event_target[eos_idx + 1 :]],
                    dim=0,
                )
                event_input_mask = torch.cat(
                    [event_mask[:eos_idx], event_mask[eos_idx + 1 :]], dim=0
                )

                decoder_input_list.append(event_input)
                decoder_input_mask_list.append(event_input_mask)

            decoder_input = torch.stack(decoder_input_list, dim=0)
            decoder_input_padding_mask = torch.stack(
                decoder_input_mask_list, dim=0
            )

            target_4_loss = target[:, 1:, :]
            target_4_loss_padding_mask = target_padding_mask[:, 1:]
            attention_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.size(1)
            ).to(device)

            optim.zero_grad()
            output = self.forward(
                input,
                decoder_input,
                input_padding_mask,
                decoder_input_padding_mask,
                attention_mask,
            )

            loss, eos_ce = self.mixed_loss(
                output, target_4_loss, target_4_loss_padding_mask
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite at epoch {epoch + 1}")
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
            ce_eos_epoch += eos_ce

            # del input, target, input_padding_mask, target_padding_mask, output, loss, attention_mask
            # torch.cuda.empty_cache()

        # gc.collect() # forcing garbage collector
        # torch.cuda.empty_cache()

        logger.debug(f"Training loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        logger.debug(
            f"Training C.E. eos at epoch {epoch + 1}: {ce_eos_epoch:.4f}"
        )
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
        ce_eos_epoch = 0
        with torch.no_grad():  # Compute only the loss value
            for (input, target), (
                input_padding_mask,
                target_padding_mask,
            ) in zip(data_loader, mask_loader):

                device = next(self.parameters()).device
                input = input.to(device)
                target = target.to(device)
                input_padding_mask = input_padding_mask.to(device)
                target_padding_mask = target_padding_mask.to(device)

                target, target_padding_mask = self.de_padding(
                    target, target_padding_mask
                )

                # Lista per decoder_input e mask
                decoder_input_list = []
                decoder_input_mask_list = []

                # Per ogni evento nel batch
                for event in range(target.shape[0]):
                    # Trova l'indice dell'EOS → ultima particella valida prima del padding
                    eos_idx = (~target_padding_mask[event]).sum().item() - 1
                    # Rimuovi l'EOS dal target: [0:eos_idx] + [eos_idx+1:]
                    event_target = target[event]
                    event_mask = target_padding_mask[event]

                    event_input = torch.cat(
                        [event_target[:eos_idx], event_target[eos_idx + 1 :]],
                        dim=0,
                    )
                    event_input_mask = torch.cat(
                        [event_mask[:eos_idx], event_mask[eos_idx + 1 :]],
                        dim=0,
                    )

                    decoder_input_list.append(event_input)
                    decoder_input_mask_list.append(event_input_mask)

                decoder_input = torch.stack(decoder_input_list, dim=0)
                decoder_input_padding_mask = torch.stack(
                    decoder_input_mask_list, dim=0
                )

                target_4_loss = target[:, 1:, :]
                target_4_loss_padding_mask = target_padding_mask[:, 1:]
                attention_mask = (
                    nn.Transformer.generate_square_subsequent_mask(
                        decoder_input.size(1)
                    ).to(device)
                )

                output = self.forward(
                    input,
                    decoder_input,
                    input_padding_mask,
                    decoder_input_padding_mask,
                    attention_mask,
                )

                loss, eos_ce = self.mixed_loss(
                    output, target_4_loss, target_4_loss_padding_mask
                )
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Loss is not finite at epoch {epoch + 1}"
                    )
                loss_epoch += loss.item()
                ce_eos_epoch += eos_ce

                # del input, target, input_padding_mask, target_padding_mask, output, loss, attention_mask
                # torch.cuda.empty_cache()

        # gc.collect()
        # torch.cuda.empty_cache()

        if val:
            logger.debug(
                f"Validation loss at epoch {epoch + 1}: {loss_epoch:.4f}"
            )
            logger.debug(
                f"Validation C.E. eos at epoch {epoch + 1}: {ce_eos_epoch:.4f}"
            )
        else:
            logger.debug(f"Test loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        return loss_epoch

    def train_val(
        self,
        num_epochs,
        optim,
        val=True,
        patient_smooth=999,
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
        if not isinstance(patient_smooth, int):
            raise TypeError(
                f"The patien must be int, got {type(patient_smooth)}"
            )
        if not (patient_smooth < num_epochs):
            raise ValueError(
                f"Patient must be smaller than the number of epochs."
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

        # smoothness initialization
        counter_smooth = 0
        best_loss = 0
        # early_stop initialization
        counter_earlystop = 0

        logger.info("Training started!")
        for epoch in range(num_epochs):
            torch.cuda.reset_peak_memory_stats()
            train_loss_epoch = self.train_one_epoch(epoch, optim)
            val_loss_epoch = self.val_one_epoch(epoch, val)
            # train_loss.append(float(train_loss_epoch))
            # val_loss.append(float(val_loss_epoch))
            train_loss.append(train_loss_epoch)
            val_loss.append(val_loss_epoch)

            if epoch >= int(num_epochs / 100):
                # smoothness check
                stop_smooth, best_loss = self.smoothness(
                    val_loss_epoch, num_epochs, epoch, best_loss
                )
                logger.info(f"stop smooth: {stop_smooth}")
                if stop_smooth:
                    counter_smooth += 1
                if counter_smooth >= patient_smooth:
                    logger.warning(f"Stop at epoch {epoch + 1}.")
                    break
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

    def smoothness(self, val_loss, num_epochs, current_epoch, best_loss):
        """
        Args:
            val_loss (float):
            num_spochs (int):
            current_epoch (int):
            best_loss (float):
        Returns:
            stop (bool):
            best_loss (float)
        """
        stop = False
        if current_epoch == int(num_epochs / 100):
            best_loss = val_loss
        else:
            if val_loss <= best_loss:
                best_loss = val_loss
            else:
                stop = True
        return stop, best_loss

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

    def generate_target(self, input, input_mask, target_reference):
        """
        Autoregressive generation of targets, one event at a time.

        Args:
            input (Tensor): Input tensor [B, N_in, F].
            input_mask (BoolTensor): Padding mask [B, N_in].
            target_reference (Tensor): Used to infer max_len per batch.

        Returns:
            padded_outputs (Tensor): [B, T_max, F] with padding.
            padding_mask (BoolTensor): [B, T_max], 1 = padding.
        """
        self.eval()
        eos_index = dict_ids[-999]

        # Project inputs
        input_proj = self.input_projection(input)
        B = input.size(0)
        max_len = target_reference.size(1)
        outputs_list = []
        device = input.device

        for i in range(B):
            single_input = input_proj[i, :, :].unsqueeze(0)  # [1, N, F]
            single_mask = input_mask[i, :].unsqueeze(0)  # [1, N]
            ############################################################
            # CONTROLLARE
            sos_vec = torch.zeros(1, 1, self.dim_features, device=device)
            sos_vec[0, 0, 0] = (
                -999
            )  # primo canale = id, lo stesso usato per EOS
            sos_proj = self.input_projection(sos_vec)
            target = sos_proj  # [1, 1, H]
            target = torch.zeros((1, 1, self.num_units), device=device)
            ############################################################
            generated = []

            found_eos = False
            for step in range(max_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    target.size(1)
                ).to(device)
                output = self.transformer(
                    src=single_input,
                    tgt=target,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=single_mask,
                )
                next_token = self.output_projection(
                    output[:, -1:, :]
                )  # shape [1, 1, F]

                # # === DEBUG: controllo EOS e distribuzione ID ===
                # logits = next_token[0, 0, :len(dict_ids)]
                # pred_id = torch.argmax(logits).item()

                # # Softmax opzionale per vedere se EOS è mai favorito
                # probs = torch.softmax(logits, dim=0)

                # print(f"[Evento {i} | Step {step+1}] pred_id = {pred_id} "
                #     f"(eos_index = {eos_index}) | prob_eos = {probs[eos_index]:.4f} | "
                #     f"max_prob = {probs[pred_id]:.4f}")
                # # ================================================

                generated.append(next_token.squeeze(1))  # [1, F]

                # Controlla EOS
                pred_id = torch.argmax(
                    next_token[0, 0, : len(dict_ids)]
                ).item()
                print(f"[Evento {i} | Step {step+1}] pred_id = {pred_id}")
                if pred_id == eos_index:
                    found_eos = True
                    break

                # Aggiungi token come input
                projected = self.input_projection(next_token)
                target = torch.cat([target, projected], dim=1)
            if not found_eos:
                print(f"⚠️  [Evento {i}] EOS NON trovato dopo {max_len} step!")
            else:
                print(f"[Evento {i}] EOS trovato dopo {step+1} step")

            sequence = torch.cat(generated, dim=0)  # [Tᵢ, F]
            outputs_list.append(sequence)

        # Pad le sequenze a lunghezza massima
        padded_outputs = pad_sequence(
            outputs_list, batch_first=True
        )  # [B, T_max, F]

        # Padding mask per gli outputs
        lengths = [seq.size(0) for seq in outputs_list]
        T_max = padded_outputs.size(1)
        padding_mask = torch.tensor(
            [[0] * l + [1] * (T_max - l) for l in lengths], dtype=torch.bool
        )

        return padded_outputs, padding_mask
