"""
Custom Transformer model for single-feature particle sequences.

This model predicts the sequence of final stable particles given the
sequence of input particles with status 23. The implementation is
currently validated and designed for **one scalar feature per token**,
namely the transverse momentum `pT`.

Architecture
------------
- Encoder: self-attention over the status-23 sequence to build a
  contextual "memory" of the event.
- Decoder: masked self-attention (causal) over the partially generated
  target sequence, plus cross-attention over the encoder memory at
  every layer.
- Heads: a regression head for `pT` and an EOS head that predicts a
  stop logit per time step.

Training & Inference
--------------------
- Training uses teacher forcing: the decoder input is
  ``[SOS, target[:-1]]``.
  ``Loss = MSE+BCEWithLogits``. ``Mse`` on `pT` (ignoring SOS and
  padding), ``BCEWithLogits`` on EOS (one logit per step, positioned
  immediately after the last real token).
- Inference is autoregressive: start from SOS, predict one scalar `pT`
  and an EOS probability at a time, and feed each prediction back into
  the decoder until EOS triggers or the max length is reached.

Shapes
------
- Encoder input  : ``[batch, L_src, 1]``  → projected to D
- Decoder input  : ``[batch, L_tgt, 1]``  → prepend SOS →
  ``[batch, L_tgt+1, D]``
- Decoder output : ``[batch, L_tgt+1, D]``
- ``pred``       : ``[batch, L_tgt+1]`` (regressed `pT`, after
  squeeze)
- ``eos_logits`` : ``[batch, L_tgt+1]`` (one logit per step, after
  squeeze)

Notes
-----
- **Single-feature constraint**: this implementation assumes exactly
  one scalar feature per token (`pT`). Passing more than one feature is
  not supported.
- Padding masks are boolean (``True=pad``) and are applied to both
  encoder and decoder; the decoder also uses a triangular causal mask.

"""

import gc

import numpy as np
import torch
from loguru import logger
from scipy.stats import ks_2samp, wasserstein_distance
from torch import nn


def _log_peak_memory(epoch=None):
    """
    Log the peak GPU memory allocated.

    Print the maximum GPU memory allocated during the current epoch and
    reset the counter.
    """
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    prefix = f"[Epoch: {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix} Max memory allocated: {peak_mb:.2f} MB")
    torch.cuda.reset_peak_memory_stats()


def _log_gpu_memory(epoch=None):
    """Print a clear summary of GPU memory usage."""
    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    stats = torch.cuda.memory_stats()
    total_alloc_mb = stats["allocation.all.allocated"] / 1024**2

    prefix = f"[Epoca {epoch + 1}] " if epoch is not None else ""
    print(f"{prefix}Memory allocated: {alloc_mb:.2f} MB")
    print(f"{prefix}Memory reserved: {reserved_mb:.2f} MB")
    print(
        f"{prefix}Total memory allocated througth epochs:"
        f" {total_alloc_mb:.2f} MB"
    )


def _check_type(var, name, t):
    """Check whether a variable is of the expected type."""
    if not isinstance(var, t):
        raise TypeError(
            f"{name} must be of type {t.__name__}, got {type(var).__name__}"
        )


class ParticleTransformer(nn.Module):
    """
    Custom PyTorch Transformer model.

    Transformer taking in input particles having status 23 (i.e.
    outgoing particles of the hardest subprocess) and as target the
    final particles of the event.
    """

    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        train_data_pad_mask,
        val_data_pad_mask,
        test_data_pad_mask,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        num_units,
        dropout,
        activation,
    ):
        """
        Class constructor.

        Parameters
        ----------
        train_data : DataLoader
            DataLoader containing the training input-target pairs.
        val_data : DataLoader
            DataLoader containing the validation input-target pairs.
        test_data : DataLoader
            DataLoader containing the test input-target pairs.
        train_data_pad_mask : DataLoader
            DataLoader containing the padding masks for the training
            set, used to mask padded values during attention
            computations.
        val_data_pad_mask : DataLoader
            DataLoader containing the padding masks for the validation
            set.
        test_data_pad_mask : DataLoader
            DataLoader containing the padding masks for the test set.
        num_heads : int
            Number of attention heads in the transformer architecture.
        num_encoder_layers : int
            Number of encoder layers.
        num_decoder_layers : int
            Number of decoder layers.
        num_units : int
            Number of units in each hidden layer. Typically chosen
            greater than the number of input features to enable more
            abstract representations.
        dropout : float
            Dropout probability for each neuron.
        activation : nn.Module
            Activation function used in encoder and/or decoder layers.
        """
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_pad_mask = train_data_pad_mask
        self.val_data_pad_mask = val_data_pad_mask
        self.test_data_pad_mask = test_data_pad_mask
        self.num_encoder_layers = num_encoder_layers

        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_units = num_units
        self.dropout = dropout
        self.activation = activation

        # Type controls
        _check_type(num_heads, "num_heads", int)
        _check_type(num_encoder_layers, "num_encoder_layers", int)
        _check_type(num_decoder_layers, "num_decoder_layers", int)
        _check_type(num_units, "num_units", int)
        _check_type(dropout, "dropout", float)

        # Data tensor shape and type control.
        DIM_TENSOR = 3
        for name, loader in [
            ("train_data", train_data),
            ("val_data", val_data),
            ("test_data", test_data),
        ]:
            _check_type(loader, name, torch.utils.data.DataLoader)
            # Verify a single batch.
            batch = next(iter(loader))
            x = batch[0]
            assert x.ndim == DIM_TENSOR, (
                f"{name} must have {DIM_TENSOR} dimensions [B, L, F],"
                f" got {x.shape}"
            )
            assert x.shape[-1] == 1, (
                f"{name} must have exactly 1 feature, got {x.shape[-1]}"
            )

        if not num_units % num_heads == 0:
            raise ValueError(
                "Number of units must be a multiple of number of heads."
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")

        # Loss functions used during the training
        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.build_projection_layer()
        self.initialize_transformer()

        self.device = next(self.parameters()).device
        logger.info(f"Model initialized on device: {self.device}")

    def build_projection_layer(self):
        """
        Build the input/output projection layers.

        Initializes the layers responible for projecting input features
        to the model's hidden dimentionality and outputs back to the
        original feature space.
        """
        self.input_projection = nn.Linear(1, self.num_units)
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.num_units))
        self.particle_head = nn.Linear(self.num_units, 1)
        # The EOS head returns one value per feature.
        self.eos_head = nn.Linear(self.num_units, 1)
        logger.info("Projection layers input/output created.")

    def initialize_transformer(self):
        """Initialize the transformer with the specified parameters."""
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

    def forward(self, input, target, enc_input_mask, dec_input_mask):
        """
        Build a custom forward method.

        Computes the output of the model by projecting both input
        and target into an hidden representation space, processing them
        through a Transformer, and projecting the result back to the
        original feature space.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor representing status 23 particles used as the
            encoder input sequence.
        target : torch.Tensor
            Tensor of stable final-state particles used as the decoder
            target.
        enc_input_mask : torch.Tensor
            Boolean mask applied to encoder input to ignore padding
            tokens.
        dec_input_mask : torch.Tensor
            Boolean mask applied to decoder input to ignore padding
            tokens.

        Returns
        -------
        pred : torch.Tensor
            The output tensor after processing through the model.
        eos_prob_vector : torch.Tensor
            End-of-sequence probabilities for each step.
        """
        batch_size = input.size(0)

        # Input projected.
        enc_input = self.input_projection(input)

        # SOS projected.
        sos = self.sos_token.expand(batch_size, -1, -1)

        # Decoder input projected without SOS. Then concatenate SOS to
        # dec_input.
        dec_input = self.input_projection(target)
        dec_input = torch.cat([sos, dec_input], dim=1)

        # Encoder and decoder input padding mask, decoder triangular
        # causal mask for self-attention.
        sos_mask = torch.zeros(batch_size, 1).to(self.device)
        dec_input_mask = torch.cat([sos_mask, dec_input_mask], dim=1)
        attention_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_input.size(1)
        ).to(self.device)

        # Compute the output and project it to the original feature
        # space.
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
        """
        Train the model for one epoch.

        This method runs one full epoch of training over the dataset.
        It performs the following steps for each batch:

        1. Move input data and masks to the correct device.
        2. Compute the model output and EOS (end-of-sequence)
           probabilities.
        3. Compute the regression loss (MSE) over valid particles
           only.
        4. Compute the EOS classification loss (BCE) over all
           tokens.
        5. Backpropagate and update model parameters.

        Notes
        -----
        - **Masked MSE loss**: computed only on real (non-padding)
          particles. The loss is averaged over the number of *valid
          tokens*, not the batch size. This prevents the loss from
          being diluted when padding is present.
        - **BCE loss**: computed on every sequence position because
          each token (including padding) has a meaningful EOS label
          (0 = not stop, 1 = stop).
        - Both losses are thus *token-level averages* and are directly
          comparable without additional normalization by batch size.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        optim : torch.optim.Optimizer
            Optimizer used for parameter updates.

        Returns
        -------
        loss_epoch : float
            Mean training loss over all batches for this epoch.
        """
        self.train()
        loss_epoch = 0

        for (enc_input_batch, dec_input_batch), (
            enc_input_padding_mask_batch,
            dec_input_padding_mask_batch,
        ) in zip(self.train_data, self.train_data_pad_mask):
            # Move tensors to GPU/CPU as needed.
            enc_input = enc_input_batch.to(self.device)
            dec_input = dec_input_batch.to(self.device)
            enc_input_padding_mask = enc_input_padding_mask_batch.to(
                self.device
            )
            dec_input_padding_mask = dec_input_padding_mask_batch.to(
                self.device
            )

            # Invert the padding mask: True where tokens are valid.
            inverse_dec_input_padding_mask = ~dec_input_padding_mask

            # Build EOS targets: one "1" at the true end-of-sequence
            # position for each event.
            eos_tensor = torch.zeros(
                dec_input.size(0), dec_input.size(1) + 1, dec_input.size(2)
            ).to(self.device)
            for event in range(dec_input.size(0)):
                len_event = inverse_dec_input_padding_mask[event].sum()
                eos_tensor[event, len_event] = 1

            optim.zero_grad()

            # Forward pass.
            output, eos_prob_vector = self.forward(
                enc_input,
                dec_input,
                enc_input_padding_mask,
                dec_input_padding_mask,
            )  # shape: (batch_size, max_particles).

            # 1) Masked MSE regression loss.
            # Compute elementwise squared error without internal
            # averaging.
            mse_elements = self.mse_loss(
                output[:, 1:],  # skip SOS token at position 0.
                dec_input.squeeze(-1),
            )  # shape: (batch_size, max_particles).

            # Build mask for valid tokens.
            # Shape: (batch_size, max_particles).
            mask = inverse_dec_input_padding_mask.float()
            valid = mask.sum()

            # Average only over real (non-padding) tokens.
            mse = (mse_elements * mask).sum() / valid

            # 2) BCE loss for EOS prediction.
            # Each time step has a valid label (EOS=0 or 1), so no
            # masking is needed.
            bce = self.bce_loss(eos_prob_vector, eos_tensor.squeeze(-1))

            # 3) Combined loss.
            # Both terms are averaged per token so directly comparable
            # in scale.
            loss = mse + bce

            if not torch.isfinite(loss):
                raise ValueError(f"Loss is not finite at epoch {epoch + 1}")

            # Backpropagation and optimization.
            loss.backward()
            optim.step()
            loss_epoch += loss.item()

            # Free memory and clear cache between batches.
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

        # Manual garbage collection.
        gc.collect()
        torch.cuda.empty_cache()

        # Average loss over the number of batches.
        loss_epoch /= len(self.train_data)
        logger.debug(f"Training loss at epoch {epoch + 1}: {loss_epoch:.4f}")
        return loss_epoch

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        """
        Validate the model for one epoch.

        This method runs one full pass over the validation dataset
        without gradient computation. For each batch, it repeats the
        steps of ``ParticleTransformer.train_one_epoch`` without
        backpropagation and weights update.

        Parameters
        ----------
        epoch : int
            Current epoch number, used for logging only.

        Returns
        -------
        loss_epoch : float
            Mean validation loss over all batches for this epoch.
        """
        self.eval()
        loss_epoch = 0.0

        for (enc_input_batch, dec_input_batch), (
            enc_input_padding_mask_batch,
            dec_input_padding_mask_batch,
        ) in zip(self.val_data, self.val_data_pad_mask):
            # Move tensors to GPU/CPU as needed.
            enc_input = enc_input_batch.to(self.device)
            dec_input = dec_input_batch.to(self.device)
            enc_input_padding_mask = enc_input_padding_mask_batch.to(
                self.device
            )
            dec_input_padding_mask = dec_input_padding_mask_batch.to(
                self.device
            )

            # Inverted mask: True on real tokens (non-padding).
            inverse_dec_input_padding_mask = ~dec_input_padding_mask

            # Build EOS target.
            eos_tensor = torch.zeros(
                dec_input.size(0),
                dec_input.size(1) + 1,
                dec_input.size(2),
                device=self.device,
            )
            for event in range(dec_input.size(0)):
                len_event = inverse_dec_input_padding_mask[event].sum()
                eos_tensor[event, len_event] = 1

            # Forward pass (no grad due to @torch.no_grad()).
            output, eos_prob_vector = self.forward(
                enc_input,
                dec_input,
                enc_input_padding_mask,
                dec_input_padding_mask,
            )

            # Masked MSE regression loss.
            mse_elements = self.mse_loss(output[:, 1:], dec_input.squeeze(-1))

            mask = inverse_dec_input_padding_mask.float()
            valid = mask.sum()
            mse = (mse_elements * mask).sum() / valid

            # BCE for EOS.
            bce = self.bce_loss(eos_prob_vector, eos_tensor.squeeze(-1))

            # Combined loss.
            loss = mse + bce

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Validation loss is not finite at epoch {epoch + 1}"
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

        # Mean over validation batches.
        loss_epoch /= len(self.val_data)
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
        """
        Implement the training and validation loop.

        Trains and validates the model for the given number of epochs,
        using ``train_one_epoch`` and ``loss_one_epoch`` methods.
        Implements early stopping if validation loss does not improve.

        Parameters
        ----------
        num_epochs: int
            Number of total epochs.
        optim: torch.optim.optimizer
            Optimizer used for updating model parameters.
        patient: int
            Number of consecutive epochs to wait for an improvement in
            validation loss before stopping early. Default 10.
        Returns
        -------
        train_loss: list[float]
            Training loss recorded at each epoch.
        val_loss: list[float]
            Validation loss recorded at each epoch.
        """
        _check_type(num_epochs, "num_epochs", int)
        _check_type(patient_early, "patient_early", int)

        if not patient_early < num_epochs:
            raise ValueError(
                "Patient must be smaller than the number of epochs."
            )

        train_loss = []
        val_loss = []

        counter_earlystop = 0

        logger.info("Training started!")
        for epoch in range(num_epochs):
            if torch.cuda.is_available():
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
                    train_loss = train_loss[: epoch + 1 - patient_early]
                    val_loss = val_loss[: epoch + 1 - patient_early]
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log_gpu_memory(epoch)
                _log_peak_memory(epoch)

        logger.info("Training completed!")
        return train_loss, val_loss

    def early_stopping(self, val_losses, current_epoch):
        """
        Implement early stopping.

        Checks whether the validation loss has increased or remained
        the same compared to the previous epoch.

        Parameters
        ----------
        val_losses: list
            List containing validation losses for each epoch before the
            current one.
        current_epoch: int
            Index of the current epoch.
        Returns
        -------
        stop: bool
            True if validation loss has no improved, False otherwise.
        """
        if val_losses[current_epoch - 1] <= val_losses[current_epoch]:
            stop = True
        else:
            stop = False
        return stop

    @torch.inference_mode()
    def generate_targets(self, stop_threshold=0.5):
        """
        Generate targets using the test set.

        Autoregressively generate final particles on the test set and
        compute diagnostics.
        This runs greedy autoregressive decoding with a learned SOS
        token:

        - the encoder consumes the padded status-23 inputs;
        - the decoder starts from SOS and generates one scalar `pT` at
          a time;
        - at each step an EOS probability is also predicted; if it
          exceeds ``stop_threshold`` the generation for that event
          stops.

        After generation, the method computes per-event and global
        diagnostics:

        - per-event relative residual on the `pT` sum;
        - per-event Wasserstein distance between the generated and
          target `pT` distributions (after removing paddings/zeros);
        - global (all-events, concatenated) Wasserstein distance and
          two-sample Kolmogorov-Smirnov statistic/p-value between
          generated and target tokens.

        Parameters
        ----------
        stop_threshold : float
            Threshold on the EOS probability to stop generation for an
            event. Default is ``0.5``.

        Returns
        -------
        residuals : list[float]
            Event-wise relative difference between the total `pT` of
            target and generated particles.
            For each event:
            ``(sum(target) - sum(pred)) / sum(target)``.
        wd_per_event : list[float]
            Per-event Wasserstein distances between generated and
            target tokens distributions.
        generated_tokens : numpy.ndarray
            All generated tokens `pT` from all events concatenated.
        target_tokens : numpy.ndarray
            All target tokens `pT` from all events concatenated.
        wd_global : float
            Wasserstein distance between the global generated vs target
            tokens distributions (concatenated over events).
        ks_stat : float
            Kolmogorov-Smirnov test statistic between global generated
            vs target tokens distributions.
        ks_p : float
            Kolmogorov-Smirnov two-sided p-value.
        generated_tokens_per_event : list[int]
            Number of non-padded generated tokens per event.
        target_tokens_per_event : list[int]
            Number of non-padded target tokens per event.
        """
        # Collectors.
        residuals = []
        wd_per_event = []
        generated_tokens = []
        generated_tokens_per_event = []
        target_tokens = []
        target_tokens_per_event = []

        for input_batch, target_batch in self.test_data:
            input = input_batch.to(self.device)
            target = target_batch.to(self.device)

            batch_size = input.size(0)
            max_len = target.size(1)

            enc_input = self.input_projection(input)
            dec_input = [
                self.sos_token.clone().squeeze(0) for _ in range(batch_size)
            ]
            generated = torch.zeros(batch_size, max_len, device=self.device)

            for event in range(batch_size):
                for t in range(max_len):
                    attention_mask = (
                        nn.Transformer.generate_square_subsequent_mask(
                            t + 1
                        ).to(self.device)
                    )
                    output = self.transformer(
                        src=enc_input[event],  # shape (src_len, num_units).
                        tgt=dec_input[event],  # shape (t+1, num_units).
                        tgt_mask=attention_mask,  # shape (t+1, t+1).
                    )  # shape (t+1, num_units).

                    # Take the last token of the sequence.
                    last_token = output[-1, :]

                    # Project it to the original feature space.
                    proj_token = self.particle_head(last_token)
                    eos = torch.sigmoid(self.eos_head(last_token))

                    # Save the last token in the tensor of generated
                    # tokens.
                    generated[event, t] = proj_token

                    # Project the last token to the embedded space,
                    # then concatenate the last token/particle to
                    # dec_input for the next timestep.
                    next_input = self.input_projection(proj_token).unsqueeze(
                        0
                    )
                    dec_input[event] = torch.cat(
                        [dec_input[event], next_input], dim=0
                    )
                    # Verify if EOS is above the threshold.
                    if eos > stop_threshold:
                        break

            for i in range(batch_size):
                # Compute the difference from target to predicted for
                # the pT sum in each event.
                target_sum = target[i].sum().item()
                generated_sum = generated[i].sum().item()
                residuals.append((target_sum - generated_sum) / target_sum)

                # Compute the wasserstain distance for each event.
                generated_np = generated[i].detach().cpu().numpy()
                target_np = target[i].detach().cpu().numpy()

                # De-padding.
                generated_np = generated_np[generated_np > 0]
                target_np = target_np[target_np > 0]

                # List of all tokens for all the events.
                generated_tokens.append(generated_np)
                target_tokens.append(target_np)

                # List of number of tokens for each event.
                target_tokens_per_event.append(int((target[i] != 0).sum()))
                generated_tokens_per_event.append(
                    int((generated[i] != 0).sum())
                )

                if len(generated_np) == 0 or len(target_np) == 0:
                    wd_per_event.append(float("nan"))
                else:
                    wd = wasserstein_distance(generated_np, target_np)
                    wd_per_event.append(wd)
            logger.info("Batch completed.")

            # cleanup batch
            del (
                input,
                target,
                enc_input,
                dec_input,
                generated,
                output,
                last_token,
                eos,
                proj_token,
                next_input,
                target_sum,
                generated_sum,
                generated_np,
                target_np,
                wd,
            )
            torch.cuda.empty_cache()

        # Global distributional comparison.
        generated_tokens = np.concatenate(generated_tokens)
        target_tokens = np.concatenate(target_tokens)
        wd_global = wasserstein_distance(generated_tokens, target_tokens)
        ks_stat, ks_p = ks_2samp(generated_tokens, target_tokens)

        return (
            residuals,
            wd_per_event,
            generated_tokens,
            target_tokens,
            wd_global,
            ks_stat,
            ks_p,
            generated_tokens_per_event,
            target_tokens_per_event,
        )
