"""
Custom transformer model and training script for a toy regression task.

This script implements a simple regression task using a transformer
architecture. Given a single float input ``x``, the model learns to
generate a sequence of values ``[y_1, y_2, ..., y_k]`` such that the
sum of the sequence equals ``x``, with optional zero-padding up to a
fixed maximum length.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _check_type(var, name, t):
    """Check whether a variable is of the expected type."""
    if not isinstance(var, t):
        raise TypeError(
            f"{name} must be of type {t.__name__}, got {type(var).__name__}"
        )


def plot_learning_curve(
    train_loss,
    filename="toy_learning_curve.pdf",
    title="Learning Curve",
    dpi=1200,
):
    """
    Plot and save the training and validation loss curves over epochs.

    Parameters
    ----------
    train_loss : list[float]
        Training loss values.
    filename: str
        File name to save the plot.
    title: str
        Title of the plot.
    dpi: int
        Resolution of the saved figure.
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    base_dir = Path(__file__).resolve().parent
    plt.savefig(base_dir / filename, dpi=dpi)
    plt.close()
    logger.info(f"Learning curve saved as {filename}")


class ToyDataset(Dataset):
    """
    Prepare the dataset for a toy regression task.

    Each sample consists of:

    - input scalar ``x`` sampled uniformly from (0, 10);
    - target sequence ``(y_1, ..., y_k)`` whose sum equals ``x``, with
      random length ``k`` in ``[1, max_len]``; zero-padded
      to ``max_len``;
    - boolean mask for valid (non-padded) elements.

    The goal is to teach models to decompose a scalar into a positive
    sequence that sums to it.
    """

    def __init__(self, n_samples=10000, max_len=10, seed=42):
        """
        Class constructor.

        Parameters
        ----------
        n_samples: int
            Number of samples to generate.
        max_len: int
            Maximum sequence length for the target y.
        seed: int
            Random seed for reproducibility.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        self.max_len = max_len
        self.data = []
        for _ in range(n_samples):
            x = random.uniform(0.0, 1.0) * 10
            k = random.randint(1, max_len)

            # Create a tensor 1D of length k with values in [0,1].
            v = torch.rand(k)
            # Normalize v to sum=1, then scale by x to decompose it
            # into y.
            v = v / v.sum()
            y = v * x
            # Create padded data tensor and padding mask.
            y_pad = torch.zeros(max_len)
            mask = torch.zeros(max_len, dtype=torch.bool)
            y_pad[:k] = y
            mask[:k] = 1
            self.data.append((x, y_pad, mask, k))

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        len: int
            The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the sample at the given index.

        Parameters
        ----------
        idx: int
            Index of the sample to retrieve.

        Returns
        -------
        x: torch.Tensor
            Scalar input.
        y_pad: torch.Tensor
            Target sequence.
        mask: torch.Tensor
            Boolean mask indicating valid (non-padded) elements.
        length: int
            Actual sequence length before padding.
        """
        x, y_pad, mask, length = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), y_pad, mask, length


class ToyTransformer(nn.Module):
    """
    Custom transformer model for the toy regression task.

    Implements a toy transformer model for sequential regression.
    The model takes as input a scalar x and learns to generate a
    sequence of positive values that sum approximately to x. It uses
    a transformer architecture with encoder-decoder structure.
    """

    def __init__(
        self,
        d_model=64,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=10,
    ):
        """
        Class constructor.

        Parameters
        ----------
        d_model: int
            Dimensionality of the internal representation.
        nhead: int
            Number of attention heads.
        num_encoder_layers: int
            Number of encoder layers.
        num_decoder_layers: int
            Number of decoder layers.
        dim_feedforward: int,
            Dimension of the feedforward network.
        dropout: float
            Dropout probability.
        max_len: int
            Maximum output sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len

        _check_type(d_model, "d_model", int)
        _check_type(nhead, "nhead", int)
        _check_type(num_encoder_layers, "num_encoder_layers", int)
        _check_type(num_decoder_layers, "num_decoder_layers", int)
        _check_type(dim_feedforward, "dim_feedforward", int)
        _check_type(dropout, "dropout", float)
        _check_type(max_len, "max_len", int)

        if d_model % nhead != 0:
            raise ValueError("d_model must be a multiple of nhead.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}")

        self.in_proj = nn.Linear(1, d_model)  # Embedding in d_model.
        # Learnable start-of-sequence token.
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.d_model, 1)
        # Head to predict the end-of-sequence (EOS) probability.
        self.stop_head = nn.Linear(self.d_model, 1)

    def forward_teacher(self, x, y, mask):
        """
        Forward pass in teacher forcing mode.

        The model receives a scalar input ``x``, a padded target
        sequence ``y`` and the corresponding padding mask.
        The decoder receives the target sequence with a special start
        of sequence (``SOS``) token. It predicts the output sequence
        and a stop signal for each step.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B,)``.
        y: torch.Tensor
            Target padded sequences of shape ``(B, T)``.
        mask: torch.Tensor
            Boolean mask of shape ``(B, T)``, where ``True`` indicates
            non-padded elements.

        Returns
        -------
        y_hat: torch.Tensor
            Predicted sequence of shape ``(B, T+1)``.
        stop_logits: torch.Tensor
            Logits for stop signalof shape ``(B, T+1)``.
        """
        if x.dim() != 1:
            raise ValueError(
                f"x must be a 1D tensor of shape (B,), got shape {x.shape}"
            )
        if y.dim() != 2:
            raise ValueError(
                f"y must be a 2D tensor of shape (B, T), got shape {y.shape}"
            )
        if y.shape != mask.shape:
            raise ValueError(
                "Shape mismatch: y and mask must have the same shape."
            )

        b, t = y.size()
        device = x.device
        src = self.in_proj(x.unsqueeze(-1)).unsqueeze(1)  # shape [B,1,d].
        sos = self.sos_token.expand(b, -1, -1)  # shape [B,1,d].
        tgt_emb = self.in_proj(y.unsqueeze(-1))  # shape [B,T,d].
        tgt = torch.cat([sos, tgt_emb], dim=1)  # shape [B,T+1,d].
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t + 1).to(
            device
        )
        src_key_padding_mask = torch.zeros(
            b, 1, dtype=torch.bool, device=device
        )
        pad_dec = ~mask
        tgt_key_padding_mask = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=device), pad_dec],
            dim=1,
        )
        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # shape [B,T+1,d].
        y_hat = self.out_proj(out).squeeze(-1)  # shape [B,T+1].
        stop_logits = self.stop_head(out).squeeze(-1)  # shape [B,T+1].
        return y_hat, stop_logits

    def generate(self, x, max_len=None, stop_thresh=0.5):
        """
        Autoregressive inference for sequence generation.

        Starts with a special start of sequence (``SOS``) token and
        generates outputs one step at a time using the model's own
        predictions.
        Generation continues until either the maximum sequence length
        is reached or all stop probabilities exceed the given threshold.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B,)``.
        max_len: int, default=None
            Maximum number of steps to generate. Optional, if ``None``
            uses ``self.max_len``.
        stop_thresh: float, default=0.5
            Threshold for the stop probability to end generation.

        Returns
        -------
        y_seq: torch.Tensor
            Generated sequence of shape ``(B, T)``, where
            ``T ≤ max_len``.
        """
        if not 0.0 <= stop_thresh <= 1.0:
            raise ValueError(
                f"stop_thresh must be between 0 and 1, got {stop_thresh}"
            )

        if max_len is None:
            max_len = self.max_len
        device = x.device
        b = x.size(0)
        src = self.in_proj(x.unsqueeze(-1)).unsqueeze(1)
        src_key_padding_mask = torch.zeros(
            b, 1, dtype=torch.bool, device=device
        )
        tgt_emb = self.sos_token.expand(b, 1, self.d_model)
        generated = []
        for t in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                t + 1
            ).to(device)
            out = self.transformer(
                src=src,
                tgt=tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            last = out[:, -1, :]
            y_t = self.out_proj(last).squeeze(-1)
            p_stop = torch.sigmoid(self.stop_head(last)).squeeze(-1)
            generated.append(y_t)
            y_emb = self.in_proj(y_t.unsqueeze(-1)).unsqueeze(1)
            tgt_emb = torch.cat([tgt_emb, y_emb], dim=1)
            if (p_stop > stop_thresh).all():
                break
        y_seq = torch.stack(generated, dim=1)
        return y_seq


def main():
    """
    Execute the training script for the ToyTransformer model.

    - generates a toy dataset of scalar inputs and target sequences;
    - defines and trains a transformer model;
    - trains for a fixed number of epochs;
    - saves the learning curve and the trained model.
    """
    # Set hyperparameters
    N_SAMPLES = 5000
    MAX_LEN = 10
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3

    dataset = ToyDataset(n_samples=N_SAMPLES, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ToyTransformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=MAX_LEN,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss(reduction="none")
    bce_loss = nn.BCEWithLogitsLoss()

    loss_history = []
    for ep in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x_loader, y_pad_loader, mask_loader, length in loader:
            x, y_pad, mask = (
                x_loader.to(device),
                y_pad_loader.to(device),
                mask_loader.to(device),
            )
            stop_target = torch.zeros(x.size(0), MAX_LEN + 1, device=device)
            for i, L in enumerate(length):
                stop_target[i, L] = 1.0
            optim.zero_grad()
            y_hat, stop_logits = model.forward_teacher(x, y_pad, mask)
            mse_elements = mse_loss(y_hat[:, :-1], y_pad)
            mask_f = mask.float()
            valid = mask_f.sum()
            mse = (mse_elements * mask_f).sum() / valid
            stop = bce_loss(stop_logits, stop_target)
            loss = mse + stop
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        logger.info(f"Epoch: {ep + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    plot_learning_curve(loss_history)
    base_dir = Path(__file__).resolve().parent
    torch.save(model.state_dict(), f"{base_dir}/toy_model.pt")
    logger.info(f"Model saved in {base_dir}/toy_model.pt")


if __name__ == "__main__":
    main()
