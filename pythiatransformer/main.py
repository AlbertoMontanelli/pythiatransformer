"""
This script builds, trains, evaluates and saves a transformer based
model designed to predict stable particles from status 23 particles.
The main steps are:
- Load the preprocessed datasets from data_processing.py
- Define and build the transformer model.
- Train and validate the model.
- Plot training and validation loss curves.
- Save the trained model.
"""
import os

from loguru import logger
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer

from data_processing import load_saved_dataloaders
from transformer import ParticleTransformer

(
    loader_train,
    loader_val,
    loader_test,
    loader_padding_train,
    loader_padding_val,
    loader_padding_test,
    subset,
) = load_saved_dataloaders(batch_size=256)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 100

def plot_losses(
    train_loss, val_loss, filename="learning_curve.pdf", dpi=1200
):
    """
    Plots and saves the training and validation loss curves over
    epochs.

    Args:
        train_loss (list): Training loss values.
        val_loss (list): Validation loss values.
        filename (string): Output file name for the saved plot.
        dpi (int): Resolution of the saved figure.
    """
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=dpi)
    plt.close()


def build_model():
    """
    Build a unique istance of ParticleTransformer class.

    Returns:
        ParticleTransformer: A model ready for training or inference.
    """
    return ParticleTransformer(
        train_data=loader_train,
        val_data=loader_val,
        test_data=loader_test,
        train_data_pad_mask=loader_padding_train,
        val_data_pad_mask=loader_padding_val,
        test_data_pad_mask=loader_padding_test,
        dim_features=subset.shape[0],
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=4,
        num_units=128,
        dropout=0.1,
        activation=nn.ReLU(),
    )

def train_and_save_model():
    """
    Trains the ParticleTransformer model and saves the trained
    weights to a `.pt` file.
    """
    transformer = build_model()
    transformer.to(device)
    transformer.device = device

    num_params_learnable = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad
    )
    num_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Number of learnable parameters: {num_params_learnable}")
    logger.info(f"Number of total parameters: {num_params}")

    learning_rate = 5e-4
    optim = optimizer.Adam(
        transformer.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    train_loss, val_loss = transformer.train_val(
        num_epochs=epochs, optim=optim
    )

    plot_losses(train_loss, val_loss)

    torch.save(transformer.state_dict(), "transformer_model.pt")
    logger.info("Model saved: transformer_model.pt")


if __name__ == "__main__":
    train_and_save_model()
