"""
This script loads a trained Transformer based model and performs
autoregressive inference to generate stable particles from status 23.
The main steps are:
- Load the preprocessed datasets from data_processing.py
- Rebuild the model architecture and load pretrained weights
- Perform autoregressive generation of target sequences.
"""
import os

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

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
) = load_saved_dataloaders(batch_size=128)


def plot_diff_histogram(res, filename="diff_hist.pdf"):
    """
    Plots a histogram of the differences between the target sums
    and predicted target sums across all events.

    Args:
        res (list): A list of differences between target sums and
                    predicted target sums for each event.
        filename (str): The name of the output PDF file where the
                        histogram will be saved. Default is
                        'diff_hist.pdf'.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(res, bins=100, color="lightgreen", edgecolor="black", alpha=0.7, log = True)
    plt.axvline(0, color="red", linestyle="--", label="Zero Error")
    plt.xlabel("Residuals")
    plt.ylabel("Counts")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Residual histogram saved to {filename}")

def plot_wasserstein_histogram(wasserstein_dists, filename="wasserstain_hist.pdf"):
    """
    Plot an histogram of Wasserstein distances per event.

    Args:
        wasserstein_dists (list): List of Wasserstein distances.
        filename (str): The name of the output PDF file where the
                        histogram will be saved. Default is
                        'wasserstain_hist.pdf'.
    """
    # Filter out NaN values if present
    clean_dists = [d for d in wasserstein_dists if not np.isnan(d)]

    plt.figure(figsize=(8, 5))
    plt.hist(clean_dists, bins=100, color='lightgreen', edgecolor='black', alpha=0.7, log = True)
    plt.xlabel('Wasserstein Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Wasserstein Distances per Event')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Residual histogram saved to {filename}")

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(
    torch.load("transformer_model.pt", map_location=device)
)
model.to(device)
model.device = device

logger.info("Starting autoregressive inference")
diff, wass = model.generate_targets()
plot_diff_histogram(diff)
plot_wasserstein_histogram(wass)
