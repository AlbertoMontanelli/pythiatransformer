"""This script loads a trained Transformer based model and performs
autoregressive inference to generate stable particles from status 23.
The main steps are:
- Load the preprocessed datasets from data_processing.py
- Rebuild the model architecture and load pretrained weights
- Perform autoregressive generation of target sequences.
"""
import os

from loguru import logger
import torch
import torch.nn as nn

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

def build_model():
    """Build a unique istance of ParticleTransformer class.

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
    torch.load("transformer_model_1M.pt", map_location=device)
)
model.to(device)
model.device = device

logger.info("Starting autoregressive inference")
with torch.no_grad():
    output_gen = model.generate_targets()
