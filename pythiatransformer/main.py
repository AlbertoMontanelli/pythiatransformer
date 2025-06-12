import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer
from loguru import logger

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
) = load_saved_dataloaders(batch_size=1)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = 100


def plot_losses(
    train_loss, val_loss, filename="learning_curve_1M_1.pdf", dpi=1200
):
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
    """Build a unique istance of ParticleTransformer class."""
    return ParticleTransformer(
        train_data=loader_train,
        val_data=loader_val,
        test_data=loader_test,
        train_data_pad_mask=loader_padding_train,
        val_data_pad_mask=loader_padding_val,
        test_data_pad_mask=loader_padding_test,
        dim_features=subset.shape[0],
        num_heads=16,
        num_encoder_layers=2,
        num_decoder_layers=4,
        num_units=64,
        num_classes=34,
        dropout=0.1,
        activation=nn.ReLU(),
    )


def train_and_save_model():
    transformer = build_model()
    transformer.to(device)
    transformer.device = device
    num_params = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad
    )
    print(f"Numero totale di parametri allenabili: {num_params}")
    print(f"Numero totali di parametri")
    print(sum(p.numel() for p in transformer.parameters()))

    learning_rate = 5e-4
    optim = optimizer.Adam(
        transformer.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    train_loss, val_loss = transformer.train_val(
        num_epochs=epochs, optim=optim
    )

    plot_losses(train_loss, val_loss)

    torch.save(transformer.state_dict(), "transformer_model_1M_1.pt")
    logger.info("Modello salvato in transformer_model_1M_1.pt")


if __name__ == "__main__":
    train_and_save_model()
