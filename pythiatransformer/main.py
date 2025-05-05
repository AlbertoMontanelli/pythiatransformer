"""
"""
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optimizer
import h5py
import numpy as np

from loguru import logger

from transformer import ParticleTransformer
from data_processing import loader_train, loader_attention_train
from data_processing import loader_val, loader_attention_val
from data_processing import loader_test, loader_attention_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 100

def plot_losses(train_loss, val_loss, filename="learning_curve_descending_pT.pdf", dpi=1200):
    """
    Plot training and validation losses and save the plot to a file.

    Parameters:
    - train_loss (list): List of training loss values.
    - val_loss (list): List of validation loss values.
    - filename (str): File name to save the plot. Defaults to 'learning_curve.png'.
    - dpi (int): Dots per inch for the saved figure. Defaults to 300.
    """
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

transformer = ParticleTransformer(
    train_data = loader_train,
    val_data = loader_val,
    test_data = loader_test,
    train_data_pad_mask = loader_attention_train,
    val_data_pad_mask = loader_attention_val,
    test_data_pad_mask = loader_attention_test,
    dim_features = 35,
    num_heads = 8,
    num_encoder_layers = 2,
    num_decoder_layers = 2,
    num_units = 64,
    dropout = 0.1,
    activation = nn.ReLU()
)
transformer.to(device)

loss_func = nn.MSELoss()
learning_rate = 5e-4

train_loss, val_loss = transformer.train_val(
    num_epochs = epochs,
    loss_func = loss_func,
    optim = optimizer.Adam(transformer.parameters(), lr=learning_rate)
)

plot_losses(train_loss, val_loss)

# File unico HDF5
output_file = "output_tensor_descending_pT.h5"

# Crea il file HDF5
with h5py.File(output_file, "w") as h5f:
    logger.info("Prova generazione particelle con forward")

    for batch_idx, ((inputs, targets), (inputs_mask, targets_mask)) in enumerate(zip(loader_train, loader_attention_train)):
	targets, target_padding_mask, attention_mask = transformer.de_padding(targets, targets_mask)
	inputs = inputs.to(device)
	targets = targets.to(device)
	inputs_mask = inputs_mask.to(device)
	target_padding_mask = target_padding_mask.to(device)
	attention_mask = attention_mask.to(device)
        outputs = transformer.forward(inputs, targets, inputs_mask, target_padding_mask, attention_mask)
        outputs_np = outputs.detach().numpy()  # Converti il tensore PyTorch in array NumPy

        # Crea un dataset per ogni batch
        h5f.create_dataset(f"batch_{batch_idx}", data=outputs_np, compression="gzip")
        logger.info(f"Batch {batch_idx+1} salvato in HDF5.")

logger.info(f"Tutti i batch salvati in un unico file HDF5: {output_file}")
