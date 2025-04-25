"""
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer
import h5py
import numpy as np

from loguru import logger

from transformer import ParticleTransformer
from data_processing import training_set_final, training_set_23
from data_processing import validation_set_final, validation_set_23
from data_processing import test_set_final, test_set_23
from data_processing import attention_train_23, attention_train_final
from data_processing import attention_val_23, attention_val_final
from data_processing import attention_test_23, attention_test_final

print(f"len train: {training_set_23.shape[0]}, len val: {validation_set_23.shape[0]}, len test: {test_set_23.shape[0]}")

def plot_losses(train_loss, val_loss, filename="learning_curve.pdf", dpi=1200):
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
    input_train = training_set_23,
    input_val = validation_set_23,
    input_test = test_set_23,
    target_train = training_set_final,
    target_val = validation_set_final,
    target_test = test_set_final,
    attention_input_train = attention_train_23,
    attention_target_train = attention_train_final,
    attention_input_val = attention_val_23,
    attention_target_val = attention_val_final,
    attention_input_test = attention_test_23,
    attention_target_test = attention_test_final,
    dim_features = training_set_23.shape[2],
    num_heads = 8,
    num_encoder_layers = 2,
    num_decoder_layers = 2,
    num_units = 64,
    dropout = 0.1,
    batch_size = 100,
    activation = nn.ReLU()
)

epochs = 100
loss_func = nn.MSELoss()
learning_rate = 1e-2
logger.info(
    f"Batch size: {transformer.batch_size}, Epochs: {epochs}, "
)


train_loss, val_loss = transformer.train_val(
    num_epochs = epochs,
    loss_func = loss_func,
    optim = optimizer.Adam(transformer.parameters(), lr=learning_rate)
)

plot_losses(train_loss, val_loss)

# File unico HDF5
output_file = "output_tensor.h5"

# Crea il file HDF5
with h5py.File(output_file, "w") as h5f:
    logger.info("Prova generazione particelle con forward")
    forward_dataset = transformer.data_processing(training_set_23, training_set_final, shuffle=False)
    forward_mask = transformer.data_processing(attention_train_23, attention_train_final, shuffle=False)

    for batch_idx, ((inputs, targets), (inputs_mask, targets_mask)) in enumerate(zip(forward_dataset, forward_mask)):
        outputs = transformer.forward(inputs, targets, inputs_mask, targets_mask)
        outputs_np = outputs.detach().numpy()  # Converti il tensore PyTorch in array NumPy

        # Crea un dataset per ogni batch
        h5f.create_dataset(f"batch_{batch_idx}", data=outputs_np, compression="gzip")
        logger.info(f"Batch {batch_idx+1} salvato in HDF5.")

logger.info(f"Tutti i batch salvati in un unico file HDF5: {output_file}")


# girare fastjet sull'output
# cluster sequence
# one hot encoding ID
# id px, py, pz
# ordinare le particelle dentro l'evento
# sorting sui pt
# utilizzare eventi più semplici (QCD)
