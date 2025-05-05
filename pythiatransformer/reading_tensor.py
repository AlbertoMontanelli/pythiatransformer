import h5py
import torch
from loguru import logger

from data_processing import loader_train

# Carica il file HDF5
output_file = "output_tensor_descending_pT.h5"

with h5py.File(output_file, "r") as h5f:
    num_batches = len(h5f)
    logger.info(f"Numero di batch trovati: {num_batches}")

    for i in range(num_batches):
        # Leggi il batch corrente come tensore
        batch_tensor = torch.tensor(h5f[f"batch_{i}"][:])
        
        # Stampa le dimensioni del batch
        logger.info(f"Dimensioni del batch {i}: {batch_tensor.shape}")
        
        # Stampa un esempio di dati
        print(f"Batch {i}, particella 1: {batch_tensor[0, 0, :]}")
        print(f"Batch {i}, ultima particella: {batch_tensor[0, -1, :]}")

for i, (batch_23, batch_final) in enumerate(loader_train):
    print(f"Batch {i}:")
    print("Batch Final - prima particella:", batch_final[0, 0, :])
    print("Batch Final - ultima particella:", batch_final[0, -1, :])
    
    print("-" * 50)
    