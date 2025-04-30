import h5py
import torch
from loguru import logger

from data_processing import training_set_final

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



# for event in range (0, 50, 1):
# 	print(f"particella 1 vera: {training_set_final[event, 0, :]}")
# 	print(f"particella 1 ricostruita: {final_tensor[event, 0, :]}")
# 	print(f"ultima particella vera: {training_set_final[event, -1, :]}")
# 	print(f"ultima particella ricostruita: {final_tensor[event, -1, :]}")


