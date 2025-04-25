import h5py
import torch
from loguru import logger

# Carica il file HDF5
output_file = "output_tensor.h5"

with h5py.File(output_file, "r") as h5f:
    # Leggi tutti i dataset
    saved_tensors = [torch.tensor(h5f[f"batch_{i}"][:]) for i in range(len(h5f))]

    # Concatena i batch (se necessario)
    final_tensor = torch.cat(saved_tensors, dim=0)
    logger.info(f"Tensore concatenato, dimensione finale: {final_tensor.shape}")

