import h5py
import torch
from loguru import logger

from data_processing import training_set_final

# Carica il file HDF5
output_file = "output_tensor_descending_pT.h5"

with h5py.File(output_file, "r") as h5f:
    # Leggi tutti i dataset
    saved_tensors = [torch.tensor(h5f[f"batch_{i}"][:]) for i in range(len(h5f))]

    # Concatena i batch (se necessario)
    final_tensor = torch.cat(saved_tensors, dim=0)
    logger.info(f"Tensore concatenato, dimensione finale: {final_tensor.shape}")

for event in range (0, 50, 1):
	print(f"particella 1 vera: {training_set_final[event, 0, :]}")
	print(f"particella 1 ricostruita: {final_tensor[event, 0, :]}")
	print(f"ultima particella vera: {training_set_final[event, -1, :]}")
	print(f"ultima particella ricostruita: {final_tensor[event, -1, :]}")


