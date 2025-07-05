import os

import torch
import torch.nn as nn
from data_processing import load_saved_dataloaders
from transformer import ParticleTransformer

# Imposta il dispositivo
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

(
    loader_train,
    loader_val,
    loader_test,
    loader_padding_train,
    loader_padding_val,
    loader_padding_test,
    subset,
) = load_saved_dataloaders(batch_size=128)


# Costruisci il modello e carica i pesi
model = ParticleTransformer(
    train_data=loader_train,
    val_data=loader_train,
    test_data=loader_test,
    train_data_pad_mask=loader_padding_train,
    val_data_pad_mask=loader_padding_train,
    test_data_pad_mask=loader_padding_test,
    dim_features=subset.shape[0],
    num_heads=8,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_units=128,
    dropout=0.1,
    activation=nn.ReLU(),
)
model.load_state_dict(
    torch.load("transformer_model_1M.pt", map_location=device)
)
model.to(device)
model.device = device


# ====== INFERENZA AUTOREGRESSIVA EVENTO PER EVENTO ======
print("Inizio inferenza autoregressiva")
with torch.no_grad():
    output_gen = model.generate_targets()
