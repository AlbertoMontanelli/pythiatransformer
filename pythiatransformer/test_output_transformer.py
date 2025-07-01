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
    test_data=None,
    train_data_pad_mask=loader_padding_train,
    val_data_pad_mask=loader_padding_train,
    test_data_pad_mask=None,
    dim_features=subset.shape[0],
    num_heads=8,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_units=128,
    num_classes=34,
    dropout=0.1,
    activation=nn.ReLU(),
)
model.load_state_dict(
    torch.load("transformer_model_1M.pt", map_location=device)
)
model.to(device)
model.device = device

# Prendi un batch di dati
inputs, targets = next(iter(loader_train))
inputs_mask, targets_mask = next(iter(loader_padding_train))

inputs = inputs.to(device)
targets = targets.to(device)
inputs_mask = inputs_mask.to(device)
targets_mask = targets_mask.to(device)

# per vedere se il training funziona
# loss = model.val_one_epoch(0, True)
# print(f"loss: {loss}")


# ====== INFERENZA AUTOREGRESSIVA EVENTO PER EVENTO ======
print("Inizio inferenza autoregressiva")
with torch.no_grad():
    output_gen = model.generate_targets(inputs, targets.size(1))

# ====== STAMPA DI CONFRONTO ======
evento_idx = 0

for evento_idx in range(10):
    print(f"\n================ Evento {evento_idx}================\n")

    half_sum = inputs[evento_idx].sum().item() / 2  # <-- FIXED LINE
    print("Input:")
    print(inputs[evento_idx].cpu().numpy().tolist())
    print(f"Metà somma degli input: {half_sum}")

    print("\n Target reale:")
    real_sum = targets[evento_idx].sum().item()  # <-- FIXED LINE
    print(f"Somma dei target reali: {real_sum}")
    print(targets[evento_idx].cpu().numpy().tolist())

    pred_sum = output_gen[evento_idx].sum().item()  # <-- FIXED LINE
    print(f"Somma dei target predetti: {pred_sum}")
    print("\n Target predetto:")
    print(output_gen[evento_idx].cpu().numpy())
