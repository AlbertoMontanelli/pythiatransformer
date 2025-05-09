import torch
import torch.nn as nn
from transformer import ParticleTransformer
from data_processing import loader_train, loader_padding_train
from data_processing import subset

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello
model = ParticleTransformer(
    train_data=loader_train,
    val_data=None,
    test_data=None,
    train_data_pad_mask=loader_padding_train,
    val_data_pad_mask=None,
    test_data_pad_mask=None,
    dim_features=subset.shape[0],
    num_heads=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_units=64,
    dropout=0.1,
    activation=nn.ReLU()
)
model.load_state_dict(torch.load("transformer_model_true.pt", map_location=device))
model.to(device)
model.eval()

# Inizializza un solo batch dai dati di training
inputs, targets = next(iter(loader_train))
inputs_mask, targets_mask = next(iter(loader_padding_train))

inputs = inputs.to(device)
targets = targets.to(device)
inputs_mask = inputs_mask.to(device)
targets_mask = targets_mask.to(device)

# Applica de-padding come nel main
targets, targets_pad_mask, attn_mask = model.de_padding(targets, targets_mask)
targets_pad_mask = targets_pad_mask.to(device)
attn_mask = attn_mask.to(device)

# Esegui l'inferenza
with torch.no_grad():
    output = model.forward(inputs, targets, inputs_mask, targets_pad_mask, attn_mask)

# Stampa esempio
# print("\n--- ESEMPIO EVENTO 0 ---")
# print("Input (status 23):")
# print(inputs[0].cpu())
# print("\nTarget (final state):")
# print(targets[0].cpu())
# print("\nPredizione (output generato):")

print(output.shape)

# Scegli un evento e una particella a caso da stampare
evento_idx = 0
# particella_idx = 0

for i in range(10):
    # print("\n--- PARTICELLA SINGOLA ---")
    print(f"Evento: {evento_idx}, Particella: {i}\n")

    # print("Input (status 23):")
    # print(inputs[evento_idx, particella_idx].cpu().numpy())

    print("\nTarget (finale vero):")
    print(targets[evento_idx, i].cpu().numpy())

    print("\nPredizione (generata):")
    print(output[evento_idx, i].cpu().numpy())

