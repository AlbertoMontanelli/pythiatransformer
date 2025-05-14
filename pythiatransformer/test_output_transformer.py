import torch
import torch.nn as nn

from data_processing import (
    dict_ids,
    loader_padding_train,
    loader_train,
    subset,
)
from transformer import ParticleTransformer

# Imposta il dispositivo
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Costruisci il modello e carica i pesi
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
    activation=nn.ReLU(),
)
model.load_state_dict(
    torch.load("transformer_model_eos_2.pt", map_location=device)
)
model.to(device)
model.eval()

# Prendi un batch di dati
inputs, targets = next(iter(loader_train))
inputs_mask, targets_mask = next(iter(loader_padding_train))

inputs = inputs.to(device)
targets = targets.to(device)
inputs_mask = inputs_mask.to(device)
targets_mask = targets_mask.to(device)

# ====== INFERENZA DURANTE IL TRAINING (forward diretto) ======
# De-padding dei target reali
targets_clean, targets_pad_mask, attn_mask = model.de_padding(
    targets, targets_mask
)
targets_pad_mask = targets_pad_mask.to(device)
attn_mask = attn_mask.to(device)

print("Inizio inferenza teacher forcing")
with torch.no_grad():
    output_direct = model.forward(
        inputs, targets_clean, inputs_mask, targets_pad_mask, attn_mask
    )

# ====== INFERENZA AUTOREGRESSIVA EVENTO PER EVENTO ======
print("Inizio inferenza autoregressiva")
with torch.no_grad():
    output_gen, output_gen_mask = model.generate_target(
        inputs, inputs_mask, targets
    )

# ====== STAMPA DI CONFRONTO ======
evento_idx = 0

for i in range(10):
    print(
        f"\n================ Evento {evento_idx}, Particella {i} ================\n"
    )

    # --- Target reale
    print("🎯 Target reale (vero):")
    print(targets_clean[evento_idx, i].cpu().numpy())

    # --- Output generato durante il training (forward diretto)
    print("\n📘 Predizione diretta (forward training):")
    print(output_direct[evento_idx, i].cpu().numpy())

    # --- Output generato via inferenza autoregressiva
    if i < output_gen.size(1):
        print("\n🤖 Inferenza autoregressiva:")
        print(output_gen[evento_idx, i].cpu().numpy())
    else:
        print("\n🤖 Inferenza autoregressiva: [n/a - sequenza più corta]")

# ====== Altre info utili ======
print("\nShape output training (forward):", output_direct.shape)
print("Shape output generato:", output_gen.shape)


# # Stampa esempio
# # print("\n--- ESEMPIO EVENTO 0 ---")
# # print("Input (status 23):")
# # print(inputs[0].cpu())
# # print("\nTarget (final state):")
# # print(targets[0].cpu())
# # print("\nPredizione (output generato):")

# print(output.shape)

# # Scegli un evento e una particella a caso da stampare
# evento_idx = 0
# # particella_idx = 0

# for i in range(10):
#     # print("\n--- PARTICELLA SINGOLA ---")
#     print(f"Evento: {evento_idx}, Particella: {i}\n")

#     # print("Input (status 23):")
#     # print(inputs[evento_idx, particella_idx].cpu().numpy())

#     print("\nTarget (finale vero):")
#     print(targets[evento_idx, i].cpu().numpy())

#     print("\nPredizione (generata):")
#     print(output[evento_idx, i].cpu().numpy())
