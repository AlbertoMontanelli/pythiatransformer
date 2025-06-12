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
    num_heads=16,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_units=64,
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

    print("Input:")
    print(inputs[evento_idx].cpu().numpy().tolist())

    print("\n Target reale:")
    print(targets[evento_idx].cpu().numpy().tolist())

    print("\n Target predetto:")
    print(output_gen[evento_idx].cpu().numpy())


# ====== INFERENZA DURANTE IL TRAINING (forward diretto) ======
# # De-padding dei target reali
# targets_clean, targets_pad_mask = model.de_padding(targets, targets_mask)
# targets_pad_mask = targets_pad_mask.to(device)

# # Lista per decoder_input e mask
# decoder_input_list = []
# decoder_input_mask_list = []

# # Per ogni evento nel batch
# for event in range(targets_clean.shape[0]):
#     event_target = targets_clean[event]
#     event_mask = targets_pad_mask[event]

#     event_input = event_target[:-1]
#     event_input_mask = event_mask[:-1]

#     decoder_input_list.append(event_input)
#     decoder_input_mask_list.append(event_input_mask)

# decoder_input = torch.stack(decoder_input_list, dim=0)
# decoder_input_padding_mask = torch.stack(decoder_input_mask_list, dim=0)

# target_4_loss = targets_clean[:, 1:, :]
# target_4_loss_padding_mask = targets_pad_mask[:, 1:]
# attention_mask = nn.Transformer.generate_square_subsequent_mask(
#     decoder_input.size(1)
# ).to(device)
# print(f"shape attention mask prima di embedding: {attention_mask.shape}")
# for i in range(10):
#     print(f"evento {i}")
#     print("decoder input:")
#     print(decoder_input[i, :, :])
#     print("decoder input padding mask:")
#     print(decoder_input_padding_mask[i, :])
#     print("target_4_loss:")
#     print(target_4_loss[i, :, :])
#     print("target_4_loss padding mask")
#     print(target_4_loss_padding_mask[i, :])
#     print("attention_mask")
#     print(attention_mask)
#     print("\n")
#     if i == 1:
#         break


# print("Inizio inferenza teacher forcing")
# with torch.no_grad():
#     output_direct = model.forward(
#         inputs,
#         decoder_input,
#         inputs_mask,
#         decoder_input_padding_mask,
#         attention_mask,
#     )

# ====== Altre info utili ======
# print("\nShape output training (forward):", output_direct.shape)
# print("Shape output generato:", output_gen.shape)


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
