import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transformer import ParticleTransformer  # Assicurati che il nome del file sia corretto

# Simula una "frase" come sequenza di vettori
# Ad esempio, ogni parola è un vettore di 6 feature
sentence_input = torch.tensor([
    [[1, 0, 0, 0, 0, 0],  # "I"
     [0, 1, 0, 0, 0, 0],  # "like"
     [0, 0, 1, 0, 0, 0]]  # "cats"
], dtype=torch.float32)

# La frase target potrebbe essere ad esempio: "I love cats"
sentence_target = torch.tensor([
    [[1, 0, 0, 0, 0, 0],  # "I"
     [0, 0, 0, 1, 0, 0],  # "love"
     [0, 0, 1, 0, 0, 0]]  # "cats"
], dtype=torch.float32)

# Padding mask: no padding in questo esempio
pad_mask = torch.tensor([[False, False, False]])

# DataLoader fittizi
train_data = DataLoader([(sentence_input[0], sentence_target[0])], batch_size=1)
val_data = DataLoader([(sentence_input[0], sentence_target[0])], batch_size=1)
test_data = DataLoader([(sentence_input[0], sentence_target[0])], batch_size=1)

train_pad = DataLoader([(pad_mask[0], pad_mask[0])], batch_size=1)
val_pad = DataLoader([(pad_mask[0], pad_mask[0])], batch_size=1)
test_pad = DataLoader([(pad_mask[0], pad_mask[0])], batch_size=1)

# Istanzia il modello
model = ParticleTransformer(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    train_data_pad_mask=train_pad,
    val_data_pad_mask=val_pad,
    test_data_pad_mask=test_pad,
    dim_features=6,
    num_heads=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    num_units=8,
    dropout=0.1,
    activation='relu'
)

# Ottimizzatore e funzione di perdita
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Addestramento (1 epoca per test rapido)
train_losses, val_losses = model.train_val(
    num_epochs=1000,
    loss_func=loss_fn,
    optim=optimizer,
    val=True,
    patient_smooth=30,
    patient_early=30
)

# Forward finale per stampa
model.eval()
with torch.no_grad():
    input_tensor = sentence_input
    target_tensor = sentence_target
    tgt, tgt_pad_mask, attn_mask = model.de_padding(target_tensor, pad_mask)
    out = model.forward(input_tensor, tgt, pad_mask, tgt_pad_mask, attn_mask)

# Stampa input, target e output
print("Input:\n", input_tensor)
print("Target:\n", target_tensor)
print("Output:\n", out)
