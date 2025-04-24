"""
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer

from loguru import logger

from transformer import ParticleTransformer
from data_processing import training_set_final, training_set_23
from data_processing import validation_set_final, validation_set_23
from data_processing import test_set_final, test_set_23
from data_processing import attention_train_23, attention_train_final
from data_processing import attention_val_23, attention_val_final
from data_processing import attention_test_23, attention_test_final

print(f"len train: {training_set_23.shape[0]}, len val: {validation_set_23.shape[0]}, len test: {test_set_23.shape[0]}")
print(torch.mean(training_set_23), torch.std(training_set_23))

def plot_losses(train_loss, val_loss):
    """
    """
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend()
    plt.grid(True)
    plt.show()

transformer = ParticleTransformer(
    input_train = training_set_23,
    input_val = validation_set_23,
    input_test = test_set_23,
    target_train = training_set_final,
    target_val = validation_set_final,
    target_test = test_set_final,
    attention_input_train = attention_train_23,
    attention_target_train = attention_train_final,
    attention_input_val = attention_val_23,
    attention_target_val = attention_val_final,
    attention_input_test = attention_test_23,
    attention_target_test = attention_test_final,
    dim_features = training_set_23.shape[2],
    num_heads = 8,
    num_encoder_layers = 2,
    num_decoder_layers = 2,
    num_units = 64,
    dropout = 0.1,
    batch_size = 100,
    activation = nn.ReLU()
)

epochs = 100
loss_func = nn.MSELoss()
learning_rate = 1e-2
logger.info(
    f"Batch size: {transformer.batch_size}, Epochs: {epochs}, "
    f"Learning rate: {learning_rate}, loss function: {loss_func}."
)

##################################################################################################
print("prova generazione particelle con generate_target")
# Lunghezza massima della sequenza target da generare
max_len = 1217  # Lunghezza delle particelle finali

# Esegui inferenza autoregressiva
output = transformer.generate_target(
    input=test_set_23,  # Input delle particelle di status 23
    input_mask=attention_test_23,  # Maschera per l'input
    max_len=max_len  # Lunghezza massima da generare
)

print(f"Output shape generate_target: {output.shape}")  # Controlla la forma dell'output


##################################################################################################

train_loss, val_loss = transformer.train_val(
    num_epochs = epochs,
    loss_func = loss_func,
    optim = optimizer.Adam(transformer.parameters(), lr=learning_rate)
)

plot_losses(train_loss, val_loss)

# girare fastjet sull'output
# cluster sequence
# one hot encoding ID
# id px, py, pz
# ordinare le particelle dentro l'evento
# sorting sui pt
# utilizzare eventi più semplici (QCD)
