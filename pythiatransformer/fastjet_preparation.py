import numpy as np
import torch
import torch.nn as nn
from transformer import ParticleTransformer
from main import build_model
#from data_processing import loader_train, loader_padding_train
from data_processing import subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Masses array; the index corresponds to the one-hot-encoding index.
masses = np.array([
    0.93827, 0.93957, 0.49368, 0.13957, 0, 0, 0.10566, 0, 0.000511, 4.183, 1.273, 0.0935, 0.00216, 0.0047,
    0.0047, 0.00216, 0.0935, 1.273, 4.183, 0.000511, 0, 0.10566, 0, 0, 0, 0, 0.49761, 0.13957, 0.49368, 0.93957, 0.93827
    ])

transformer = build_model()
transformer.load_state_dict(torch.load("transformer_model_true.pt", map_location=device))
transformer.to(device)

outputs = []

transformer.eval()
with torch.no_grad():
    for (input, target), (input_padding_mask, target_padding_mask) in zip(transformer.train_data, transformer.train_data_pad_mask):

        input = input.to(device)
        target = target.to(device)
        input_padding_mask = input_padding_mask.to(device)
        target_padding_mask = target_padding_mask.to(device)

        target, target_padding_mask, attention_mask = transformer.de_padding(target, target_padding_mask)
        attention_mask = attention_mask.to(device)

        output = transformer.forward(input, target, input_padding_mask, target_padding_mask, attention_mask)
        print(f"il batch di output ha shape: {output.shape}")
        outputs.append(output)
