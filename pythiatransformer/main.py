import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer
from loguru import logger

from data_processing import load_and_prepare_data
from transformer import ParticleTransformer

batch_size = 32
(
    loader_train,
    loader_val,
    loader_test,
    loader_padding_train,
    loader_padding_val,
    loader_padding_test,
    subset,
    mean_final,
    std_final,
) = load_and_prepare_data(filename="events_1M.root", batch_size=batch_size)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
epochs = 500
steps_per_epoch = len(loader_train)  # dataset_size // batch_size
total_steps = epochs * steps_per_epoch


def plot_losses(
    train_loss, val_loss, filename="learning_curve_800k.pdf", dpi=1200
):
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=dpi)
    plt.close()


def build_model():
    """Build a unique istance of ParticleTransformer class."""
    return ParticleTransformer(
        train_data=loader_train,
        val_data=loader_val,
        test_data=loader_test,
        train_data_pad_mask=loader_padding_train,
        val_data_pad_mask=loader_padding_val,
        test_data_pad_mask=loader_padding_test,
        dim_features=subset.shape[0],
        num_heads=16,
        num_encoder_layers=2,
        num_decoder_layers=4,
        num_units=64,
        num_classes=34,
        dropout=0.1,
        activation=nn.ReLU(),
    )


def train_and_save_model():
    transformer = build_model()
    transformer.to(device)
    num_params = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad
    )
    print(f"Numero totale di parametri allenabili: {num_params}")
    print(f"Numero totali di parametri")
    print(sum(p.numel() for p in transformer.parameters()))

    learning_rate = 5e-4
    optim = optimizer.Adam(
        transformer.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss(ignore_index=0)
    train_loss, val_loss = transformer.train_val(
        num_epochs=epochs, optim=optim, loss_func=ce
    )

    plot_losses(train_loss, val_loss)

    torch.save(transformer.state_dict(), "transformer_model_800k.pt")
    logger.info("Modello salvato in transformer_model_800k.pt")


# def generate_outputs_and_save():
#     transformer = build_model()
#     transformer.load_state_dict(torch.load("transformer_model_true.pt"))
#     transformer.to(device)
#     transformer.eval()

#     output_file = "output_tensor_true.h5"
#     logger.info("Prova generazione particelle con forward")

#     with h5py.File(output_file, "w") as h5f:
#         with torch.no_grad():
#             for batch_idx, (
#                 (inputs, targets),
#                 (inputs_mask, targets_mask)
#             ) in enumerate(zip(loader_train, loader_padding_train)):

#                 targets, target_padding_mask, attention_mask = (
#                     transformer.de_padding(targets, targets_mask)
#                 )

#                 inputs = inputs.to(device)
#                 targets = targets.to(device)
#                 inputs_mask = inputs_mask.to(device)
#                 target_padding_mask = target_padding_mask.to(device)
#                 attention_mask = attention_mask.to(device)

#                 outputs = transformer.forward(
#                     inputs,
#                     targets,
#                     inputs_mask,
#                     target_padding_mask,
#                     attention_mask
#                 )

#                 outputs_np = outputs.detach().cpu().numpy()

#                 h5f.create_dataset(
#                     f"batch_{batch_idx}",
#                     data=outputs_np,
#                     compression="gzip"
#                 )
#                 logger.info(f"Batch {batch_idx + 1} salvato in HDF5.")

#                 del inputs, targets, outputs, outputs_np
#                 del inputs_mask, target_padding_mask, attention_mask
#                 torch.cuda.empty_cache()

#     logger.info(
#         f"Tutti i batch salvati in un unico file HDF5: {output_file}"
#     )


if __name__ == "__main__":
    train_and_save_model()
