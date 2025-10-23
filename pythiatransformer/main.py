"""
Build, train, evaluate and save a transformer based model.

The model is designed to predict stable particles from status 23
particles.
The main steps are:

- load the preprocessed datasets from
  ``pythiatransformer.data_processing``;
- define and build the transformer model;
- train and validate the model;
- plot training and validation loss curves;
- save the trained model.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim as optimizer
from loguru import logger
from torch import nn

from pythiatransformer.data_processing import load_saved_dataloaders
from pythiatransformer.pythia_generator import _dir_path_finder
from pythiatransformer.transformer import ParticleTransformer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20


def plot_losses(train_loss, val_loss, ev_suffix, plot_suffix=None, dpi=1200):
    """
    Plot and save the training and validation loss curves over epochs.

    Parameters
    ----------
    train_loss: list[float]
        Training loss values.
    val_loss: list[float]
        Validation loss values.
    ev_suffix: str
        String appended to the output filename identifying the number
        of events of the dataset. The file is saved as
        ``plots/learning_curve_<suffix>.pdf``
    plot_suffix: str
        Default ``None``. Additional string appended to the output
        filename to give additional identification apart from number of
        events. If not ``None`` the file is saved as
        ``plots/learning_curve_<suffix>_<plot_suffix>.pdf``
    dpi: int
        Resolution of the saved figure.
    """
    plot_dir = _dir_path_finder(data=False)

    if plot_suffix is None:
        filename = plot_dir / f"learning_curve_{ev_suffix}.pdf"
    else:
        filename = plot_dir / f"learning_curve_{ev_suffix}_{plot_suffix}.pdf"

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


def build_model(batch_size, suffix):
    """
    Build and configure a ParticleTransformer instance.

    Load the saved dataloaders for the given batch size and dataset
    suffix, build the Transformer architecture, and return both the
    model and its configuration dictionary.

    Parameters
    ----------
    batch_size : int
        Batch size to be used for loading the datasets.
    suffix : str
        Suffix identifying the dataset files to be loaded.

    Returns
    -------
    model : ParticleTransformer
        A model ready for training or inference.
    config : dict
        Dictionary containing model hyperparameters and metadata
        (batch size, layers, units, dropout, activation, suffix, etc.).
    """
    config = {
        "batch_size": batch_size,
        "num_heads": 8,
        "num_encoder_layers": 2,
        "num_decoder_layers": 4,
        "num_units": 128,
        "dropout": 0.1,
        "activation": "ReLU",
        "suffix": suffix,
    }

    (
        loader_train,
        loader_val,
        loader_test,
        loader_padding_train,
        loader_padding_val,
        loader_padding_test,
    ) = load_saved_dataloaders(
        batch_size=config["batch_size"], suffix=config["suffix"]
    )

    model = ParticleTransformer(
        train_data=loader_train,
        val_data=loader_val,
        test_data=loader_test,
        train_data_pad_mask=loader_padding_train,
        val_data_pad_mask=loader_padding_val,
        test_data_pad_mask=loader_padding_test,
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        num_units=config["num_units"],
        dropout=config["dropout"],
        activation=nn.ReLU(),
    )
    return model, config


def train_and_save_model(batch_size, suffix, info_suffix):
    """
    Train and save the weights of the ParticleTransformer model.

    Model configuration and metadata are also saved in JSON format.

    Params
    ------
    batch_size : int
        Batch size to be used for loading the datasets.
    suffix : str
        String appended to the data tensor filename identifying the
        number of events in the dataset to be loaded. The same string
        is appendend to the output plot and model filenames saved.
    info_suffix: str
        Additional string appended to the output plot and model
        filename to give additional identification apart from number of
        events.
    """
    transformer, config = build_model(batch_size, suffix)
    transformer.to(device)
    transformer.device = device

    num_params_learnable = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad
    )
    num_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Number of learnable parameters: {num_params_learnable}")
    logger.info(f"Number of total parameters: {num_params}")

    learning_rate = 5e-4
    optim = optimizer.Adam(
        transformer.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    train_loss, val_loss = transformer.train_val(
        num_epochs=EPOCHS, optim=optim
    )

    plot_losses(train_loss, val_loss, suffix, info_suffix)

    data_dir = _dir_path_finder(data=True)
    if info_suffix:
        filename = data_dir / f"transformer_model_{suffix}_{info_suffix}.pt"
        torch.save(transformer.state_dict(), filename)
        logger.info(
            f"Model saved: data/transformer_model_{suffix}_{info_suffix}.pt"
        )
        meta_path = data_dir / f"meta_{suffix}_{info_suffix}.json"
    else:
        filename = data_dir / f"transformer_model_{suffix}.pt"
        torch.save(transformer.state_dict(), filename)
        logger.info(f"Model saved: data/transformer_model_{suffix}.pt")
        meta_path = data_dir / f"meta_{suffix}.json"
    meta = {
        "model_path": filename,
        "model_config": config,
        "timestamp": datetime.now().isoformat(),
    }
    (meta_path).write_text(json.dumps(meta, indent=2))


def main():
    """
    Call ``train_and_save_model`` with parser arguments.

    CLI Parameters
    --------------
    batch_size : int, optional, default=256
        Batch size to be used for loading the datasets.
    suffix : str, required
        String appended to the data tensor filename identifying the
        number of events in the dataset to be loaded. The same string
        is appendend to output plot and model filenames saved.
    info_suffix: str, optional, default=None
        Additional string appended to the plot and model output
        filename to give additional identification apart from number of
        events.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help=(
            "String appended to the data tensor filename identifying the"
            " number of events in the dataset to be loaded. The same string"
            " is appendend to output plot and model filenames saved."
        ),
    )
    parser.add_argument(
        "--info_suffix",
        default=None,
        help=(
            "Additional string appended to the output plot and model filename"
            " to give additional identification apart from number of events"
        ),
    )
    args = parser.parse_args()

    train_and_save_model(args.batch_size, args.suffix, args.info_suffix)


if __name__ == "__main__":
    main()
