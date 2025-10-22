"""
Evaluate a pretrained ToyTransformer model on the ToyDataset.

- loads model weights from ``toy_model.pt``;
- runs autoregressive inference on a fixed-size synthetic dataset;
- computes per-event residuals and Wasserstein distances;
- evaluates global KS and Wasserstein metrics;
- generates diagnostic histograms for residuals, per-event distances,
  token values, and number of tokens per event.

All plots are saved as PDF files in the same directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.stats import ks_2samp, wasserstein_distance

from pythiatransformer.toy.toy_model import ToyDataset, ToyTransformer

base_dir = Path(__file__).resolve().parent

# Update matplotlib.pyplot parameters.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def plot_1hist(
    data,
    residuals=False,
    wd=False,
    bins=100,
):
    """
    Plot one of: ``residuals | wd`` (one flag only must be ``True``).

    Parameters
    ----------
    data : list[float] | list[float]
        Data to plot. Expected shapes by mode: ``residuals``-> list of
        per-event relative differences | ``wd`` -> list of per-event
        Wassertein distances.
    suffix : str
        Tag appended to the output filename.
    residuals : bool
        Plot histogram of event-wise relative differences:
        ``(sum(targets) - sum(generated)) / sum(targets)``.
    wd : bool
        Plot histogram of per-event Wasserstein distances.
    bins : int
        Number of bins for the histogram.
    """
    modes = {"residuals": residuals, "wd": wd}
    if sum(bool(v) for v in modes.values()) != 1:
        raise ValueError("Select exactly one of: residuals | wd .")

    # Find the only True mode.
    mode = next(k for k, v in modes.items() if v)

    # Titles, axes labels, filenames.
    titles = {
        "residuals": "Distribution of residuals per event",
        "wd": "Distribution of Wasserstein distances per event",
    }
    xlabels = {
        "residuals": "Per-event residual",
        "wd": "Per-event Wasserstein distance",
    }
    base_names = {
        "residuals": "residuals_hist",
        "wd": "wd_hist",
    }

    # Create output dir & figure
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / f"{base_names[mode]}.pdf"
    plt.figure(figsize=(10, 6), dpi=1200)

    # Plot.
    plt.hist(
        data,
        bins=bins,
        color="lightgreen",
        edgecolor="black",
        alpha=0.7,
        log=True,
    )
    if mode == "residuals":
        plt.axvline(0, color="red", linestyle="--", label="Zero Error")

    plt.xlabel(xlabels[mode])
    plt.ylabel("Counts")
    plt.title(titles[mode])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()
    logger.info(f"histogram saved to {filename}")


def plot_value_hist(generated_tokens, target_tokens, bins=100):
    """
    Plot tokens value for all the particles from all events.

    Plot ``generated_tokens`` vs ``target_tokens``, on the same axes.

    Parameters
    ----------
    generated_tokens : numpy.ndarray
        All generated tokens values from all events concatenated.
    target_tokens : numpy.ndarray
        All target tokens values from all events concatenated.
    suffix : str
        Tag appended to the output filename.
    bins : int
        Number of bins for the histogram.

    """
    # Build a common set of bin edges based on the combined data.
    # This ensures both histograms share identical bin boundaries.
    lo = np.nanmin([np.nanmin(generated_tokens), np.nanmin(target_tokens)])
    hi = np.nanmax([np.nanmax(generated_tokens), np.nanmax(target_tokens)])
    rng = (lo, hi)
    bin_edges = np.histogram_bin_edges(
        np.concatenate([generated_tokens, target_tokens]),
        bins=bins,
        range=rng,
    ).tolist()

    # Create output dir.
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / "pt_hist.pdf"

    # Plot.
    plt.figure(figsize=(10, 6), dpi=1200)
    plt.hist(
        target_tokens,
        bins=bin_edges,
        histtype="step",
        color="steelblue",
        linewidth=2,
        label="Target particles",
        log=True,
    )
    plt.hist(
        generated_tokens,
        bins=bin_edges,
        histtype="stepfilled",
        color="coral",
        alpha=0.5,
        edgecolor="black",
        label="Generated particles",
        log=True,
    )
    plt.legend(fontsize=12)
    plt.xlabel("Token value")
    plt.ylabel("Counts")
    plt.title("Distribution of tokens values for all the events")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()
    logger.info(f"histogram saved to {filename}")


def plot_token_hist(generated_tokens_per_event, target_tokens_per_event):
    """
    Plot histogram of number of particles per event.

    Plot ``generated_tokens_per_event`` vs ``target_tokens_per_event``,
    on the same axes.

    Parameters
    ----------
    generated_tokens_per_event : list[int]
        Number of non-padded generated tokens per event.
    target_tokens_per_event : list[int]
        Number of non-padded target tokens per event.
    suffix : str
        Tag appended to the output filename.
    """
    # Define bin edges centered on integer values.
    gen = np.asarray(generated_tokens_per_event, dtype=int)
    tgt = np.asarray(target_tokens_per_event, dtype=int)
    lo = int(min(gen.min(), tgt.min()))
    hi = int(max(gen.max(), tgt.max()))
    bin_edges = np.arange(lo - 0.5, hi + 1.5, 1).tolist()

    # Create output dir.
    base_dir = Path(__file__).resolve().parent
    filename = base_dir / "token_hist.pdf"

    # Plot.
    plt.figure(figsize=(10, 6), dpi=1200)
    plt.hist(
        generated_tokens_per_event,
        bins=bin_edges,
        color="coral",
        alpha=0.6,
        edgecolor="black",
        linewidth=1.2,
        label="generated particles",
        log=True,
    )
    plt.hist(
        target_tokens_per_event,
        bins=bin_edges,
        color=None,
        edgecolor="black",
        alpha=0.4,
        log=True,
        label="target particles",
    )
    plt.legend()
    plt.xlabel("Per-event number of particles")
    plt.ylabel("Counts")
    plt.title("Distribution of number of particles per event")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()
    logger.info(f"histogram saved to {filename}")


def main():
    """Run inference on ToyDataset and log results."""
    # Set generation parameters.
    MAX_LEN = 10
    THRESHOLD = 0.5
    # Load pretrained model.
    model = ToyTransformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=MAX_LEN,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(f"{base_dir}/toy_model.pt", map_location=device)
    )
    model.to(device)
    model.eval()
    # Prepare deterministic test dataset.
    N_SAMPLES = 10000
    counter = 0
    testset = ToyDataset(n_samples=N_SAMPLES, max_len=MAX_LEN, seed=999)
    # Prepare deterministic test dataset.
    residuals = []
    target_tokens = []
    generated_tokens = []
    target_tokens_per_event = []
    generated_tokens_per_event = []
    wd_per_event = []
    for i in range(N_SAMPLES):
        x, y_true, _, _ = testset[i]
        x = x.to(device)
        with torch.no_grad():
            y_pred = model.generate(
                x.unsqueeze(0), max_len=MAX_LEN, stop_thresh=THRESHOLD
            )
        y_pred = y_pred.squeeze(0).cpu()
        sum_pred = y_pred.sum().item()
        residual = (x.item() - sum_pred) / x.item()
        if residual > -30:
            residuals.append(residual)

        generated_np = y_pred[y_pred > 0]
        target_np = y_true[y_true > 0]

        if len(generated_np) == 0 or len(target_np) == 0:
            wd_per_event.append(float("nan"))
        else:
            wd = wasserstein_distance(generated_np, target_np)
            wd_per_event.append(wd)

        generated_tokens.append(generated_np)
        target_tokens.append(target_np)

        target_tokens_per_event.append(int((y_true != 0).sum()))
        generated_tokens_per_event.append(int((y_pred != 0).sum()))
        counter += 1
        if counter % 1000 == 0:
            logger.info(f"{counter} events processed")

    generated_tokens = np.concatenate(generated_tokens)
    target_tokens = np.concatenate(target_tokens)
    wd_global = wasserstein_distance(generated_tokens, target_tokens)
    ks_stat, ks_p = ks_2samp(generated_tokens, target_tokens)

    plot_1hist(residuals, residuals=True)
    plot_1hist(wd_per_event, wd=True)
    plot_value_hist(generated_tokens, target_tokens)
    plot_token_hist(generated_tokens_per_event, target_tokens_per_event)
    logger.info(f"WD_global={wd_global}, KS_stat={ks_stat}, KS_pvalue={ks_p}")


if __name__ == "__main__":
    main()
