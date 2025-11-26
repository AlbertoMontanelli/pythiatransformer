"""
Run inference with a trained Transformer to generate final particles.

This script loads a pretrained Transformer model and performs
autoregressive generation of stable final-state particles starting
from status-23 inputs. It can either run a new inference or reload
previously saved results to reproduce diagnostic plots.

Workflow
--------
- load preprocessed datasets and rebuild the Transformer architecture;
- load pretrained model weights;
- run autoregressive generation of target particle sequences;
- compute physics-level diagnostics (residuals, Wasserstein, KS tests);
- save all results and plots to ``data/`` and ``plots/`` directories.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from pythiatransformer.main import build_model
from pythiatransformer.pythia_generator import _dir_path_finder

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Update matplotlib.pyplot parameters.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def plot_1hist(
    data,
    suffix,
    residuals=False,
    wd=False,
    bins=100,
):
    """
    Plot one of: ``residuals | wd`` (one flag only must be ``True``).

    Parameters
    ----------
    data : list[float] | list[float]
        Data to plot. Expected shapes by mode: ``residuals`` -> list of
        per-event relative differences | ``wd`` -> list of per-event
        Wassertein distances;
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
        "wd": "Per-event Wasserstein distance [GeV]",
    }
    base_names = {
        "residuals": "residuals_hist",
        "wd": "wd_hist",
    }

    # Create output dir & figure
    plot_dir = _dir_path_finder(data=False)
    filename = plot_dir / f"{base_names[mode]}_{suffix}.pdf"
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


def plot_pt_hist(generated_tokens, target_tokens, suffix, bins=100):
    """
    Plot tokens `pT` for all the particles from all events.

    Plot ``generated_tokens`` vs ``target_tokens``, on the same axes.

    Parameters
    ----------
    generated_tokens : numpy.ndarray
        All generated tokens `pT` from all events concatenated.
    target_tokens : numpy.ndarray
        All target tokens `pT` from all events concatenated.
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
    plot_dir = _dir_path_finder(data=False)
    filename = plot_dir / f"pt_hist_{suffix}.pdf"

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
    plt.xlabel("$p_T$ [GeV]")
    plt.ylabel("Counts")
    plt.title("Distribution of particles $p_T$ for all the events")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()
    logger.info(f"histogram saved to {filename}")


def plot_token_hist(generated_tokens_per_event, target_tokens_per_event, suffix):
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
    plot_dir = _dir_path_finder(data=False)
    filename = plot_dir / f"token_hist_{suffix}.pdf"

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


def run_inference(batch_size, model_suffix):
    """
    Run autoregressive inference using a trained Transformer model.

    Rebuild the model architecture with the same configuration used
    during training, loads the pretrained weights, and performs
    autoregressive target generation.

    Parameters
    ----------
    model_path : str
        Path to the pretrained model checkpoint (.pt file).
    batch_size : int
        Batch size for dataloader reconstruction.
    model_suffix : str
        String appended to the ParticleTransformer trained model to be
        loaded.

    Returns
    -------
    results : tuple
        Tuple containing all computed quantities:
        residuals, Wasserstein distances, generated tokens,
        target tokens, global WD, KS statistics and per-event token
        lists.
    config : dict
        Dictionary with model configuration parameters.
    """
    data_dir = _dir_path_finder(data=True)
    model_path = data_dir / f"transformer_model_{model_suffix}.pt"
    # Identify the nr of events in the suffix as a string to correctly
    # load the dataset with build_model.
    match = re.search(r"transformer_model_(\d+)", str(model_path))
    if match:
        data_suffix = match.group(1)
    else:
        raise ValueError(f"Could not extract numeric suffix from {model_path}")
    model, _ = build_model(batch_size, data_suffix)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.device = device

    logger.info("Starting autoregressive inference")
    results = model.generate_targets()

    return results


def save_data(
    suffix,
    residual,
    wd_per_event,
    generated_tokens,
    target_tokens,
    wd_global,
    ks_stat,
    ks_pvalue,
    generated_tokens_per_event,
    target_tokens_per_event,
):
    """
    Save inference results and metadata to the ``data/`` directory.

    All computed results (residuals, Wasserstein distances, predicted
    and target tokens, etc.) are serialized to a single ``.pt`` file.

    Parameters
    ----------
    suffix : str
        String appended to the output filenames. Files are saved as
        ``data/results_<suffix>.pt`` and ``data/meta_<suffix>.json``.
    residuals : list[float]
        Event-wise relative difference between the total `pT` of target
        and generated particles. For each event:
        ``(sum(target) - sum(pred)) / sum(target)``.
    wd_per_event : list[float]
        Per-event Wasserstein distances between generated and target
        tokens distributions.
    generated_tokens : numpy.ndarray
        All generated tokens `pT` from all events concatenated.
    target_tokens : numpy.ndarray
        All target tokens `pT` from all events concatenated.
    wd_global : float
        Wasserstein distance between the global generated vs target
        tokens distributions (concatenated over events).
    ks_stat : float
        Kolmogorov-Smirnov test statistic between global generated vs
        target tokens distributions.
    ks_p : float
        Kolmogorov-Smirnov two-sided p-value.
    generated_tokens_per_event : list[int]
        Number of non-padded generated tokens per event.
    target_tokens_per_event : list[int]
        Number of non-padded target tokens per event.
    """
    data_dir = _dir_path_finder(data=True)
    dict = {
        "residual": residual,
        "wd_per_event": wd_per_event,
        "generated_tokens": generated_tokens,
        "target_tokens": target_tokens,
        "wd_global": wd_global,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "generated_tokens_per_event": generated_tokens_per_event,
        "target_tokens_per_event": target_tokens_per_event,
    }
    np.savez(data_dir / f"results_{suffix}.npz", **dict)
    logger.info(f"Saved data to {data_dir}/results_{suffix}.npz")


def load_data(suffix):
    """
    Load previously saved inference results.

    Parameters
    ----------
    suffix : str
        Suffix identifying which result files to load.

    Returns
    -------
    tuple
        Tuple containing all stored results:
        (residual, wd_per_event, generated_tokens, target_tokens,
        wd_global, ks_stat, ks_pvalue, generated_tokens_per_event,
        target_tokens_per_event).
    """
    data_dir = _dir_path_finder(data=True)
    filename = data_dir / f"results_{suffix}.npz"
    d = np.load(filename, allow_pickle=True)
    return (
        d["residual"],
        d["wd_per_event"],
        d["generated_tokens"],
        d["target_tokens"],
        d["wd_global"],
        d["ks_stat"],
        d["ks_pvalue"],
        d["generated_tokens_per_event"],
        d["target_tokens_per_event"],
    )


def make_plots(
    residuals,
    wd_per_event,
    generated_tokens,
    target_tokens,
    generated_tokens_per_event,
    target_tokens_per_event,
    suffix,
):
    """
    Generate and save summary plots of inference results.

    Create all diagnostic plots:

    - histogram of event-wise residuals between the total `pT` of
      target and generated particles;
    - histogram of Wasserstein distances per event;
    - histogram of transverse momentum (pT) distributions for generated
      particles and target particles.
    - histogram of number of generated and target particles for each
      event.

    Parameters
    ----------
    residuals : list[float]
        Event-wise relative difference between the total `pT` of target
        and generated particles. For each event:
        ``(sum(target) - sum(pred)) / sum(target)``.
    wd_per_event : list[float]
        Per-event Wasserstein distances between generated and target
        tokens distributions.
    generated_tokens : numpy.ndarray
        All generated tokens `pT` from all events concatenated.
    target_tokens : numpy.ndarray
        All target tokens `pT` from all events concatenated.
    generated_tokens_per_event : list[int]
        Number of non-padded generated tokens per event.
    target_tokens_per_event : list[int]
        Number of non-padded target tokens per event.
    suffix : str
        String appended to output filenames in ``plots/``.
    """
    plot_1hist(residuals, suffix, residuals=True)
    plot_1hist(wd_per_event, suffix, wd=True)
    plot_pt_hist(generated_tokens, target_tokens, suffix)
    plot_token_hist(
        generated_tokens_per_event,
        target_tokens_per_event,
        suffix,
    )


def main():
    """
    Control model inference and plotting modes.

    Depending on the selected mode, either:

    - rebuild and run the model to generate new inference results,
      saving outputs and metadata to ``data/``;
    - or load existing results and regenerate the plots.

    CLI Parameters
    --------------
    mode : {'infer', 'plot'}
        Operation mode. ``infer`` runs inference and saves results;
        ``plot`` loads saved results and regenerates plots.
    --suffix : str, required
        Suffix identifying both loaded model and output files.
    --batch : int, optional, default=64
        Batch size used to load datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["infer", "plot"],
        help="infer: run model and save results; plot: load results and plot",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="suffix identifying both loaded model and output files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="test set batch size"
    )
    args = parser.parse_args()

    if args.mode == "infer":
        (results) = run_inference(args.batch_size, args.suffix)
        (
            residual,
            wd_per_event,
            generated_tokens,
            target_tokens,
            wd_global,
            ks_stat,
            ks_pvalue,
            generated_tokens_per_event,
            target_tokens_per_event,
        ) = results

        save_data(
            args.suffix,
            residual,
            wd_per_event,
            generated_tokens,
            target_tokens,
            wd_global,
            ks_stat,
            ks_pvalue,
            generated_tokens_per_event,
            target_tokens_per_event,
        )
        logger.info(
            f"WD_global={wd_global} GeV, KS_stat={ks_stat}, KS_pvalue={ks_pvalue}"
        )
    else:
        (
            residual,
            wd_per_event,
            generated_tokens,
            target_tokens,
            wd_global,
            ks_stat,
            ks_pvalue,
            generated_tokens_per_event,
            target_tokens_per_event,
        ) = load_data(args.suffix)
        make_plots(
            residual,
            wd_per_event,
            generated_tokens,
            target_tokens,
            generated_tokens_per_event,
            target_tokens_per_event,
            args.suffix,
        )
        logger.info(
            f"WD_global={wd_global} GeV, KS_stat={ks_stat}, KS_pvalue={ks_pvalue}"
        )


if __name__ == "__main__":
    main()
