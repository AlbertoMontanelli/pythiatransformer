"""Script to load a pretrained ToyTransformer model and run inference
on a ToyDataset test set.
- Loads model weights from 'toy_model.pt'.
- Uses a fixed max sequence length and stop threshold for generation.
- Runs inference on 20 samples from ToyDataset with fixed seed.
- Prints input scalar, true target sequence (masked), predicted
  sequence, and sum of predictions.
"""

from loguru import logger
import matplotlib.pyplot as plt
import torch

from toy_model import ToyTransformer, ToyDataset

def plot_residual_histogram(residuals, filename="toy_residuals.pdf"):
    """
    Function to plot histogram of relative residuals.

    Args:
      residuals (list): list of computed residuals, one per event;
      filename (str): name of the figure that will be saved.

    Returns:
      None
    """
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=5000, alpha=0.7, color='lightgreen', edgecolor='black', log=True)
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel("Residuals")
    plt.ylabel("Counts")
    plt.title("Distribution of Prediction Residuals")
    plt.grid(True)
    plt.legend()
    plt.xlim(-20, 1)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Residual histogram saved as {filename}")


if __name__ == "__main__":
    # Set generation parameters.
    max_len = 10
    threshold = 0.5
    # Load pretrained model.
    model = ToyTransformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=max_len
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("toy_model.pt", map_location=device))
    model.to(device)
    model.eval()
    # Prepare deterministic test dataset.
    n_samples = 100000
    testset = ToyDataset(n_samples=n_samples, max_len=max_len, seed=999)
    # Prepare deterministic test dataset.
    residuals = []
    for i in range(n_samples):
        x, y_true, mask, length = testset[i]
        x = x.to(device)
        with torch.no_grad():
            y_pred = model.generate(
                x.unsqueeze(0),
                max_len=max_len,
                stop_thresh=threshold
            )
        y_pred = y_pred.squeeze(0).cpu()
        pred_list = [f"{v:.4f}" for v in y_pred.tolist()]
        sum_pred = y_pred.sum().item()
        residual = (x.item()-sum_pred)/x.item()
        residuals.append(residual)

        logger.info(f"\nExample {i+1}")
        # logger.info(f"Input x = {x.item():.4f}")
        # logger.info(f"Target: {[f'{v:.4f}' for v in y_true[mask].tolist()]}")
        # logger.info(f"Prediction: {pred_list}")
        # logger.info(f"Sum of prediction: {sum_pred:.6f}")
        # logger.info(f"Residual: {residual}")
    plot_residual_histogram(residuals)
