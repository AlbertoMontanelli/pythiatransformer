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

def plot_residual_histogram(model, dataset, device, max_len, stop_thresh, filename="residuals.pdf"):
    residuals = []
    for i in range(len(dataset)):
        x, _, _, _ = dataset[i]
        x = x.to(device)
        with torch.no_grad():
            y_pred = model.generate(
                x.unsqueeze(0),
                max_len=max_len,
                stop_thresh=stop_thresh
            )
        sum_pred = y_pred.squeeze(0).sum().item()
        residual = x.item() - sum_pred
        residuals.append(residual)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=40, density=False, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel("Residual (Input - Sum of Prediction)")
    plt.ylabel("Counts")
    plt.title("Distribution of Prediction Residuals")
    plt.grid(True)
    plt.legend()
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

        logger.info(f"\nExample {i+1}")
        # logger.info(f"Input x = {x.item():.4f}")
        # logger.info(f"Target: {[f'{v:.4f}' for v in y_true[mask].tolist()]}")
        # logger.info(f"Prediction: {pred_list}")
        # logger.info(f"Sum of prediction: {sum_pred:.6f}")
    plot_residual_histogram(model, testset, device, max_len, threshold)
