import torch
from toy import ToyTransformer, ToyDataset

if __name__ == "__main__":
    max_len = 10
    threshold = 0.5
    model = ToyTransformer(d_model=64, nhead=4,
                           num_encoder_layers=2, num_decoder_layers=2,
                           dim_feedforward=256, dropout=0.1,
                           max_len=max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("toy_transformer.pt", map_location=device))
    model.to(device)
    model.eval()

    testset = ToyDataset(n_samples=20, max_len=max_len, seed=999)

    for i in range(20):
        x, y_true, mask, length = testset[i]
        x = x.to(device)
        with torch.no_grad():
            y_pred = model.generate(x.unsqueeze(0), max_len=max_len, stop_thresh=threshold)
        y_pred = y_pred.squeeze(0).cpu()
        # stampo tutta la sequenza generata e la sua somma
        pred_list = [f"{v:.4f}" for v in y_pred.tolist()]
        sum_pred = y_pred.sum().item()
        print(f"\nEsempio {i+1}")
        print(f"Input x = {x.item():.4f}")
        print("Target reale:", [f"{v:.4f}" for v in y_true[mask].tolist()])
        print("Predizione:", pred_list)
        print(f"Somma predizioni: {sum_pred:.6f}")
        
