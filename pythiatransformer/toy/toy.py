import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Dataset di esempio: input x float, target y_i liste di float la cui somma fa x
class ToyDataset(Dataset):
    def __init__(self, n_samples=10000, max_len=10, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.max_len = max_len
        self.data = []
        for _ in range(n_samples):
            x = random.uniform(0.0, 1.0) * 10  # input da 0 a 10
            k = random.randint(1, max_len)
            v = torch.rand(k)
            v = v / v.sum()
            y = v * x
            # padding a max_len con zeri
            y_pad = torch.zeros(max_len)
            mask = torch.zeros(max_len, dtype=torch.bool)
            y_pad[:k] = y
            mask[:k] = 1
            self.data.append((x, y_pad, mask, k))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y_pad, mask, length = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), y_pad, mask, length


# Modello Transformer per regressione sequenziale
class ToyTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=10,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.in_proj = nn.Linear(1, d_model)  # embedding del float in d_model
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))  # token SOS
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, 1)  # proiezione output y_t
        self.stop_head = nn.Linear(d_model, 1)  # head EOS

    def forward_teacher(self, x, y, mask, lengths):
        B, T = y.size()
        device = x.device
        src = self.in_proj(x.unsqueeze(-1)).unsqueeze(1)  # [B,1,d]
        sos = self.sos_token.expand(B, -1, -1)  # [B,1,d]
        tgt_emb = self.in_proj(y.unsqueeze(-1))  # [B,T,d]
        tgt = torch.cat([sos, tgt_emb], dim=1)  # [B,T+1,d]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T + 1).to(
            device
        )
        src_key_padding_mask = torch.zeros(
            B, 1, dtype=torch.bool, device=device
        )
        pad_dec = ~mask
        tgt_key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), pad_dec],
            dim=1,
        )
        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B,T+1,d]
        y_hat = self.out_proj(out).squeeze(-1)  # [B,T+1]
        stop_logits = self.stop_head(out).squeeze(-1)  # [B,T+1]
        return y_hat, stop_logits

    def generate(self, x, max_len=None, stop_thresh=0.5):
        """
        Inferenza autoregressiva con SOS e decisione EOS.
        Genera fino a max_len o finché p_stop > stop_thresh.
        """
        if max_len is None:
            max_len = self.max_len
        device = x.device
        B = x.size(0)
        src = self.in_proj(x.unsqueeze(-1)).unsqueeze(1)
        print(f"src shape: {src.shape}")
        src_key_padding_mask = torch.zeros(
            B, 1, dtype=torch.bool, device=device
        )
        tgt_emb = self.sos_token.expand(B, 1, self.d_model)
        generated = []
        for t in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                t + 1
            ).to(device)
            out = self.transformer(
                src=src,
                tgt=tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
            )  # [B,t+1,d]
            last = out[:, -1, :]
            y_t = self.out_proj(last).squeeze(-1)
            p_stop = torch.sigmoid(self.stop_head(last)).squeeze(-1)
            generated.append(y_t)
            # embedding per passo successivo
            y_emb = self.in_proj(y_t.unsqueeze(-1)).unsqueeze(1)
            tgt_emb = torch.cat([tgt_emb, y_emb], dim=1)
            if (p_stop > stop_thresh).all():
                break
        # ritorna la sequenza raw senza alcuna normalizzazione
        y_seq = torch.stack(generated, dim=1)
        return y_seq


if __name__ == "__main__":
    # Iperparametri
    n_samples = 5000
    max_len = 10
    batch_size = 64
    epochs = 20
    lr = 1e-3

    dataset = ToyDataset(n_samples=n_samples, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ToyTransformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=max_len,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y_pad, mask, length in loader:
            x, y_pad, mask = x.to(device), y_pad.to(device), mask.to(device)
            stop_target = torch.zeros(x.size(0), max_len + 1, device=device)
            for i, L in enumerate(length):
                stop_target[i, L] = 1.0
            optim.zero_grad()
            y_hat, stop_logits = model.forward_teacher(x, y_pad, mask, length)
            mse = mse_loss(y_hat[:, 1:] * mask.float(), y_pad)
            stop = bce_loss(stop_logits, stop_target)
            loss = mse + stop
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoca {ep+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "toy_transformer.pt")
    print("Modello salvato in toy_transformer.pt")
