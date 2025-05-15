# Codice aggiornato con NUM_TOTAL_CLASSES usato correttamente
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# === PARAMETRI ===
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

NUM_EVENTS = 1000
MAX_OUTPUT_PARTICLES = 6
LEPTON_CLASSES = {"e": 0, "mu": 1, "tau": 2}
NUM_CLASSES = len(LEPTON_CLASSES)  # solo classi fisiche
NUM_SPECIAL_TOKENS = 2  # SOS + EOS
NUM_TOTAL_CLASSES = NUM_CLASSES + NUM_SPECIAL_TOKENS  # tutto incluso
SOS_TOKEN_IDX = NUM_CLASSES  # 3
EOS_TOKEN_IDX = NUM_CLASSES + 1  # 4

INPUT_DIM = NUM_TOTAL_CLASSES + 1  # one-hot + pT
OUTPUT_DIM = NUM_TOTAL_CLASSES + 1
HIDDEN_DIM = 128
EPOCHS = 4000
LR = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === GENERAZIONE DATI ===
def generate_input_events(num_events):
    inputs, total_pts = [], []
    for _ in range(num_events):
        lepton = random.choice(list(LEPTON_CLASSES))
        l_idx = LEPTON_CLASSES[lepton]
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(l_idx), num_classes=NUM_TOTAL_CLASSES
        ).float()
        pt1, pt2 = random.uniform(5, 50), random.uniform(5, 50)
        total_pt = pt1 + pt2
        p1 = torch.cat([one_hot, torch.tensor([pt1])])
        p2 = torch.cat([one_hot, torch.tensor([pt2])])
        inputs.append(torch.stack([p1, p2]))
        total_pts.append(total_pt)
    return torch.stack(inputs), total_pts


def generate_output_events(total_pts):
    outputs, masks = [], []
    for total_pt in total_pts:
        n_leptons = random.choice([2, 4, 6])
        pt_vals = torch.rand(n_leptons)
        pt_vals = pt_vals / pt_vals.sum() * total_pt

        counts = [0, 0, 0]
        rem = n_leptons
        while rem > 0:
            t = random.choice([0, 1, 2])
            if rem >= 2:
                counts[t] += 2
                rem -= 2

        particles = []
        i = 0
        for idx, count in enumerate(counts):
            for _ in range(count):
                one_hot = torch.nn.functional.one_hot(
                    torch.tensor(idx), num_classes=NUM_TOTAL_CLASSES
                ).float()
                particles.append(torch.cat([one_hot, pt_vals[i].unsqueeze(0)]))
                i += 1

        eos_vec = torch.zeros(OUTPUT_DIM)
        eos_vec[EOS_TOKEN_IDX] = 1.0
        particles.append(eos_vec)

        n_pad = MAX_OUTPUT_PARTICLES + 1 - len(particles)
        pad = [torch.zeros(OUTPUT_DIM) for _ in range(n_pad)]
        mask = [0] * len(particles) + [1] * n_pad

        outputs.append(torch.stack(particles + pad))
        masks.append(
            torch.tensor(mask[: MAX_OUTPUT_PARTICLES + 1], dtype=torch.bool)
        )

    return torch.stack(outputs), torch.stack(masks)


# === MODELLO ===
class ParticleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.output_proj = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.transformer = nn.Transformer(
            d_model=HIDDEN_DIM,
            nhead=16,
            num_encoder_layers=4,
            num_decoder_layers=4,
            batch_first=True,
        )

    def forward(self, src, tgt, src_mask, tgt_mask, attn_mask):
        src_proj = self.input_proj(src)
        tgt_proj = self.input_proj(tgt)
        out = self.transformer(
            src_proj,
            tgt_proj,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            tgt_mask=attn_mask,
        )
        return self.output_proj(out)


def log_eos_prediction(pred, target, mask, eos_token_idx, num_total_classes):
    """
    Logga statistiche sulla predizione dell'EOS durante teacher forcing.

    Args:
        pred: tensor [B, T, F] - output del modello
        target: tensor [B, T, F] - target vero (contenente EOS)
        mask: tensor [B, T] - padding mask (True = padding)
        eos_token_idx: int - indice del token EOS
        num_total_classes: int - classi totali (e, mu, tau, SOS, EOS)
    """
    with torch.no_grad():
        pred_ids = torch.argmax(pred[:, :, :num_total_classes], dim=-1)
        target_ids = torch.argmax(target[:, :, :num_total_classes], dim=-1)

        eos_pred = (pred_ids == eos_token_idx) & (~mask)
        eos_true = (target_ids == eos_token_idx) & (~mask)

        true_positive = (eos_pred & eos_true).sum().item()
        total_true = eos_true.sum().item()
        total_pred = eos_pred.sum().item()

        print("📊 EOS prediction stats:")
        print(f"✔️  EOS nei target:        {total_true}")
        print(f"🔍 EOS predetti:          {total_pred}")
        print(f"🎯 EOS corretti (match):  {true_positive}")
        if total_true > 0:
            print(f"🔁 Recall:    {true_positive / total_true:.2%}")
        if total_pred > 0:
            print(f"🔁 Precision: {true_positive / total_pred:.2%}")
        print()


# === LOSS ===
def mixed_loss(pred, target, mask, alpha=1):
    # Separazione componenti
    pred_cls = pred[:, :, :NUM_TOTAL_CLASSES]
    target_cls = target[:, :, :NUM_TOTAL_CLASSES]
    pred_pt = pred[:, :, -1]
    target_pt = target[:, :, -1]

    # ID target come indice
    target_idx = torch.argmax(target_cls, dim=-1)  # [B, T]

    # CrossEntropy
    ce = nn.CrossEntropyLoss(reduction="none")
    ce_loss = ce(pred_cls.transpose(1, 2), target_idx)  # [B, T]

    # MSE su pT (solo sui token validi, esclusi i padding)
    mse_loss = nn.functional.mse_loss(pred_pt[~mask], target_pt[~mask])

    # Separazione delle posizioni EOS
    eos_mask = target_idx == EOS_TOKEN_IDX

    # Loss finale
    total_loss = ce_loss[~mask].mean() + mse_loss
    if eos_mask.any():
        eos_loss = ce_loss[eos_mask].mean()
        print(f"[DEBUG] eos_loss: {eos_loss.item():.4f}, alpha: {alpha}")
        total_loss += alpha * eos_loss
    else:
        print("[DEBUG] nessun EOS trovato nei target")

    return total_loss


# === SHIFT DECODER INPUTS (aggiunge SOS all'inizio) ===
def shift_targets_with_sos(targets):
    B, T, F = targets.shape
    sos = torch.zeros(B, 1, F, device=targets.device)
    sos[:, 0, SOS_TOKEN_IDX] = 1.0
    return torch.cat([sos, targets[:, :-1, :]], dim=1)


# === GENERAZIONE ===
inputs, total_pts = generate_input_events(NUM_EVENTS)
targets, masks = generate_output_events(total_pts)
inputs, targets, masks = (
    inputs.to(device),
    targets.to(device),
    masks.to(device),
)
decoder_inputs = shift_targets_with_sos(targets)
attn_mask = nn.Transformer.generate_square_subsequent_mask(targets.size(1)).to(
    device
)

# === TRAINING ===
model = ParticleTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
losses = []
for epoch in range(EPOCHS):
    model.train()
    pred = model(inputs, decoder_inputs, None, masks, attn_mask)
    loss = mixed_loss(pred, targets, masks)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    print(f"[Epoca {epoch+1}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "toy_model_with_sos_eos.pt")
print("✅ Modello salvato in toy_model_with_sos_eos.pt")

# === PLOT ===
plt.plot(losses)
plt.title("Training Loss (con SOS/EOS)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# === CONFRONTO TRA TARGET E OUTPUT (teacher forcing) ===
model.eval()
with torch.no_grad():
    pred = model(inputs, decoder_inputs, None, masks, attn_mask)
    log_eos_prediction(pred, targets, masks, EOS_TOKEN_IDX, NUM_TOTAL_CLASSES)

evento_idx = 0
print(f"\n🔎 Confronto evento {evento_idx} (teacher forcing)\n")

for i in range(pred.size(1)):
    print(f"Particella {i+1}:")

    print("Target:")
    print(targets[evento_idx, i].cpu().numpy())

    print("Output predetto:")
    print(pred[evento_idx, i].cpu().numpy())

    if masks[evento_idx, i]:
        print("Padding\n")
    else:
        print()
