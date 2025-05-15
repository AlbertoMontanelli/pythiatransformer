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
HIDDEN_DIM = 256
EPOCHS = 8000
LR = 2e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def print_pred_event(pred, evento_idx, num_total_classes):
    print(f"\n🔍 Predizioni per evento {evento_idx} (teacher forcing)")
    for i in range(pred.size(1)):
        pred_token = pred[evento_idx, i].detach().cpu().numpy()
        cls_logits = pred_token[:num_total_classes]
        pt_val = pred_token[-1]

        pred_id = cls_logits.argmax()
        cls_str = (
            ["e", "mu", "tau", "SOS", "EOS"][pred_id] if pred_id < 5 else "??"
        )

        print(
            f"🧩 Particella {i+1}: {cls_str:<4} | pT = {pt_val:.3f} | logits = {cls_logits.round(2)}"
        )
    print()


# === LOSS ===
def mixed_loss(pred, target, mask, alpha=20, beta=2.5):
    """
    Loss mista con:
    - CrossEntropy sugli ID (inclusi EOS),
    - MSE sul pT,
    - penalità extra per EOS predetti fuori posizione (beta),
    - rinforzo della loss sugli EOS corretti (alpha).
    """
    # Separazione classi e pT
    pred_cls = pred[:, :, :NUM_TOTAL_CLASSES]
    target_cls = target[:, :, :NUM_TOTAL_CLASSES]
    pred_pt = pred[:, :, -1]
    target_pt = target[:, :, -1]

    # Class index per ogni token
    target_idx = torch.argmax(target_cls, dim=-1)
    pred_idx = torch.argmax(pred_cls, dim=-1)

    # CrossEntropy standard (token-wise)
    ce = nn.CrossEntropyLoss(reduction="none")
    ce_loss = ce(pred_cls.transpose(1, 2), target_idx)

    # MSE solo sui pT validi
    mse_loss = nn.functional.mse_loss(pred_pt[~mask], target_pt[~mask])

    # Maschere EOS
    eos_mask_true = target_idx == EOS_TOKEN_IDX  # EOS nei target
    eos_mask_pred = pred_idx == EOS_TOKEN_IDX  # EOS predetti
    eos_correct = eos_mask_true & eos_mask_pred & (~mask)  # EOS giusti
    eos_wrong = eos_mask_pred & (~eos_mask_true) & (~mask)  # EOS sbagliati

    # Loss totale
    total_loss = ce_loss[~mask].mean() + mse_loss

    # Penalizza EOS fuori posto
    if eos_wrong.any():
        extra_eos_penalty = ce_loss[eos_wrong].mean()
        total_loss += beta * extra_eos_penalty

    # Rinforza EOS corretti
    if eos_correct.any():
        eos_correct_loss = ce_loss[eos_correct].mean()
        total_loss += alpha * eos_correct_loss

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
    log_eos_prediction(pred, targets, masks, EOS_TOKEN_IDX, NUM_TOTAL_CLASSES)
    print_pred_event(pred, evento_idx=0, num_total_classes=NUM_TOTAL_CLASSES)
    loss = mixed_loss(pred, targets, masks)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    print(f"[Epoca {epoch+1}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "toy_model_with_sos_eos.pt")
print("✅ Modello salvato in toy_model_with_sos_eos.pt")

# === PLOT ===
plt.figure()
plt.plot(losses)
plt.title("Training Loss (con SOS/EOS)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("toy_training_curve.pdf", dpi=1200)
plt.close()

# === CONFRONTO TRA TARGET E OUTPUT (teacher forcing) ===
model.eval()
with torch.no_grad():
    pred = model(inputs, decoder_inputs, None, masks, attn_mask)

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
