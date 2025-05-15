import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# === PARAMETRI ===
NUM_EVENTS = 1000
MAX_OUTPUT_PARTICLES = 6
LEPTON_CLASSES = {'e': 0, 'mu': 1, 'tau': 2}
NUM_CLASSES = len(LEPTON_CLASSES)
INPUT_DIM = 4  # 3 one-hot + 1 pT
OUTPUT_DIM = 4
HIDDEN_DIM = 32
EPOCHS = 1000
LR = 5e-4

# === GENERAZIONE DATI ===
def generate_input_events(num_events):
    inputs, total_pts = [], []
    for _ in range(num_events):
        lepton = random.choice(list(LEPTON_CLASSES))
        l_idx = LEPTON_CLASSES[lepton]
        one_hot = torch.nn.functional.one_hot(torch.tensor(l_idx), num_classes=NUM_CLASSES).float()
        pt1 = random.uniform(5.0, 50.0)
        pt2 = random.uniform(5.0, 50.0)
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
        pt_fractions = torch.rand(n_leptons)
        pt_fractions /= pt_fractions.sum()
        pt_vals = pt_fractions * total_pt

        counts = [0, 0, 0]
        remaining = n_leptons
        while remaining > 0:
            l_type = random.choice([0, 1, 2])
            if remaining >= 2:
                counts[l_type] += 2
                remaining -= 2

        particles = []
        i = 0
        for l_idx, count in enumerate(counts):
            for _ in range(count):
                one_hot = torch.nn.functional.one_hot(torch.tensor(l_idx), num_classes=NUM_CLASSES).float()
                pt = pt_vals[i].unsqueeze(0)
                particles.append(torch.cat([one_hot, pt]))
                i += 1

        pad = [torch.zeros(OUTPUT_DIM) for _ in range(MAX_OUTPUT_PARTICLES - n_leptons)]
        mask = [0] * n_leptons + [1] * (MAX_OUTPUT_PARTICLES - n_leptons)

        outputs.append(torch.stack(particles + pad))
        masks.append(torch.tensor(mask, dtype=torch.bool))

    return torch.stack(outputs), torch.stack(masks)

# === MODELLO ===
class ParticleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.output_proj = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.transformer = nn.Transformer(
            d_model=HIDDEN_DIM,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )

    def forward(self, src, tgt, src_mask, tgt_mask, attn_mask):
        src_proj = self.input_proj(src)
        tgt_proj = self.input_proj(tgt)
        out = self.transformer(
            src_proj, tgt_proj,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            tgt_mask=attn_mask
        )
        return self.output_proj(out)

# === LOSS MISTA ===
def mixed_loss(pred, target, mask):
    pred_cls = pred[:, :, :NUM_CLASSES]
    target_cls = target[:, :, :NUM_CLASSES]
    pred_pt = pred[:, :, NUM_CLASSES:]
    target_pt = target[:, :, NUM_CLASSES:]

    ce = nn.CrossEntropyLoss(reduction='none')
    target_idx = torch.argmax(target_cls, dim=-1)
    ce_loss = ce(pred_cls.transpose(1, 2), target_idx)
    mse_loss = nn.functional.mse_loss(pred_pt[~mask], target_pt[~mask])

    return ce_loss[~mask].mean() + mse_loss

# === TRAINING SETUP ===
inputs, total_pts = generate_input_events(NUM_EVENTS)
outputs, masks = generate_output_events(total_pts)

model = ParticleTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
attn_mask = nn.Transformer.generate_square_subsequent_mask(MAX_OUTPUT_PARTICLES).to(torch.float32)

# === TRAINING LOOP ===
losses = []
for epoch in range(EPOCHS):
    model.train()
    pred = model(inputs, outputs, None, masks, attn_mask)
    loss = mixed_loss(pred, outputs, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"finita epoca {epoch}")

# === PLOT ===
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# === ESEMPIO INFERENZA ===
model.eval()
with torch.no_grad():
    pred = model(inputs, outputs, None, masks, attn_mask)
    print("\n--- Evento 0 ---")
    print("Input:")
    print(inputs[0])
    print("Target:")
    print(outputs[0])
    print("Predizione:")
    print(pred[0])
