# =========================================================
# FINETUNE EXISTING BEST MODEL (OPTIMIZED FOR RTX 3060)
# =========================================================

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# SPEED MODE (IMPORTANT CHANGE: float32)
# =========================================================

torch.set_default_dtype(torch.float32)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

# =========================================================
# EQUATION
# =========================================================

def equation(x1, x2, x3, x4, x5):
    return (
        np.sin(5 * x1)
        + np.cos(5 * x2)
        + np.exp(-0.5 * np.abs(x3))
        + 0.3 * (x4 ** 2)
        + np.tanh(x1 * x5)
        + 0.5 * np.sin(x2 * x3)
    )

# =========================================================
# DATASET (80% normal + 20% edge)
# =========================================================

N_FINE = 300_000
rng = np.random.default_rng(42)

N_NORMAL = int(N_FINE * 0.8)
N_EDGE = N_FINE - N_NORMAL

X_normal = np.column_stack([
    rng.uniform(-5, 5, (N_NORMAL, 5))
])

def edge_samples(n):
    return rng.choice([-5, 5], n) + rng.normal(0, 0.25, n)

X_edge = np.column_stack([
    edge_samples(N_EDGE),
    edge_samples(N_EDGE),
    edge_samples(N_EDGE),
    edge_samples(N_EDGE),
    edge_samples(N_EDGE),
])

X_fine = np.vstack([X_normal, X_edge]).astype(np.float32)
rng.shuffle(X_fine)

y_fine = equation(
    X_fine[:, 0],
    X_fine[:, 1],
    X_fine[:, 2],
    X_fine[:, 3],
    X_fine[:, 4]
).reshape(-1, 1).astype(np.float32)

print("Fine-tune dataset:", X_fine.shape)

# =========================================================
# USE OLD SCALERS (IMPORTANT)
# =========================================================

X_fine = x_scaler.transform(X_fine)
y_fine = y_scaler.transform(y_fine)

# =========================================================
# DATASET
# =========================================================

class FineTuneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================================================
# DATALOADER (RTX 3060 OPTIMIZED)
# =========================================================

fine_loader = DataLoader(
    FineTuneDataset(X_fine, y_fine),
    batch_size=8192,   # 🔥 IMPORTANT UPGRADE
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# =========================================================
# MODEL (UNCHANGED)
# =========================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 1024),
            nn.GELU(),

            nn.Linear(1024, 1024),
            nn.GELU(),

            nn.Linear(1024, 512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# LOAD MODEL
# =========================================================

model = MLP().to(device)

checkpoint = torch.load("./saved_models/MLP_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

print("Loaded existing best model")

# =========================================================
# LOSS FUNCTION
# =========================================================

def criterion(pred, target):
    mse = torch.mean((pred - target) ** 2)

    relative = torch.mean(
        torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    )

    return mse + 0.1 * relative

# =========================================================
# OPTIMIZER (ADJUSTED FOR LARGE BATCH)
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-6,   # 🔥 adjusted for batch 8192
    weight_decay=1e-7
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# =========================================================
# FINETUNING LOOP
# =========================================================

best_loss = float("inf")

for epoch in range(20):

    model.train()
    total_loss = 0.0

    for xb, yb in fine_loader:

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    total_loss /= len(fine_loader.dataset)
    scheduler.step(total_loss)

    print(
        f"Epoch {epoch+1:03d} | "
        f"LR {optimizer.param_groups[0]['lr']:.8e} | "
        f"Loss {total_loss:.8f}"
    )

    # SAVE BEST
    if total_loss < best_loss:
        best_loss = total_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss
            },
            "./saved_models/MLP_finetuned_best.pth"
        )

        print("Best finetuned model saved")

# =========================================================
# LOAD BEST MODEL
# =========================================================

best_checkpoint = torch.load(
    "./saved_models/MLP_finetuned_best.pth",
    map_location=device
)

model.load_state_dict(best_checkpoint["model_state_dict"])
print("Best epoch:", best_checkpoint["epoch"])

# =========================================================
# EVALUATION
# =========================================================

model.eval()

N_TEST = 100_000

X_test = np.column_stack([
    rng.uniform(-5, 5, (N_TEST, 5))
]).astype(np.float32)

y_true = equation(
    X_test[:, 0],
    X_test[:, 1],
    X_test[:, 2],
    X_test[:, 3],
    X_test[:, 4]
).reshape(-1, 1)

X_test_scaled = x_scaler.transform(X_test)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    preds = model(X_test_tensor).cpu().numpy()

preds = y_scaler.inverse_transform(preds)

# =========================================================
# METRICS
# =========================================================

mse = mean_squared_error(y_true, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, preds)
r2 = r2_score(y_true, preds)

relative_error = np.abs(preds - y_true) / (np.abs(y_true) + 1e-8)

within_1 = np.mean(relative_error < 0.01) * 100
within_5 = np.mean(relative_error < 0.05) * 100
within_10 = np.mean(relative_error < 0.10) * 100

print("\n" + "=" * 80)
print(f"MSE        : {mse:.8f}")
print(f"RMSE       : {rmse:.8f}")
print(f"MAE        : {mae:.8f}")
print(f"R2         : {r2:.8f}")
print(f"Within 1%  : {within_1:.2f}%")
print(f"Within 5%  : {within_5:.2f}%")
print(f"Within 10% : {within_10:.2f}%")
print("=" * 80)