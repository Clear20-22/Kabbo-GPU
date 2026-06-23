# =========================================================
# CELL 1 - IMPORTS
# =========================================================

import os
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from torch.utils.data import Dataset, DataLoader


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

os.makedirs("saved_models", exist_ok=True)

# Directory inside the workspace to save epoch checkpoints and CSV history
eq_dir = os.path.join("Assignment", "Equation")
os.makedirs(eq_dir, exist_ok=True)

# =========================================================
# CELL 2 - EQUATION
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
# CELL 3 - DATA GENERATION
# =========================================================

N = 1_000_000

rng = np.random.default_rng(SEED)

X = np.column_stack([
    rng.uniform(-5, 5, N),
    rng.uniform(-5, 5, N),
    rng.uniform(-5, 5, N),
    rng.uniform(-5, 5, N),
    rng.uniform(-5, 5, N),
]).astype(np.float32)

y = equation(
    X[:,0],
    X[:,1],
    X[:,2],
    X[:,3],
    X[:,4]
).reshape(-1, 1).astype(np.float32)

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# =========================================================
# CELL 4 - TRAIN VALID TEST SPLIT
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED
)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# =========================================================
# CELL 5 - SCALING
# =========================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)


# =========================================================
# CELL 6 - DATASET
# =========================================================

class RegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# CELL 7 - DATALOADER
# =========================================================

train_loader = DataLoader(
    RegressionDataset(X_train, y_train),
    batch_size=8192,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    RegressionDataset(X_val, y_val),
    batch_size=8192,
    pin_memory=True
)

test_loader = DataLoader(
    RegressionDataset(X_test, y_test),
    batch_size=8192,
    pin_memory=True
)


# =========================================================
# CELL 8 - ORIGINAL MLP
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

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================================================
# CELL 9 - RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(dim),

            nn.Linear(dim, dim * 4),
            nn.GELU(),

            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.block(x)


# =========================================================
# CELL 10 - RESMLP (NO FOURIER FEATURES)
# =========================================================

class ResMLP(nn.Module):

    def __init__(self):

        super().__init__()

        hidden = 512

        self.input = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU()
        )

        self.blocks = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),

            nn.Linear(hidden, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 1)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)


# =========================================================
# CELL 11 - TRAIN FUNCTION
# (UNCHANGED)
# =========================================================

def train_model(model, name):

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    best_val = float("inf")
    best_state = None

    patience = 40
    counter = 0
    # history for CSV logging
    history = []
    csv_path = os.path.join(eq_dir, f"{name}_history.csv")

    for epoch in range(1000):

        model.train()

        train_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)

            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss = criterion(pred, yb)

                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0:
            print(f"{name} | Epoch {epoch:03d} | LR {current_lr:.7f} | Train {train_loss:.8f} | Val {val_loss:.8f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

            # save best checkpoint to both saved_models and Equation folder
            ckpt = {
                "epoch": epoch,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
            }

            # overwrite single best file in Equation folder only
            torch.save(ckpt, os.path.join(eq_dir, f"{name}_best.pth"))

            counter = 0
        else:
            counter += 1

        # append metrics and write CSV (overwrites each epoch with full history)
        is_best = val_loss <= best_val
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "is_best": bool(is_best),
        })

        try:
            pd.DataFrame(history).to_csv(csv_path, index=False)
        except Exception:
            pass

        if counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    # final save to saved_models and Equation folder
    torch.save(
        model.state_dict(),
        f"saved_models/{name}.pth"
    )
    torch.save(
        model.state_dict(),
        os.path.join(eq_dir, f"{name}_final.pth")
    )

    return model


# =========================================================
# CELL 12 - EVALUATION
# =========================================================

def evaluate(model):

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for xb, yb in test_loader:

            xb = xb.to(device)

            pred = model(xb)

            preds.append(pred.cpu().numpy())
            trues.append(yb.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    preds = y_scaler.inverse_transform(preds)
    trues = y_scaler.inverse_transform(trues)

    r2 = r2_score(trues, preds)
    mae = mean_absolute_error(trues, preds)

    return r2, mae


# =========================================================
# CELL 13 - TRAIN MODELS
# =========================================================
# =========================================================
# CELL 13 - MAIN
# =========================================================

def main():

    models = {
        "MLP": MLP(),
        "ResMLP": ResMLP()
    }

    trained_models = {}
    results = []

    for name, model in models.items():

        print("\nTraining:", name)

        model = train_model(model, name)
        trained_models[name] = model

        r2, mae = evaluate(model)

        results.append([name, r2, mae])

        print(f"\n{name} Results")
        print(f"R2  : {r2:.8f}")
        print(f"MAE : {mae:.8f}")

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(
        results,
        columns=["Model", "R2", "MAE"]
    )

    results_df = results_df.sort_values(
        "R2",
        ascending=False
    )

    print(results_df)

    # =====================================================
    # PLOTS
    # =====================================================

    plt.figure(figsize=(6, 4))
    plt.bar(results_df["Model"], results_df["R2"])
    plt.title("R² Score")
    plt.ylabel("R²")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(results_df["Model"], results_df["MAE"])
    plt.title("MAE")
    plt.ylabel("MAE")
    plt.show()


if __name__ == "__main__":
    main()