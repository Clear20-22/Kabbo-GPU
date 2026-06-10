import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


SEED = 42


def equation(x1, x2, x3, x4, x5):
    return (
        np.sin(5 * x1)
        + np.cos(5 * x2)
        + np.exp(-0.5 * np.abs(x3))
        + 0.3 * (x4 ** 2)
        + np.tanh(x1 * x5)
        + 0.5 * np.sin(x2 * x3)
    )


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
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class RegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_finetune_data():
    rng = np.random.default_rng(SEED)
    N_FINE = 300_000
    N_NORMAL = int(N_FINE * 0.8)
    N_EDGE = N_FINE - N_NORMAL

    X_normal = rng.uniform(-5, 5, size=(N_NORMAL, 5)).astype(np.float32)
    X_edge = rng.choice([-5.0, 5.0], size=(N_EDGE, 5)).astype(np.float32)
    X_edge += rng.normal(0.0, 0.25, size=(N_EDGE, 5)).astype(np.float32)
    X_edge = np.clip(X_edge, -5.0, 5.0).astype(np.float32)

    X = np.vstack([X_normal, X_edge])
    rng.shuffle(X)
    y = equation(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]).reshape(-1, 1).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_original_scalers():
    rng = np.random.default_rng(SEED)
    sample_count = 1_000_000
    X = np.column_stack([
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
    ]).astype(np.float32)
    y = equation(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]).reshape(-1, 1).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    return x_scaler, y_scaler


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint.get("model_state_dict", checkpoint)


def metric_report(y_true, preds):
    pct_error = np.abs(preds - y_true) / (np.abs(y_true) + 1e-8) * 100.0
    return {
        "mse": float(mean_squared_error(y_true, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
        "mean_pct_error": float(np.mean(pct_error)),
        "max_pct_error": float(np.max(pct_error)),
        "within_1pct": float(np.mean(pct_error <= 1.0)) * 100.0,
        "within_5pct": float(np.mean(pct_error <= 5.0)) * 100.0,
        "within_10pct": float(np.mean(pct_error <= 10.0)) * 100.0,
    }


def print_report(name, report):
    print(f"\n=== {name} ===")
    print(f"MSE        : {report['mse']:.8f}")
    print(f"RMSE       : {report['rmse']:.8f}")
    print(f"MAE        : {report['mae']:.8f}")
    print(f"R2         : {report['r2']:.8f}")
    print(f"Mean % err : {report['mean_pct_error']:.4f}%")
    print(f"Max % err  : {report['max_pct_error']:.4f}%")
    print(f"Within 1%  : {report['within_1pct']:.2f}%")
    print(f"Within 5%  : {report['within_5pct']:.2f}%")
    print(f"Within 10% : {report['within_10pct']:.2f}%")


def model_param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    abs_mean = np.mean([p.abs().mean().item() for p in model.parameters() if p.numel() > 0])
    norm = np.sqrt(sum((p.detach() ** 2).sum().item() for p in model.parameters()))
    return total, abs_mean, norm


def compute_loss(model, X, y, x_scaler, y_scaler, criterion, device):
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_scaled).float().to(device))
        loss = criterion(preds, torch.from_numpy(y_scaled).float().to(device))
    return loss.item()


def train_and_evaluate():
    device = get_device()
    print("Device:", device)

    original_x_scaler, original_y_scaler = build_original_scalers()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_finetune_data()
    print("Train samples:", len(X_train), "Val samples:", len(X_val), "Test samples:", len(X_test))

    X_train_scaled = original_x_scaler.transform(X_train)
    X_val_scaled = original_x_scaler.transform(X_val)
    X_test_scaled = original_x_scaler.transform(X_test)

    y_train_scaled = original_y_scaler.transform(y_train)
    y_val_scaled = original_y_scaler.transform(y_val)
    y_test_scaled = original_y_scaler.transform(y_test)

    train_loader = DataLoader(RegressionDataset(X_train_scaled, y_train_scaled), batch_size=8192, shuffle=True, pin_memory=True)
    val_loader = DataLoader(RegressionDataset(X_val_scaled, y_val_scaled), batch_size=8192, shuffle=False, pin_memory=True)

    model = MLP().to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "MLP_best.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "MLP_best.pth")
    state_dict = load_checkpoint(checkpoint_path, device)
    load_result = model.load_state_dict(state_dict)
    print("Loaded checkpoint:", checkpoint_path)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    total_params, abs_mean, norm = model_param_stats(model)
    print(f"Model params: {total_params}, mean abs {abs_mean:.6g}, norm {norm:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    print("Initial loaded model train loss:", compute_loss(model, X_train, y_train, original_x_scaler, original_y_scaler, criterion, device))
    print("Initial loaded model val loss:", compute_loss(model, X_val, y_val, original_x_scaler, original_y_scaler, criterion, device))

    best_val_loss = float("inf")
    best_path = os.path.join(os.path.dirname(__file__), "MLP_finetuned_best.pth")

    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d} | train {train_loss:.8f} | val {val_loss:.8f} | lr {optimizer.param_groups[0]['lr']:.8e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "val_loss": best_val_loss}, best_path)
            print("Saved best fine-tuned model ->", best_path)

    print("\nLoading best fine-tuned model for evaluation...")
    state_dict = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict))
    model.eval()

    def eval_set(name, X, y_true):
        X_scaled = original_x_scaler.transform(X)
        with torch.no_grad():
            preds_norm = model(torch.from_numpy(X_scaled).float().to(device)).cpu().numpy()
        preds = original_y_scaler.inverse_transform(preds_norm)
        print_report(name, metric_report(y_true, preds))

    print("\nEvaluating on held-out test set")
    eval_set("Random held-out test", X_test, y_test)

    rng = np.random.default_rng(SEED + 3)
    X_edge = np.clip(rng.choice([-5.0, 5.0], size=(20000, 5)).astype(np.float32) + rng.normal(0.0, 0.2, size=(20000, 5)).astype(np.float32), -5.0, 5.0)
    y_edge = equation(X_edge[:, 0], X_edge[:, 1], X_edge[:, 2], X_edge[:, 3], X_edge[:, 4]).reshape(-1, 1).astype(np.float32)
    eval_set("Edge / boundary test", X_edge, y_edge)

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    train_and_evaluate()
