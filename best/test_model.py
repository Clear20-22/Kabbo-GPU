import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_scalers(sample_count=200_000):
    rng = np.random.default_rng(SEED)
    X = np.column_stack([
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
        rng.uniform(-5, 5, sample_count),
    ]).astype(np.float32)
    y = equation(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]).reshape(-1, 1).astype(np.float32)

    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    return x_scaler, y_scaler


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = MLP().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_random_inputs(count, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
    ]).astype(np.float32)


def generate_edge_inputs(count, seed=SEED + 1, noise_std=0.15):
    rng = np.random.default_rng(seed)
    edges = rng.choice([-5.0, 5.0], size=(count, 5))
    noise = rng.normal(0.0, noise_std, size=(count, 5)).astype(np.float32)
    return np.clip(edges + noise, -5.0, 5.0).astype(np.float32)


def evaluate(X, y_true, model, x_scaler, y_scaler, device, top_n=20):
    X_scaled = x_scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled).float().to(device)

    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()

    preds = y_scaler.inverse_transform(pred_norm)
    errors = np.abs(preds - y_true)
    pct_errors = np.abs(errors / (np.abs(y_true) + 1e-8)) * 100.0

    metrics = {
        "count": len(y_true),
        "mse": float(mean_squared_error(y_true, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
        "mean_pct_error": float(np.nanmean(pct_errors)),
        "max_pct_error": float(np.nanmax(pct_errors)),
        "within_1pct": float(np.mean(pct_errors <= 1.0)) * 100.0,
        "within_5pct": float(np.mean(pct_errors <= 5.0)) * 100.0,
        "within_10pct": float(np.mean(pct_errors <= 10.0)) * 100.0,
        "top_errors": None,
    }

    order = np.argsort(-errors.reshape(-1))
    worst_idx = order[:top_n]
    top_errors = [
        {
            "index": int(i),
            "x": X[int(i)].tolist(),
            "actual": float(y_true[int(i), 0]),
            "predicted": float(preds[int(i), 0]),
            "abs_error": float(errors[int(i), 0]),
            "pct_error": float(pct_errors[int(i), 0]),
        }
        for i in worst_idx
    ]
    metrics["top_errors"] = top_errors
    return metrics


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Samples          : {metrics['count']}")
    print(f"R² score         : {metrics['r2']:.6f}")
    print(f"MSE              : {metrics['mse']:.8f}")
    print(f"RMSE             : {metrics['rmse']:.8f}")
    print(f"MAE              : {metrics['mae']:.8f}")
    print(f"Mean pct error   : {metrics['mean_pct_error']:.4f}%")
    print(f"Max pct error    : {metrics['max_pct_error']:.4f}%")
    print(f"Within 1%        : {metrics['within_1pct']:.2f}%")
    print(f"Within 5%        : {metrics['within_5pct']:.2f}%")
    print(f"Within 10%       : {metrics['within_10pct']:.2f}%")


def print_top_errors(metrics, label="Hardest samples"):
    print(f"\n{label} (top {len(metrics['top_errors'])})")
    print("idx | abs_error | pct_error | actual | predicted | x1 x2 x3 x4 x5")
    for item in metrics["top_errors"]:
        x_values = " ".join(f"{v:+.3f}" for v in item["x"])
        print(
            f"{item['index']:4d} | "
            f"{item['abs_error']:.6f} | "
            f"{item['pct_error']:.3f}% | "
            f"{item['actual']:+.6f} | "
            f"{item['predicted']:+.6f} | "
            f"{x_values}"
        )


def main():
    parser = argparse.ArgumentParser(description="Test MLP model and identify hard input cases.")
    parser.add_argument("--model", default="MLP_best.pth", help="Path to the saved model checkpoint.")
    parser.add_argument("--random", type=int, default=50000, help="Number of random uniform test samples.")
    parser.add_argument("--edge", type=int, default=5000, help="Number of edge/near-boundary test samples.")
    parser.add_argument("--top", type=int, default=15, help="Number of hardest samples to show.")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cwd, args.model)
    if not os.path.exists(model_path):
        alt = os.path.join(cwd, "..", "saved_models", os.path.basename(args.model))
        if os.path.exists(alt):
            model_path = alt

    device = get_device()
    print("Device:", device)
    print("Loading model from:", model_path)

    x_scaler, y_scaler = build_scalers(sample_count=200_000)
    model = load_model(model_path, device)

    X_random = generate_random_inputs(args.random)
    y_random = equation(X_random[:, 0], X_random[:, 1], X_random[:, 2], X_random[:, 3], X_random[:, 4]).reshape(-1, 1)
    random_metrics = evaluate(X_random, y_random, model, x_scaler, y_scaler, device, top_n=args.top)
    print_metrics("Random test set", random_metrics)
    print_top_errors(random_metrics, label="Hardest random test samples")

    X_edge = generate_edge_inputs(args.edge)
    y_edge = equation(X_edge[:, 0], X_edge[:, 1], X_edge[:, 2], X_edge[:, 3], X_edge[:, 4]).reshape(-1, 1)
    edge_metrics = evaluate(X_edge, y_edge, model, x_scaler, y_scaler, device, top_n=args.top)
    print_metrics("Edge / boundary test set", edge_metrics)
    print_top_errors(edge_metrics, label="Hardest edge test samples")

    print("\nDone.")


if __name__ == "__main__":
    main()
