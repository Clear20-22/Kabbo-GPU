import argparse
import os
import numpy as np
import torch
import torch.nn as nn
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_model(model_path, device):
    if not os.path.exists(model_path):
        candidate = os.path.join(os.path.dirname(__file__), "..", "saved_models", os.path.basename(model_path))
        if os.path.exists(candidate):
            model_path = candidate
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = MLP().to(device)
    load_result = model.load_state_dict(state_dict)
    model.eval()
    return model, model_path, load_result


def generate_random_inputs(count):
    rng = np.random.default_rng(SEED + 1)
    return np.column_stack([
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
    ]).astype(np.float32)


def generate_edge_inputs(count):
    rng = np.random.default_rng(SEED + 2)
    X_edge = rng.choice([-5.0, 5.0], size=(count, 5)).astype(np.float32)
    X_edge += rng.normal(0.0, 0.2, size=(count, 5)).astype(np.float32)
    return np.clip(X_edge, -5.0, 5.0).astype(np.float32)


def metric_report(y_true, preds, min_target: float = 1e-3):
    pct_error = np.abs(preds - y_true) / np.maximum(np.abs(y_true), min_target) * 100.0
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


def print_metrics(name, stats):
    print(f"\n{name}")
    print("-------------------------------")
    print(f"MSE             : {stats['mse']:.8f}")
    print(f"RMSE            : {stats['rmse']:.8f}")
    print(f"MAE             : {stats['mae']:.8f}")
    print(f"R2              : {stats['r2']:.8f}")
    print(f"Mean pct error  : {stats['mean_pct_error']:.4f}%")
    print(f"Max pct error   : {stats['max_pct_error']:.4f}%")
    print(f"Within 1%       : {stats['within_1pct']:.2f}%")
    print(f"Within 5%       : {stats['within_5pct']:.2f}%")
    print(f"Within 10%      : {stats['within_10pct']:.2f}%")


def compare_stats(stats1, stats2):
    diff = {}
    for key in stats1.keys():
        if isinstance(stats1[key], float):
            diff[key] = stats2[key] - stats1[key]
        else:
            diff[key] = None
    return diff


def eval_model(model, x_scaler, y_scaler, X, y_true, device):
    X_scaled = x_scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()
    preds = y_scaler.inverse_transform(pred_norm)
    return metric_report(y_true, preds)


def get_hardest_samples(model, x_scaler, y_scaler, X, y_true, device, top_n, sort_by="pct_err"):
    X_scaled = x_scaler.transform(X)
    X_tensor = torch.from_numpy(X_scaled).float().to(device)
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()
    preds = y_scaler.inverse_transform(pred_norm).reshape(-1)
    y_true_flat = y_true.reshape(-1)
    abs_err = np.abs(preds - y_true_flat)
    pct_err = abs_err / np.maximum(np.abs(y_true_flat), 1e-3) * 100.0
    if sort_by == "pct_err":
        top_idx = np.argsort(-pct_err)[:top_n]
    else:
        top_idx = np.argsort(-abs_err)[:top_n]
    return [
        {
            "index": int(i),
            "x": X[i].tolist(),
            "y_true": float(y_true_flat[i]),
            "y_pred": float(preds[i]),
            "abs_err": float(abs_err[i]),
            "pct_err": float(pct_err[i]),
        }
        for i in top_idx
    ]


def print_hardest_samples(name, hardest):
    print(f"\n{name}")
    print("-------------------------------")
    if not hardest:
        print("No samples available.")
        return
    for sample in hardest:
        x_str = ", ".join(f"{v:.4f}" for v in sample["x"])
        print(
            f"#{sample['index']:d} | x=[{x_str}] | "
            f"y_true={sample['y_true']:.6f} | y_pred={sample['y_pred']:.6f} | "
            f"abs_err={sample['abs_err']:.6f} | pct_err={sample['pct_err']:.3f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare two MLP checkpoints on the same datasets.")
    parser.add_argument("--model1", default="MLP_best.pth", help="First model checkpoint")
    parser.add_argument("--model2", default="MLP_finetuned_best.pth", help="Second model checkpoint")
    parser.add_argument("--random", type=int, default=50000, help="Number of random comparison samples")
    parser.add_argument("--edge", type=int, default=5000, help="Number of edge comparison samples")
    parser.add_argument("--top", type=int, default=20, help="Number of hardest samples to print")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    x_scaler, y_scaler = build_original_scalers()

    model1, path1, load1 = load_model(args.model1, device)
    model2, path2, load2 = load_model(args.model2, device)

    print(f"\nModel 1: {path1}")
    print("Missing keys:", load1.missing_keys)
    print("Unexpected keys:", load1.unexpected_keys)
    print(f"Model 2: {path2}")
    print("Missing keys:", load2.missing_keys)
    print("Unexpected keys:", load2.unexpected_keys)

    X_random = generate_random_inputs(args.random)
    y_random = equation(X_random[:, 0], X_random[:, 1], X_random[:, 2], X_random[:, 3], X_random[:, 4]).reshape(-1, 1)
    X_edge = generate_edge_inputs(args.edge)
    y_edge = equation(X_edge[:, 0], X_edge[:, 1], X_edge[:, 2], X_edge[:, 3], X_edge[:, 4]).reshape(-1, 1)

    stats1_random = eval_model(model1, x_scaler, y_scaler, X_random, y_random, device)
    stats2_random = eval_model(model2, x_scaler, y_scaler, X_random, y_random, device)
    stats1_edge = eval_model(model1, x_scaler, y_scaler, X_edge, y_edge, device)
    stats2_edge = eval_model(model2, x_scaler, y_scaler, X_edge, y_edge, device)

    print_metrics("Model 1 random stats", stats1_random)
    print_metrics("Model 2 random stats", stats2_random)
    print_metrics("Model 1 edge stats", stats1_edge)
    print_metrics("Model 2 edge stats", stats2_edge)

    diff_random = compare_stats(stats1_random, stats2_random)
    diff_edge = compare_stats(stats1_edge, stats2_edge)
    print_metrics("Delta random (model2 - model1)", diff_random)
    print_metrics("Delta edge (model2 - model1)", diff_edge)

    hardest1_random = get_hardest_samples(model1, x_scaler, y_scaler, X_random, y_random, device, args.top, sort_by="pct_err")
    hardest2_random = get_hardest_samples(model2, x_scaler, y_scaler, X_random, y_random, device, args.top, sort_by="pct_err")
    hardest1_edge = get_hardest_samples(model1, x_scaler, y_scaler, X_edge, y_edge, device, args.top, sort_by="pct_err")
    hardest2_edge = get_hardest_samples(model2, x_scaler, y_scaler, X_edge, y_edge, device, args.top, sort_by="pct_err")

    print_hardest_samples("Model 1 random hardest samples", hardest1_random)
    print_hardest_samples("Model 2 random hardest samples", hardest2_random)
    print_hardest_samples("Model 1 edge hardest samples", hardest1_edge)
    print_hardest_samples("Model 2 edge hardest samples", hardest2_edge)

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
