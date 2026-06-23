import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_scalers():
    rng = np.random.default_rng(SEED)

    X = np.column_stack([
        rng.uniform(-5, 5, 1_000_000),
        rng.uniform(-5, 5, 1_000_000),
        rng.uniform(-5, 5, 1_000_000),
        rng.uniform(-5, 5, 1_000_000),
        rng.uniform(-5, 5, 1_000_000),
    ]).astype(np.float32)

    y = equation(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]).reshape(-1, 1).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SEED,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)

    return x_scaler, y_scaler


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def parse_input(text):
    tokens = [t for t in text.replace(",", " ").split() if t.strip()]
    if len(tokens) == 1:
        value = int(tokens[0])
        if value < 1 or value > 1000:
            raise ValueError("Enter a sample count between 1 and 1000.")
        return "sample", value

    if len(tokens) != 5:
        raise ValueError("Please enter exactly 5 numeric values, or a single integer sample count.")

    return "single", np.array([float(t) for t in tokens], dtype=np.float32).reshape(1, 5)


def generate_random_rows(count):
    rng = np.random.default_rng(SEED + 1)
    X = np.column_stack([
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
        rng.uniform(-5, 5, count),
    ]).astype(np.float32)
    return X


def describe_error_stats(pred, actual):
    errors = np.abs(pred - actual)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = np.where(
            np.abs(actual) > 1e-8,
            100.0 * errors / np.abs(actual),
            np.nan,
        )

    stats = {
        "count": len(errors),
        "mean_abs_error": float(np.mean(errors)),
        "median_abs_error": float(np.median(errors)),
        "max_abs_error": float(np.max(errors)),
        "min_abs_error": float(np.min(errors)),
        "std_abs_error": float(np.std(errors)),
        "mean_pct_error": float(np.nanmean(pct_errors)),
        "max_pct_error": float(np.nanmax(pct_errors[np.isfinite(pct_errors)])) if np.any(np.isfinite(pct_errors)) else float('nan'),
        "accuracy_1pct": float(np.nanmean(pct_errors[np.isfinite(pct_errors)] <= 1.0)) * 100.0 if np.any(np.isfinite(pct_errors)) else float('nan'),
        "accuracy_5pct": float(np.nanmean(pct_errors[np.isfinite(pct_errors)] <= 5.0)) * 100.0 if np.any(np.isfinite(pct_errors)) else float('nan'),
        "accuracy_10pct": float(np.nanmean(pct_errors[np.isfinite(pct_errors)] <= 10.0)) * 100.0 if np.any(np.isfinite(pct_errors)) else float('nan'),
        "r2": float(r2_score(actual.reshape(-1), pred.reshape(-1))) if len(errors) > 1 else float('nan'),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pct_errors": pct_errors,
    }
    return stats


def display_samples(X, preds, actuals, pct_errors, max_rows=20):
    rows = min(len(X), max_rows)
    print(f"\nShowing {rows} sample rows (max {max_rows}):")
    print("idx |      x1      x2      x3      x4      x5 |   actual   | predicted  | pct error")
    print("---+---------------------------------------+-----------+-----------+----------")
    for i in range(rows):
        print(
            f"{i+1:3d} | "
            f"{X[i,0]:8.4f} {X[i,1]:8.4f} {X[i,2]:8.4f} {X[i,3]:8.4f} {X[i,4]:8.4f} | "
            f"{actuals[i,0]:9.6f} | {preds[i,0]:9.6f} | {pct_errors[i,0]:8.4f}%"
        )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "MLP_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    device = get_device()
    x_scaler, y_scaler = build_scalers()

    model = MLP().to(device)
    state_dict = load_checkpoint(model_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input(
            "Enter 5 numbers separated by spaces or commas for one row,\n"
            "or enter a single integer (1-1000) to sample random rows: "
        )

    mode, payload = parse_input(user_input)

    if mode == "single":
        X_user = payload
        X_scaled = x_scaler.transform(X_user)

        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled).to(device)
            pred_norm = model(tensor).cpu().numpy()

        pred = y_scaler.inverse_transform(pred_norm)
        actual = equation(*X_user.flatten())
        abs_error = float(abs(pred[0, 0] - actual))
        pct_error = float(100.0 * abs_error / abs(actual)) if abs(actual) > 1e-8 else float('nan')
        r2 = float(r2_score(np.array([actual]), pred.reshape(-1)))

        print("\nUser input:", X_user.flatten().tolist())
        print(f"Predicted value: {pred[0,0]:.6f}")
        print(f"Actual value:    {actual:.6f}")
        print(f"Absolute error:  {abs_error:.6f}")
        print(f"Percentage error:{pct_error:.4f}%")
        print(f"R² score: {r2:.6f}")

    else:
        sample_count = payload
        X_random = generate_random_rows(sample_count)
        X_scaled = x_scaler.transform(X_random)

        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled).to(device)
            pred_norm = model(tensor).cpu().numpy()

        preds = y_scaler.inverse_transform(pred_norm)
        actuals = equation(
            X_random[:, 0],
            X_random[:, 1],
            X_random[:, 2],
            X_random[:, 3],
            X_random[:, 4],
        ).reshape(-1, 1)

        stats = describe_error_stats(preds, actuals)
        print(f"\nRandom sample count: {sample_count}")
        print(f"R² score: {stats['r2']:.6f}")
        print(f"Mean absolute error: {stats['mean_abs_error']:.6f}")
        print(f"Median absolute error: {stats['median_abs_error']:.6f}")
        print(f"RMSE: {stats['rmse']:.6f}")
        print(f"Max absolute error: {stats['max_abs_error']:.6f}")
        print(f"Mean percentage error: {stats['mean_pct_error']:.4f}%")
        print(f"Max percentage error: {stats['max_pct_error']:.4f}%")
        print(f"Accuracy within 1%: {stats['accuracy_1pct']:.2f}%")
        print(f"Accuracy within 5%: {stats['accuracy_5pct']:.2f}%")
        print(f"Accuracy within 10%: {stats['accuracy_10pct']:.2f}%")

        display_samples(X_random, preds, actuals, stats['pct_errors'], max_rows=20)


if __name__ == "__main__":
    main()
