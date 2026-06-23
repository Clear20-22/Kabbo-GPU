import numpy as np

SEED = 42


def edge_sample(noise_std: float = 0.25) -> np.ndarray:
    """Generate a random edge sample near the domain boundaries [-5, 5]."""
    rng = np.random.default_rng(SEED)
    base = rng.choice([-5.0, 5.0], size=(5,)).astype(np.float32)
    noise = rng.normal(0.0, noise_std, size=(5,)).astype(np.float32)
    return np.clip(base + noise, -5.0, 5.0).astype(np.float32)


def edge_batch(count: int = 10, noise_std: float = 0.25) -> np.ndarray:
    """Generate a batch of random edge samples."""
    rng = np.random.default_rng(SEED)
    base = rng.choice([-5.0, 5.0], size=(count, 5)).astype(np.float32)
    noise = rng.normal(0.0, noise_std, size=(count, 5)).astype(np.float32)
    return np.clip(base + noise, -5.0, 5.0).astype(np.float32)


if __name__ == "__main__":
    sample = edge_sample()
    print("Single edge sample:", sample)

    batch = edge_batch(count=10)
    print("\nBatch of 10 edge samples:")
    for row in batch:
        print(row)


