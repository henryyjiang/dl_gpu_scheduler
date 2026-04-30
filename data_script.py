import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(n_samples=50000, seed=42):
    np.random.seed(seed)

    gpu_mem_free = np.random.uniform(5, 60, n_samples)
    gpu_util     = np.random.uniform(0, 100, n_samples)
    queue_len    = np.random.randint(0, 11, n_samples)
    model_size   = np.random.uniform(2, 20, n_samples)
    batch_size   = np.random.randint(1, 33, n_samples)
    seq_len      = np.random.randint(64, 2049, n_samples)

    # latency model from assignment spec
    latency = (
        model_size * batch_size * seq_len * 0.0001
        + queue_len * 5
        + gpu_util * 0.2
        + np.maximum(0, model_size - gpu_mem_free) * 10
    )

    return pd.DataFrame({
        "gpu_mem_free": gpu_mem_free,
        "gpu_util":     gpu_util,
        "queue_len":    queue_len,
        "model_size":   model_size,
        "batch_size":   batch_size,
        "seq_len":      seq_len,
        "latency":      latency,
    })


def split_dataset(df, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def save_datasets(df, train_df, val_df, test_df, output_dir="gpu_dataset"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gpu_inference_dataset_full.csv", index=False)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    dataset = generate_dataset(n_samples=50000)
    train_df, val_df, test_df = split_dataset(dataset)
    save_datasets(dataset, train_df, val_df, test_df)
    print("Dataset generated and saved.")
    print(dataset.head())
