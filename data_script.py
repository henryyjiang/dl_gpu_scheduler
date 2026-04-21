import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(n_samples=50000, seed=42):
    np.random.seed(seed)

    # gpu_mem_free = random(5GB → 60GB),
    gpu_mem_free = np.random.uniform(5, 60, n_samples) # Check the use of uniform please, as it generates floats and not ints

    # gpu_util = random(0 → 100%)
    gpu_util = np.random.uniform(0, 100, n_samples) # Check the use of uniform please, as it generates floats and not ints

    # queue_len = random(0 → 10)
    queue_len = np.random.randint(0, 11, n_samples)

    # model_size = random(2GB → 20GB)
    model_size = np.random.uniform(2, 20, n_samples) # Check the use of uniform please, as it generates floats and not ints

    # batch_size = random(1 → 32)
    batch_size = np.random.randint(1, 33, n_samples)

    # seq_len = random(64 → 2048)
    seq_len = np.random.randint(64, 2049, n_samples)

    # Latency formula that approximates compute time, queue delay, GPU load, and memory pressure. (Given in pdf)
    latency = (
        model_size * batch_size * seq_len * 0.0001
        + queue_len * 5
        + gpu_util * 0.2
        + np.maximum(0, model_size - gpu_mem_free) * 10
    )

    df = pd.DataFrame({
        "gpu_mem_free": gpu_mem_free,
        "gpu_util": gpu_util,
        "queue_len": queue_len,
        "model_size": model_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "latency": latency
    })

    return df


def split_dataset(df, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


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