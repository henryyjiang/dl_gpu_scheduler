import torch
import torch.nn as nn

from model.neural_scheduler_model import NeuralSchedulerModel


def train_neural_scheduler_model(
    dataset,
    epochs=15,
    learning_rate=1e-4,
    d_model=128,
    nhead=4,
    num_layers=4,
):
    model = NeuralSchedulerModel(d_model=d_model, nhead=nhead, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        loss_steps = 0

        for sample in dataset:
            job_feats = sample["job_feats"].clone().float()
            gpu_feats = sample["gpu_feats"].clone().float()
            labels = sample["labels"].float()

            # Normalize job features.
            job_feats[:, 0] /= 20.0
            job_feats[:, 1] /= 32.0
            job_feats[:, 2] /= 2048.0
            job_feats[:, 3] /= 100.0
            job_feats[:, 4] /= 80.0

            # Normalize GPU features.
            gpu_feats[:, 0] /= 120.0
            gpu_feats[:, 1] /= 100.0
            gpu_feats[:, 2] /= 10.0

            valid_jobs = labels.sum(dim=1) > 0
            if valid_jobs.sum().item() == 0:
                continue

            target_gpus = labels[valid_jobs].argmax(dim=1).long()

            scores = model(job_feats, gpu_feats)
            valid_scores = scores[valid_jobs]

            loss = criterion(valid_scores, target_gpus)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loss_steps += 1

        avg_loss = total_loss / loss_steps if loss_steps > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs} | avg_loss: {avg_loss:.6f}")

    return model
