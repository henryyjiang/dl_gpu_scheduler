import torch
import torch.nn as nn


class NeuralSchedulerModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()

        self.job_encoder = nn.Linear(5, d_model)
        self.gpu_encoder = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.scoring_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, job_feats, gpu_feats):
        """
        job_feats: shape (num_jobs, 5)
        gpu_feats: shape (num_gpus, 3)
        returns: score matrix shape (num_jobs, num_gpus)
        """

        job_emb = self.job_encoder(job_feats)
        gpu_emb = self.gpu_encoder(gpu_feats)

        tokens = torch.cat([job_emb, gpu_emb], dim=0)
        tokens = tokens.unsqueeze(0)

        encoded = self.transformer(tokens)
        encoded = encoded.squeeze(0)

        num_jobs = job_feats.shape[0]

        job_out = encoded[:num_jobs]
        gpu_out = encoded[num_jobs:]

        scores = []

        for j in range(num_jobs):
            repeated_job = job_out[j].unsqueeze(0).repeat(gpu_out.shape[0], 1)
            pair_features = torch.cat([repeated_job, gpu_out], dim=1)
            job_gpu_scores = self.scoring_head(pair_features).squeeze(-1)
            scores.append(job_gpu_scores)

        return torch.stack(scores, dim=0)