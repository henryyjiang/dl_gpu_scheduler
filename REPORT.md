# Deep RL GPU Scheduler — Architecture & Training Report

## Problem framing

The goal is to learn a scheduling policy that minimises average **Job Completion Time (JCT)** across a GPU cluster. JCT = `completion_time − arrival_time` for each job. There are no labelled "correct decisions" to learn from, which makes supervised learning inapplicable — hence deep RL.

The simulator is a discrete-event system: jobs arrive via a Poisson process, queue while GPUs are busy, then run for their true latency once assigned. A **scheduling decision** is any moment when at least one queued job can fit on at least one GPU.

---

## Environment (`rl/environment.py`)

### What a "step" is

At each decision point the agent picks **one job** from the visible queue. The environment assigns it to the best-fit GPU (most free memory), then fast-forwards time until the next decision point, accumulating rewards from any completions along the way.

```
Decision point → agent picks job j → assign j to GPU g
                                          ↓
                           more jobs fit right now?
                          yes ──────────────────→ return new obs, reward=0
                          no → advance event queue until next decision
                                  (collect −JCT/100 for each completion)
                                  → return new obs + accumulated reward
```

Keeping one assignment per step (rather than batching them all at once) gives the policy gradient a clear causal signal: which choice led to which reward.

### State representation

Each step the agent sees two fixed-size tensors:

| Tensor | Shape | Contents |
|---|---|---|
| `job_feats` | `[32, 6]` | Up to 32 queued jobs, each with 6 normalised features |
| `gpu_feats` | `[num_gpus, 3]` | Per-GPU: free memory, current utilisation, running job count |
| `mask` | `[32]` | `True` only for slots holding a real job that can fit somewhere |

Job features: `[model_size/70, batch_size/128, seq_len/4096, latency/300, mem_req/78, util/100]`. GPU features: `[free_mem/80, util/100, n_running/10]`. All divided by approximate realistic maxima so values live in roughly [0, 1].

### The large-queue fix

`_can_assign_any()` checks the whole queue, but the observation window only shows the first 32 positions. If the queue depth exceeds 32 and the only feasible job happens to be at position ≥ 32, the mask would be all-`False` — the `Categorical` distribution would receive all-`−∞` logits and crash. The fix: `_obs()` **sorts feasible jobs to the front** before slicing to 32, and stores the resulting ordering in `self._obs_order` so `step()` can correctly map the action index back to the right `Job` object.

### Reward

`reward = −JCT / 100` per completed job, collected at the step where the completion is observed. Dividing by 100 keeps rewards O(1) regardless of load level. Rewards are sparse — a job assigned at step *t* only generates a reward many steps later when it finishes — which is exactly what GAE is designed to handle.

---

## Network (`rl/networks.py`)

### Pointer-network attention

The key design question is: how do you produce a probability distribution over a **variable-length** queue? A standard softmax over a fixed-size output would need the network to "know" which positions are real vs padding. The pointer-network approach is cleaner:

```
job_feats  [B, Q, 6] ──→  JobEncoder     ──→  job_embeds   [B, Q, D]
gpu_feats  [B, G, 3] ──→  ClusterEncoder ──→  cluster_ctx  [B, D]

logits = (job_embeds @ cluster_ctx) / √D      [B, Q]
logits[~mask] = −1e9
π = softmax(logits)                            action distribution over queue
```

`cluster_ctx` acts as the **query** ("given the current cluster state, which job should I prioritise?") and `job_embeds` act as **keys**. The dot product scores each job relative to the cluster's needs, scaled by √D to prevent vanishing gradients with large embed_dim.

This structure is **permutation-equivariant**: reordering the queue changes which action index gets selected, but not the overall quality of the policy. That's the right inductive bias — a good scheduler shouldn't care whether the shortest job is at position 0 or position 5.

The mask fill uses `−1e9` rather than `−inf`. On MPS (Apple Silicon), `exp(−inf)` returns `NaN` in the backward pass through `logsumexp`, which corrupts the `Categorical` log-prob and entropy computations and propagates `NaN` gradients throughout the network. `−1e9` produces `exp(−1e9) ≈ 0` numerically, achieving the same masking effect without the NaN.

### Value head

```
value = MLP([cluster_ctx ‖ mean_job_embed])   scalar
```

`mean_job_embed` is the mean of embeddings for **valid** (mask=True) jobs only. The padding positions are zeroed out before averaging. This ensures the value estimate depends only on real pending work, not on how many empty slots the queue has.

### `LayerNorm` placement

LayerNorm is applied immediately after the first linear layer in each encoder, before ReLU. This stabilises early training when feature scales are mismatched (e.g. model_size in [0, 1] vs latency in [0, 1] after normalisation, but raw encoder weights are random).

---

## PPO (`rl/ppo.py`)

### Why PPO over DQN

DQN requires a fixed discrete action space (one output neuron per action), which would mean hard-coding a maximum queue length. PPO with the pointer-network naturally handles variable action spaces through masking. PPO is also more sample-efficient than vanilla policy gradient and more stable than DQN for problems with many local optima.

### Generalised Advantage Estimation (GAE)

The core credit-assignment mechanism:

```
δₜ = rₜ + γ · (1−doneₜ) · V(sₜ₊₁) − V(sₜ)       TD residual
Aₜ = δₜ + (γλ) · (1−doneₜ) · Aₜ₊₁                GAE recursion
```

- `λ=1` gives Monte Carlo returns (low bias, high variance)
- `λ=0` gives one-step TD (low variance, higher bias)
- `λ=0.95` interpolates between them

`γ=0.95` rather than the conventional 0.99: with 500-step episodes, `γ^500 ≈ 0.007` at γ=0.99, meaning early assignments get near-zero gradient signal. At γ=0.95 the effective horizon is ~60 steps (`0.95^60 ≈ 0.05`), which matches the typical lag between assigning a job and observing its completion.

The `done` flag correctly resets GAE at episode boundaries, preventing reward bleed-over between concatenated episodes.

### Clipped surrogate objective

```
L = −min(r·A,  clip(r, 1−ε, 1+ε)·A)  +  0.5·MSE(V, R)  −  0.001·H
```

where `r = π_new/π_old` is the probability ratio. The clip at `ε=0.2` prevents the policy from moving too far in one update. The entropy coefficient was set to 0.001 (down from the conventional 0.01) after observing that at extreme load (`λ=1.0`) the high-variance value estimates caused the entropy bonus to dominate the policy gradient, pushing the policy to become *more* random over training rather than less.

### Per-arrival-rate reward normalisation

`λ=0.5` and `λ=1.0` episodes produce reward magnitudes that differ by roughly 78×: at light load the queue rarely builds and total reward is near zero, at extreme load the accumulated JCT penalty can be hundreds of units. Without normalisation the value-loss scale shifts dramatically every episode depending on which arrival rate was sampled, making the value target non-stationary in a way the critic can't easily track.

The fix is a `RewardNormalizer` using **Welford online variance** keyed by arrival rate. Each reward is normalised as `(r − μ_λ) / σ_λ` before being stored in the rollout buffer. This keeps value-loss O(1) regardless of load level and removes the per-episode scale jump entirely.

### NaN gradient guard

During early training on MPS, a second failure mode can occur: even with the `−1e9` mask fix, a rare pathological batch can produce a near-infinite gradient norm before the weights have settled. `clip_grad_norm_` returns the pre-clip norm; if it is non-finite the `optimizer.step()` is skipped entirely:

```python
total_norm = nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
if torch.isfinite(total_norm):
    optimizer.step()
```

This prevents a single bad batch from corrupting the weights and triggering a cascade of NaN losses across subsequent episodes.

---

## RLScheduler (`rl/rl_scheduler.py`)

After training, the learned weights are wrapped as a `SchedulerInterface` so the model plugs directly into the original `Simulator` for fair comparison against FIFO and SJF under identical conditions.

The key difference from the training environment: the scheduler is called synchronously at each simulator event and must return **all assignments for that event at once** (not one per call). It handles this by looping — sort feasible jobs first, query the network for one job, assign it, decrement the `temp_used` memory ledger, repeat — until no more jobs can fit. The `temp_used` dict prevents double-allocating the same GPU slot within a single callback.

---

## Training (`rl/train.py`)

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `NUM_GPUS` | 10 | Matches evaluation cluster size |
| `EMBED_DIM` | 64 | Sufficient capacity; larger dims add cost without clear benefit |
| `TOTAL_EPISODES` | 500 | Enough for convergence at heavy/extreme load |
| `MAX_TRAIN_JOBS` | 5000 | Must be large enough that the queue reaches steady-state at λ=1.0 |
| `ARRIVAL_RATES` | [0.5, 1.0] | λ=0.25 dropped: queue never builds, entropy=0, zero gradient signal |
| `gamma` | 0.95 | Keeps meaningful credit signal within ~60 steps |
| `gae_lambda` | 0.95 | Bias-variance tradeoff between MC and TD |
| `ent_coef` | 0.001 | Prevents entropy bonus from overpowering noisy gradient at extreme load |
| `clip_eps` | 0.2 | Standard PPO trust-region width |
| `lr` | 3e-4 | Adam default |
| `batch_size` | 256 | Mini-batch size for PPO update |
| `n_epochs` | 4 | PPO update passes per rollout |

---

## Training dynamics

The full 500-episode run completed in approximately 3.7 hours on Apple Silicon MPS. Every episode ran to exactly 5,000 steps (the full `MAX_TRAIN_JOBS` job pool), so the step count per episode is constant and total training steps were 2.5 million. Arrival rate was sampled uniformly from `{0.5, 1.0}` each episode — 252 episodes at λ=0.5 and 248 at λ=1.0.

### Episode reward

Episode reward is the sum of Welford-normalised per-step rewards for the entire episode. Because normalisation is per-λ bucket, rewards from different arrival rates are directly comparable on the same axis.

| Phase | Episodes | Avg reward |
|---|---|---|
| Early | 1–125 | −43.8 |
| Mid | 126–250 | +32.2 |
| Late-A | 251–375 | +14.0 |
| Late-B | 376–500 | +27.6 |

The negative average in the early phase reflects an untrained policy making near-random assignments. The first positive reward appeared at episode 9, indicating the policy discovered basic memory-feasibility prioritisation very quickly. The large jump between the early and mid phases (+76 units) corresponds to the period when value-loss drops enough for GAE advantage estimates to become meaningful signal rather than noise.

The dip in Late-A (ep 251–375, avg +14.0) relative to Mid is driven by the composition of sampled arrival rates in that window: a higher-than-average proportion of λ=1.0 episodes, which are structurally harder — the queue saturates at ~1,900 jobs and the agent must work against a massive backlog. The 10-episode rolling mean reward reached its peak of **65.5** at episode ~498, indicating the policy was still improving at the end of training and that additional episodes would likely yield further gains.

Broken out by arrival rate:
- **λ=0.5**: average +19.9 per episode across all 252 episodes
- **λ=1.0**: average −5.1 per episode across all 248 episodes

The negative mean at λ=1.0 is expected and does not indicate divergence. At extreme overload the queue accumulates faster than it drains; the agent still receives negative reward proportional to JCT, and its Welford-normalised reward can be negative if its policy is worse than the running mean for that λ bucket.

### Value loss

The value network's MSE against GAE returns is the most direct indicator of how well the critic is calibrated. A well-calibrated critic is necessary for GAE to produce low-variance advantage estimates; a poorly calibrated one causes the policy gradient to point in the wrong direction.

| Phase | Avg value loss |
|---|---|
| ep 1–10 | 25.4 |
| ep 1–125 | 6.3 |
| ep 126–250 | 4.3 |
| ep 251–375 | 4.3 |
| ep 376–500 | 4.1 |
| Global min | 2.07 (ep 212) |

The initial value loss of 25.4 reflects a completely uncalibrated critic — at random initialisation the value head has no model of the reward scale, so its predictions can be far from the actual returns. It dropped below 12.7 within the first five episodes as the critic rapidly anchored to the magnitude of rewards.

The plateau around 4.0–4.3 from episode 126 onward is structurally expected rather than a sign of underfitting. Two factors keep value loss elevated above zero:

1. **Reward non-stationarity from arrival-rate switching.** Every time the episode flips from λ=0.5 to λ=1.0 (or back), the distribution of returns shifts. The critic must adapt its predictions to a different regime on the next episode, introducing a persistent recalibration overhead.

2. **Welford normalisation removes the scale.** Once rewards are normalised per-λ, the absolute reward magnitude no longer drifts — but the normalised returns still vary episode to episode based on how well the policy performed relative to its recent average. The critic is learning a moving target by construction.

The minimum of 2.07 at episode 212 represents the best single-episode fit the critic achieved and coincides with the checkpoint period (ep200, best saved checkpoint) when the policy had converged enough to produce stable rollouts without yet hitting the regime-switch overhead of the late training phase.

### Policy loss

The PPO clipped surrogate loss measures the tightness of the trust-region constraint. Its magnitude stayed very small throughout training, typically in the range −0.0001 to −0.0002, with a brief excursion to −0.0037 at episode 43.

A near-zero policy loss in PPO does **not** mean the policy is not learning. It means the clip is binding: the ratio `r = π_new/π_old` is frequently hitting the `1 ± 0.2` boundary, so the minimum in `min(r·A, clip(r)·A)` regularly selects the clipped term. This is the intended PPO regime — the gradient is present and the policy is updating, but updates are constrained to stay within the trust region. If the loss were deeply negative and growing, the policy would be taking large unconstrained steps, which typically leads to catastrophic forgetting.

The episode-43 excursion to −0.0037 coincides with early training instability: before the critic has calibrated, advantages can be large and noisy, producing occasionally large policy updates before the clip catches them. After episode ~100 the policy loss stabilised near zero and remained there for the rest of training.

### Entropy

Policy entropy measures how spread the action distribution is over the queue. High entropy (close to `log(32) ≈ 3.47` nats) means near-uniform assignment; zero entropy means deterministic.

| Stat | Value |
|---|---|
| Global mean | 1.41 nats |
| Global std | 1.31 nats |
| Minimum | 0.052 nats (ep 207) |
| Maximum | 2.801 nats (ep 302) |
| Late-training avg (ep 450–500) | 1.43 nats |

The high standard deviation is dominated by two events: the collapse to 0.052 at episode 207 and the spike to 2.801 at episode 302.

The **collapse at ep 207** is a local convergence: immediately after the value loss reached its global minimum (ep 212) the policy briefly converged to near-deterministic, having found a locally optimal assignment strategy for the queue states it was encountering. Near-zero entropy is dangerous because it prevents the policy from discovering better strategies and can cause the PPO ratio to diverge (a deterministic policy has zero probability on non-chosen actions, making `log π_new/π_old → −∞`). The small entropy bonus (`ent_coef=0.001`) was sufficient to escape this local minimum.

The **recovery to 2.801 at ep 302** is the rebound — after escaping the deterministic trap the policy briefly overexplored before settling. The fact that late-training entropy (1.43 nats) is marginally *higher* than early-training entropy (1.38 nats) confirms the policy did not collapse over the full run; the exploration bonus maintained action diversity throughout.

The 1.41 nats equilibrium represents approximately `e^1.41 ≈ 4.1` effective choices out of 32, meaning the policy was typically concentrating probability mass on 4–5 candidate jobs rather than spreading uniformly or picking deterministically. This is consistent with a policy that has learned to identify a small set of good candidates (e.g. jobs matching available GPU memory) but has not overfit to a rigid ordering rule.

---

## File map

```
GPU Scheduler/
├── simulator/            — Discrete-event simulator (Job, GPU, Cluster, Simulator)
│   ├── simulator.py
│   ├── schedulers.py     — FIFOScheduler, SJFScheduler (baselines)
│   └── data_loader.py
├── rl/
│   ├── environment.py    — SchedulingEnv: event-loop replay with step/reset API
│   ├── networks.py       — ActorCritic: pointer-network attention + value head
│   ├── ppo.py            — RolloutBuffer + PPOTrainer (GAE + clipped surrogate)
│   ├── train.py          — Training loop, evaluation table, checkpoint saving
│   ├── rl_scheduler.py   — RLScheduler: wraps trained model as SchedulerInterface
│   └── checkpoints/      — best_ep200_p95_128s.pt, final.pt, history.json
├── gpu_dataset/          — train.csv, val.csv, test.csv
├── evaluate.py           — Evaluation script; generates stats table + figures
└── analysis/             — Output subfolders: jct/, 500ep/, comparison/
```
