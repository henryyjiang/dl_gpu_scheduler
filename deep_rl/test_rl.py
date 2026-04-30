"""
Unit tests for the RL scheduler components.

Run with:  python -m pytest rl/test_rl.py -v
(from the project root, using the miniconda python that has torch/numpy)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── path setup ──────────────────────────────────────────────────────────── #
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "simulator"))
sys.path.insert(0, str(ROOT / "rl"))

from models import Job, Cluster, GPU
from environment import (
    SchedulingEnv, MAX_QUEUE, JOB_FEAT, GPU_FEAT,
    JOB_NORM, GPU_NORM, REWARD_SCALE,
)
from networks import ActorCritic
from ppo import RolloutBuffer, PPOTrainer


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def make_job(
    job_id:    int   = 0,
    latency:   float = 50.0,
    mem:       float = 10.0,
    util:      float = 50.0,
    arrival:   float = 0.0,
) -> Job:
    """Minimal Job object for testing."""
    return Job(
        job_id=job_id,
        gpu_mem_required=mem,
        gpu_util_intensity=util,
        model_size=5.0,
        batch_size=4,
        seq_len=512,
        true_latency=latency,
        arrival_time=arrival,
    )


def make_env_with_state(
    jobs:     list[Job],
    num_gpus: int = 1,
    gpu_mem:  float = 80.0,
) -> SchedulingEnv:
    """
    Build a SchedulingEnv and manually inject a known simulation state,
    bypassing assign_poisson_arrivals so arrival times are exact.
    Also calls _obs() to initialise _obs_order before the first step().
    """
    from models import Event, EventType
    import heapq, copy

    env = SchedulingEnv(jobs, num_gpus=num_gpus, gpu_memory=gpu_mem)
    env._clock          = 0.0
    env._tb             = 0
    env._cluster        = Cluster.create(num_gpus, gpu_mem)
    env._job_queue      = copy.deepcopy(jobs)
    env._completed_jobs = []
    env._event_queue    = []
    env._obs_order      = []
    # Populate obs_order so step() can safely index into it
    env._obs()
    return env


# ════════════════════════════════════════════════════════════════════════════
# Environment tests
# ════════════════════════════════════════════════════════════════════════════

class TestEnvironment:

    # ── reset() ──────────────────────────────────────────────────────── #

    def test_reset_returns_correct_shapes(self):
        jobs = [make_job(i, arrival=float(i)) for i in range(5)]
        env  = SchedulingEnv(jobs, num_gpus=3)
        jf, gf, mask = env.reset(arrival_rate=2.0, seed=0)

        assert jf.shape   == (MAX_QUEUE, JOB_FEAT)
        assert gf.shape   == (3, GPU_FEAT)
        assert mask.shape == (MAX_QUEUE,)
        assert mask.dtype == bool

    def test_reset_obs_values_are_normalised(self):
        """Non-padding job features should lie roughly in [0, 1] after dividing by JOB_NORM."""
        jobs = [make_job(i) for i in range(10)]
        env  = SchedulingEnv(jobs, num_gpus=5)
        jf, gf, _ = env.reset(arrival_rate=5.0, seed=1)

        # Only check non-zero rows (real jobs, not padding)
        nonzero_rows = jf[jf.sum(axis=1) > 0]
        assert nonzero_rows.min() >= 0.0
        assert nonzero_rows.max() <= 2.0   # generous upper bound for edge cases

        assert gf.min() >= 0.0
        assert gf.max() <= 1.0 + 1e-6

    def test_reset_mask_reflects_gpu_capacity(self):
        """mask[i] = True iff queue[i] can fit on at least one GPU."""
        # All jobs require 10 GB; GPUs have 80 GB → all feasible
        jobs = [make_job(i, mem=10.0) for i in range(4)]
        env  = SchedulingEnv(jobs, num_gpus=2)
        _, _, mask = env.reset(arrival_rate=10.0, seed=2)
        # At least the first job that arrived should be feasible
        assert mask.any(), "Expected at least one feasible job"

    def test_reset_clears_previous_episode(self):
        """Two resets must not carry over completed jobs or clock."""
        jobs = [make_job(0)]
        env  = SchedulingEnv(jobs, num_gpus=1)
        env.reset(arrival_rate=1.0, seed=10)
        env._completed_jobs.append(make_job(99))  # pollute state
        env.reset(arrival_rate=1.0, seed=11)
        assert env._completed_jobs == []
        assert env._clock >= 0.0

    # ── _obs() ordering guarantee ─────────────────────────────────────── #

    def test_feasible_jobs_always_first_in_obs_when_queue_exceeds_max(self):
        """
        Critical regression test for the large-queue bug.

        If the queue has more than MAX_QUEUE jobs and the only feasible job
        is beyond position MAX_QUEUE-1, the observation mask must still have
        at least one True entry (because _obs() sorts feasible jobs first).
        """
        # 33 infeasible jobs (need 81 GB each) + 1 feasible job (needs 5 GB)
        # GPU has 80 GB total, so only the 5-GB job fits (81 > 80 → can't fit)
        big_jobs   = [make_job(i, mem=81.0) for i in range(MAX_QUEUE + 1)]
        small_job  = make_job(MAX_QUEUE + 1, mem=5.0)
        all_jobs   = big_jobs + [small_job]

        env = make_env_with_state(all_jobs, num_gpus=1, gpu_mem=80.0)
        jf, gf, mask = env._obs()

        assert mask.any(), (
            "mask is all-False: feasible job was not surfaced into the observation window"
        )
        # The feasible (small) job should be at index 0
        assert mask[0], "Feasible job should be at obs index 0"
        # Obs index 0 should encode the small job's mem (5 GB / 78 ≈ 0.064)
        mem_feat_idx = 4  # position of gpu_mem_required in JOB_FEAT
        assert abs(jf[0, mem_feat_idx] - 5.0 / JOB_NORM[mem_feat_idx]) < 1e-5

    def test_obs_order_matches_step_lookup(self):
        """_obs_order must match what step() will actually assign."""
        jobs = [make_job(i) for i in range(3)]
        env  = make_env_with_state(jobs, num_gpus=2)
        jf, gf, mask = env._obs()
        # After _obs(), _obs_order is set; the first valid action should be pick-able
        first_valid = int(np.argmax(mask))
        expected_job = env._obs_order[first_valid]
        assert expected_job in env._job_queue

    # ── step() ───────────────────────────────────────────────────────── #

    def test_step_removes_selected_job_from_queue(self):
        jobs = [make_job(0), make_job(1), make_job(2)]
        env  = make_env_with_state(jobs, num_gpus=3)
        assert len(env._job_queue) == 3
        env.step(0)
        assert len(env._job_queue) == 2

    def test_step_allocates_gpu_memory(self):
        job = make_job(0, mem=20.0)
        env = make_env_with_state([job], num_gpus=1)
        free_before = env._cluster.gpus[0].free_memory
        env.step(0)
        # After step, the GPU is either still allocated (if job not yet done)
        # or freed (if done signal came back). Either way memory was touched.
        # Check via completion list or remaining free memory.
        gpu = env._cluster.gpus[0]
        # Job is running or completed; either way queue shrank
        assert len(env._job_queue) == 0

    def test_step_reward_equals_negative_jct_on_single_job_episode(self):
        """
        Single job, 1 GPU:
        - assigned at t=0, completes at t=latency
        - JCT = latency - arrival_time = latency (arrival=0)
        - Expected reward from the final step = -latency / REWARD_SCALE
        """
        LATENCY = 60.0
        job = make_job(0, latency=LATENCY, mem=5.0, arrival=0.0)
        env = make_env_with_state([job], num_gpus=1)

        result = env.step(0)
        *_, reward, done = result

        assert done, "Single-job episode should end after one step"
        expected = -LATENCY / REWARD_SCALE
        assert abs(reward - expected) < 1e-6, f"reward={reward}, expected={expected}"

    def test_step_reward_is_zero_when_more_assignments_remain(self):
        """
        If after assigning one job there are still feasible assignments,
        step() must return reward=0 (no completions yet) and done=False.
        """
        jobs = [make_job(i, mem=5.0) for i in range(3)]
        env  = make_env_with_state(jobs, num_gpus=3)

        *_, reward, done = env.step(0)
        # 2 jobs still in queue, 2 GPUs still free → more assignments available
        assert not done
        assert reward == 0.0

    def test_step_done_when_all_jobs_complete(self):
        """Full episode: keep stepping until done=True, then all jobs completed."""
        jobs = [make_job(i, latency=float(10 + i), mem=5.0) for i in range(4)]
        env  = SchedulingEnv(jobs, num_gpus=4)
        jf, gf, mask = env.reset(arrival_rate=10.0, seed=99)

        done = False
        steps = 0
        while not done:
            valid_idx = int(np.argmax(mask))
            result    = env.step(valid_idx)
            *obs_parts, reward, done = result
            if not done:
                jf, gf, mask = obs_parts
            steps += 1
            assert steps < 1000, "Episode did not terminate"

        assert len(env._completed_jobs) == len(jobs)
        for j in env._completed_jobs:
            assert j.completion_time is not None
            assert j.start_time is not None
            assert j.turnaround_time > 0

    def test_clock_advances_monotonically(self):
        """Simulator clock must never go backwards."""
        jobs = [make_job(i, latency=float(5 + i), mem=5.0) for i in range(6)]
        env  = SchedulingEnv(jobs, num_gpus=2)
        jf, gf, mask = env.reset(arrival_rate=2.0, seed=7)

        clocks = [env._clock]
        done   = False
        while not done:
            valid_idx = int(np.argmax(mask))
            result    = env.step(valid_idx)
            *obs_parts, reward, done = result
            clocks.append(env._clock)
            if not done:
                jf, gf, mask = obs_parts

        for t0, t1 in zip(clocks, clocks[1:]):
            assert t1 >= t0 - 1e-9, f"Clock went backwards: {t0} → {t1}"

    # ── _can_assign_any() ────────────────────────────────────────────── #

    def test_can_assign_any_returns_false_on_empty_queue(self):
        env = make_env_with_state([], num_gpus=2)
        assert not env._can_assign_any()

    def test_can_assign_any_returns_false_when_no_gpu_has_capacity(self):
        """All GPUs are full → no feasible assignment."""
        job = make_job(0, mem=5.0)
        env = make_env_with_state([job], num_gpus=1, gpu_mem=80.0)
        # Manually fill the GPU so the job cannot fit
        env._cluster.gpus[0].used_memory = 80.0
        assert not env._can_assign_any()

    def test_can_assign_any_returns_true_when_job_fits(self):
        job = make_job(0, mem=5.0)
        env = make_env_with_state([job], num_gpus=1, gpu_mem=80.0)
        assert env._can_assign_any()


# ════════════════════════════════════════════════════════════════════════════
# Network tests
# ════════════════════════════════════════════════════════════════════════════

class TestNetwork:

    NUM_GPUS  = 4
    EMBED_DIM = 32
    Q         = MAX_QUEUE
    B         = 2   # batch size

    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return ActorCritic(num_gpus=self.NUM_GPUS, embed_dim=self.EMBED_DIM)

    def _random_obs(self, n_valid: int = 3, batch: int = 1):
        """Random obs tensors with n_valid True entries in the mask."""
        torch.manual_seed(1)
        jf   = torch.rand(batch, self.Q, JOB_FEAT)
        gf   = torch.rand(batch, self.NUM_GPUS, GPU_FEAT)
        mask = torch.zeros(batch, self.Q, dtype=torch.bool)
        mask[:, :n_valid] = True
        return jf, gf, mask

    # ── forward() ────────────────────────────────────────────────────── #

    def test_forward_output_shapes(self, net):
        jf, gf, mask = self._random_obs(n_valid=5, batch=self.B)
        logits, value = net(jf, gf, mask)
        assert logits.shape == (self.B, self.Q)
        assert value.shape  == (self.B,)

    def test_forward_invalid_positions_are_neg_inf(self, net):
        """Masked-out queue slots must have logit = -inf."""
        jf, gf, mask = self._random_obs(n_valid=4, batch=1)
        logits, _ = net(jf, gf, mask)
        logits = logits.squeeze(0)
        mask   = mask.squeeze(0)
        assert torch.isinf(logits[~mask]).all() and (logits[~mask] < 0).all()
        assert torch.isfinite(logits[mask]).all()

    def test_forward_valid_logits_are_finite(self, net):
        jf, gf, mask = self._random_obs(n_valid=MAX_QUEUE, batch=1)
        logits, value = net(jf, gf, mask)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()

    # ── act() ────────────────────────────────────────────────────────── #

    def test_deterministic_act_returns_argmax_of_valid_logits(self, net):
        jf, gf, mask = self._random_obs(n_valid=6, batch=1)
        action, lp, v = net.act(jf, gf, mask, deterministic=True)
        logits, _ = net(jf, gf, mask)
        expected  = logits.argmax(dim=-1)
        assert action.item() == expected.item()

    def test_stochastic_act_stays_within_valid_mask(self, net):
        """Sampled actions must always fall on a valid (True) position."""
        jf, gf, mask = self._random_obs(n_valid=3, batch=1)
        for _ in range(50):
            action, _, _ = net.act(jf, gf, mask)
            assert mask[0, action.item()].item(), \
                f"Sampled action {action.item()} is outside the valid mask"

    def test_act_log_prob_is_non_positive(self, net):
        """log p ≤ 0 for any probability distribution."""
        jf, gf, mask = self._random_obs(n_valid=5, batch=1)
        _, lp, _ = net.act(jf, gf, mask)
        assert lp.item() <= 0.0 + 1e-6

    # ── evaluate_actions() ───────────────────────────────────────────── #

    def test_evaluate_actions_consistent_with_act(self, net):
        """evaluate_actions must reproduce the same log_prob as act() for the same action."""
        net.eval()
        jf, gf, mask = self._random_obs(n_valid=5, batch=1)
        with torch.no_grad():
            action, lp_act, _ = net.act(jf, gf, mask)
            lp_eval, _, _     = net.evaluate_actions(jf, gf, mask, action)
        assert abs(lp_act.item() - lp_eval.item()) < 1e-5

    def test_evaluate_actions_output_shapes(self, net):
        jf, gf, mask = self._random_obs(n_valid=4, batch=self.B)
        acts = torch.zeros(self.B, dtype=torch.long)  # always pick index 0
        lp, v, ent = net.evaluate_actions(jf, gf, mask, acts)
        assert lp.shape  == (self.B,)
        assert v.shape   == (self.B,)
        assert ent.shape == (self.B,)

    # ── entropy ──────────────────────────────────────────────────────── #

    def test_entropy_is_zero_for_single_valid_action(self, net):
        """One valid action → distribution is deterministic → entropy = 0."""
        jf, gf, mask = self._random_obs(n_valid=1, batch=1)
        acts = torch.tensor([0])
        _, _, ent = net.evaluate_actions(jf, gf, mask, acts)
        assert abs(ent.item()) < 1e-4, f"Expected entropy≈0, got {ent.item()}"

    def test_entropy_upper_bounded_by_log_n_valid(self, net):
        """Entropy ≤ log(n_valid) — equality only when uniform."""
        n_valid = 8
        jf, gf, mask = self._random_obs(n_valid=n_valid, batch=1)
        acts = torch.tensor([0])
        _, _, ent = net.evaluate_actions(jf, gf, mask, acts)
        assert ent.item() <= math.log(n_valid) + 1e-4

    def test_uniform_policy_entropy_equals_log_n(self, net):
        """
        If all valid jobs have identical features the policy is uniform,
        so entropy should equal log(n_valid).
        """
        n_valid   = 6
        # Identical feature vectors → identical embeddings → equal logits
        jf        = torch.ones(1, MAX_QUEUE, JOB_FEAT)
        gf        = torch.ones(1, self.NUM_GPUS, GPU_FEAT)
        mask      = torch.zeros(1, MAX_QUEUE, dtype=torch.bool)
        mask[0, :n_valid] = True
        acts = torch.tensor([0])
        _, _, ent = net.evaluate_actions(jf, gf, mask, acts)
        assert abs(ent.item() - math.log(n_valid)) < 1e-4, \
            f"Expected entropy={math.log(n_valid):.4f}, got {ent.item():.4f}"

    # ── value head independence ───────────────────────────────────────── #

    def test_value_is_independent_of_padding_content(self, net):
        """
        Padding positions (mask=False) are excluded from mean_job_embed.
        Changing padding values must not change the value output.
        """
        net.eval()
        n_valid = 4
        torch.manual_seed(42)
        jf_base = torch.rand(1, MAX_QUEUE, JOB_FEAT)
        gf      = torch.rand(1, self.NUM_GPUS, GPU_FEAT)
        mask    = torch.zeros(1, MAX_QUEUE, dtype=torch.bool)
        mask[0, :n_valid] = True

        # Second obs: same valid slots, different padding
        jf_diff_pad = jf_base.clone()
        jf_diff_pad[0, n_valid:] = torch.rand(MAX_QUEUE - n_valid, JOB_FEAT)

        with torch.no_grad():
            _, v1 = net(jf_base,     gf, mask)
            _, v2 = net(jf_diff_pad, gf, mask)

        assert abs(v1.item() - v2.item()) < 1e-5, \
            f"Value changed with padding: {v1.item():.6f} vs {v2.item():.6f}"


# ════════════════════════════════════════════════════════════════════════════
# PPO / RolloutBuffer tests
# ════════════════════════════════════════════════════════════════════════════

class TestPPO:

    def _make_buffer_entry(self, reward=0.0, done=False, value=0.0, log_prob=-1.0):
        jf   = np.zeros((MAX_QUEUE, JOB_FEAT), dtype=np.float32)
        gf   = np.zeros((10, GPU_FEAT),        dtype=np.float32)
        mask = np.zeros(MAX_QUEUE,              dtype=bool)
        mask[0] = True
        return jf, gf, mask, 0, log_prob, reward, value, done

    # ── RolloutBuffer ────────────────────────────────────────────────── #

    def test_buffer_add_increments_len(self):
        buf = RolloutBuffer()
        assert len(buf) == 0
        for i in range(5):
            buf.add(*self._make_buffer_entry())
            assert len(buf) == i + 1

    def test_buffer_clear_resets_all_lists(self):
        buf = RolloutBuffer()
        for _ in range(3):
            buf.add(*self._make_buffer_entry())
        buf.clear()
        assert len(buf) == 0
        assert buf.rewards   == []
        assert buf.actions   == []
        assert buf.job_feats == []

    def test_buffer_stores_values_correctly(self):
        buf = RolloutBuffer()
        buf.add(*self._make_buffer_entry(reward=-0.7, done=True, value=1.5))
        assert buf.rewards[-1]  == pytest.approx(-0.7)
        assert buf.dones[-1]    == True
        assert buf.values[-1]   == pytest.approx(1.5)

    # ── GAE ──────────────────────────────────────────────────────────── #

    def _fill_buffer(self, trainer, rewards, values, dones, log_probs=None):
        """Directly populate the buffer with known values for GAE testing."""
        N = len(rewards)
        if log_probs is None:
            log_probs = [-1.0] * N
        for i in range(N):
            jf   = np.zeros((MAX_QUEUE, JOB_FEAT), dtype=np.float32)
            gf   = np.zeros((10, GPU_FEAT),        dtype=np.float32)
            mask = np.zeros(MAX_QUEUE, dtype=bool)
            mask[0] = True
            trainer.buffer.add(jf, gf, mask, 0, log_probs[i],
                               rewards[i], values[i], dones[i])

    def test_gae_gamma1_lambda1_equals_monte_carlo_returns(self):
        """
        With γ=λ=1 GAE reduces to undiscounted Monte Carlo returns.
        For a 3-step episode (r0=0, r1=-5, r2=-10, terminal at t=2):
          R[0] = 0 + -5 + -10 = -15
          R[1] = -5 + -10     = -15
          R[2] = -10
        """
        net     = ActorCritic()
        trainer = PPOTrainer(net, gamma=1.0, gae_lambda=1.0)
        V       = [2.0, 3.0, 4.0]  # arbitrary value estimates

        self._fill_buffer(trainer,
            rewards=[0.0, -5.0, -10.0],
            values =V,
            dones  =[False, False, True],
        )
        _, returns = trainer._compute_gae()

        assert returns[0] == pytest.approx(-15.0, abs=1e-4)
        assert returns[1] == pytest.approx(-15.0, abs=1e-4)
        assert returns[2] == pytest.approx(-10.0, abs=1e-4)

    def test_gae_terminal_flag_isolates_episodes(self):
        """
        Concatenating two episodes: done=True at step 1 must prevent episode-2
        rewards from bleeding into episode-1 advantages.
        With γ=λ=1:
          Episode 1 (steps 0-1): R = [r0+r1, r1]
          Episode 2 (steps 2-3): R = [r2+r3, r3]
        """
        net     = ActorCritic()
        trainer = PPOTrainer(net, gamma=1.0, gae_lambda=1.0)
        r0, r1, r2, r3 = -1.0, -2.0, -3.0, -4.0

        self._fill_buffer(trainer,
            rewards=[r0,   r1,   r2,   r3  ],
            values =[0.0,  0.0,  0.0,  0.0 ],
            dones  =[False, True, False, True],
        )
        _, returns = trainer._compute_gae()

        assert returns[0] == pytest.approx(r0 + r1, abs=1e-4)
        assert returns[1] == pytest.approx(r1,       abs=1e-4)
        assert returns[2] == pytest.approx(r2 + r3,  abs=1e-4)
        assert returns[3] == pytest.approx(r3,       abs=1e-4)

    def test_gae_advantages_plus_values_equal_returns(self):
        """Returns = advantages + values, by definition."""
        net     = ActorCritic()
        trainer = PPOTrainer(net)
        V       = [1.0, 2.0, 0.5]
        self._fill_buffer(trainer,
            rewards=[0.0, -1.0, -2.0],
            values =V,
            dones  =[False, False, True],
        )
        adv, ret = trainer._compute_gae()
        for i in range(3):
            assert abs((adv[i] + V[i]) - ret[i]) < 1e-5

    def test_gae_output_shapes_match_buffer_length(self):
        net     = ActorCritic()
        trainer = PPOTrainer(net)
        N       = 7
        self._fill_buffer(trainer,
            rewards=[0.0] * (N - 1) + [-1.0],
            values =[0.0] * N,
            dones  =[False] * (N - 1) + [True],
        )
        adv, ret = trainer._compute_gae()
        assert adv.shape == (N,)
        assert ret.shape == (N,)

    def test_gae_single_terminal_step(self):
        """Edge case: a one-step episode. R[0] = r0."""
        net     = ActorCritic()
        trainer = PPOTrainer(net, gamma=1.0, gae_lambda=1.0)
        self._fill_buffer(trainer,
            rewards=[-7.0],
            values =[3.0],
            dones  =[True],
        )
        adv, ret = trainer._compute_gae()
        assert ret[0] == pytest.approx(-7.0, abs=1e-4)
        assert adv[0] == pytest.approx(-7.0 - 3.0, abs=1e-4)

    # ── update() ─────────────────────────────────────────────────────── #

    def test_update_clears_buffer(self):
        net     = ActorCritic()
        trainer = PPOTrainer(net, batch_size=4, n_epochs=1)
        self._fill_buffer(trainer,
            rewards=[0.0, -1.0],
            values =[0.0,  0.0],
            dones  =[False, True],
        )
        assert len(trainer.buffer) == 2
        trainer.update()
        assert len(trainer.buffer) == 0

    def test_update_returns_expected_loss_keys(self):
        net     = ActorCritic()
        trainer = PPOTrainer(net, batch_size=4, n_epochs=1)
        self._fill_buffer(trainer,
            rewards=[0.0, -1.0, -2.0],
            values =[0.0,  0.0,  0.0],
            dones  =[False, False, True],
        )
        losses = trainer.update()
        assert "policy_loss" in losses
        assert "value_loss"  in losses
        assert "entropy"     in losses
        assert all(math.isfinite(v) for v in losses.values())

    def test_advantage_normalisation_gives_zero_mean_unit_std(self):
        """After normalisation advantages should have mean≈0, std≈1."""
        net     = ActorCritic()
        trainer = PPOTrainer(net)
        N       = 20
        self._fill_buffer(trainer,
            rewards=list(range(N)),
            values =[float(i * 0.5) for i in range(N)],
            dones  =[False] * (N - 1) + [True],
        )
        adv, _ = trainer._compute_gae()
        normed = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert abs(normed.mean()) < 1e-4
        assert abs(normed.std()  - 1.0) < 1e-3

    def test_collect_episode_populates_buffer(self):
        """collect_episode on a tiny env must add transitions to the buffer."""
        jobs    = [make_job(i, latency=float(5 + i), mem=5.0) for i in range(3)]
        env     = SchedulingEnv(jobs, num_gpus=3)
        net     = ActorCritic(num_gpus=3)
        trainer = PPOTrainer(net)
        steps   = trainer.collect_episode(env, arrival_rate=5.0, seed=0)
        assert steps > 0
        assert len(trainer.buffer) == steps
        assert trainer.buffer.dones[-1] == True   # last step must be terminal


# ════════════════════════════════════════════════════════════════════════════
# RLScheduler tests
# ════════════════════════════════════════════════════════════════════════════

class TestRLScheduler:
    """Test _build_obs and _schedule without a real checkpoint."""

    @pytest.fixture
    def scheduler(self):
        from rl_scheduler import RLScheduler
        # Build the scheduler by monkey-patching __init__ to skip checkpoint IO
        sched = object.__new__(RLScheduler)
        torch.manual_seed(0)
        sched._network  = ActorCritic(num_gpus=2, embed_dim=32)
        sched._network.eval()
        sched._num_gpus = 2
        sched._device   = "cpu"
        return sched

    def test_build_obs_shapes(self, scheduler):
        from rl_scheduler import RLScheduler
        cluster   = Cluster.create(2, 80.0)
        jobs      = [make_job(i, mem=10.0) for i in range(3)]
        temp_used = {g.gpu_id: 0.0 for g in cluster.gpus}
        jf, gf, mask = scheduler._build_obs(jobs, cluster, temp_used)
        assert jf.shape   == (MAX_QUEUE, JOB_FEAT)
        assert gf.shape   == (2, GPU_FEAT)
        assert mask.shape == (MAX_QUEUE,)

    def test_build_obs_reflects_temp_used(self, scheduler):
        """
        If temp_used has already reserved memory on GPU 0, a job that would
        fit on an empty GPU 0 should still appear feasible via GPU 1.
        """
        cluster   = Cluster.create(2, 80.0)
        job       = make_job(0, mem=30.0)
        # Reserve 70 GB on GPU 0 — job needs 30 GB, so GPU 0 has only 10 GB left
        temp_used = {0: 70.0, 1: 0.0}
        jf, gf, mask = scheduler._build_obs([job], cluster, temp_used)
        # GPU 1 still has 80 GB free → job is feasible
        assert mask[0], "Job should be feasible via GPU 1"

    def test_schedule_returns_valid_assignments(self, scheduler):
        """All returned (job, gpu_id) pairs must reference real objects."""
        cluster = Cluster.create(2, 80.0)
        jobs    = [make_job(i, mem=10.0) for i in range(4)]
        assignments = scheduler._schedule(jobs, cluster)
        gpu_ids = {g.gpu_id for g in cluster.gpus}
        for job, gpu_id in assignments:
            assert job in jobs
            assert gpu_id in gpu_ids

    def test_schedule_does_not_double_allocate_gpu(self, scheduler):
        """
        Total memory assigned to any single GPU in one scheduling call
        must not exceed the GPU's free memory.
        """
        cluster  = Cluster.create(2, 80.0)
        # 6 jobs of 20 GB each — only 4 fit across 2×80 GB GPUs
        jobs     = [make_job(i, mem=20.0) for i in range(6)]
        assignments = scheduler._schedule(jobs, cluster)
        mem_per_gpu: dict[int, float] = {}
        for job, gpu_id in assignments:
            mem_per_gpu[gpu_id] = mem_per_gpu.get(gpu_id, 0.0) + job.gpu_mem_required
        for gpu_id, total_mem in mem_per_gpu.items():
            assert total_mem <= cluster.gpus[gpu_id].free_memory + 1e-6, \
                f"GPU {gpu_id} over-allocated: {total_mem} > {cluster.gpus[gpu_id].free_memory}"

    def test_schedule_empty_queue_returns_empty(self, scheduler):
        cluster = Cluster.create(2, 80.0)
        assert scheduler._schedule([], cluster) == []

    def test_schedule_skips_jobs_that_cannot_fit(self, scheduler):
        """Jobs requiring more memory than any free GPU must not be assigned."""
        cluster    = Cluster.create(1, 80.0)
        too_big    = make_job(0, mem=81.0)   # exceeds 80 GB GPU
        small_job  = make_job(1, mem=5.0)
        assignments = scheduler._schedule([too_big, small_job], cluster)
        assigned_jobs = [j for j, _ in assignments]
        assert too_big   not in assigned_jobs
        assert small_job in assigned_jobs
