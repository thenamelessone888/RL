# Phase 6, attempt 1: single-track baseline (human-warmstart TD3)

**Purpose**: verify the agent learns something on one track (Section 31, Phase 6).

## Reproducibility

- Date: 2026-09-19
- Git commit: `4896370b63f597efe23f552110f6053020308609`
- Config: `td3.config.TD3_HUMAN_WARMSTART_TRAINER` / `TD3_HUMAN_WARMSTART_AGENT`
  (`critic_warmup_steps=1000`, `policy_delay=2`, `tau=0.005`, `lr_actor=lr_critic=1e-3`)
- Warm-start actor: `TD3_LIDAR_SAC_4_LIDAR_pretrained_HUMAN_actor.tmod` (behavior-cloned
  from `TmrlData/dataset_human`, 5046 transitions)
- Track: `tmrl-test`
- Run artifacts: `C:\Users\sreci\TmrlData\experiments\TD3_CLEAN_HUMAN_WARMSTART\`

## Result: agent got WORSE, not better -- policy collapse immediately after warm-up

Per-round `return_train` / `episode_length_train` (chronological):

| Round | return_train | episode_length_train | critic_warmup | policy_updated |
|---|---|---|---|---|
| 0 | 37.39 | 485 | 1.0 (frozen) | 0.0 |
| 2 | 29.63 | 384.4 | 1.0 (frozen) | 0.0 |
| 3 | 24.99 | 321.4 | 1.0 (frozen) | 0.0 |
| 4 | 76.69 | 881.0 | 1.0 (frozen) | 0.0 |
| 5 | 76.69 | 881.0 | **0.0 (ended)** | **0.5 (actor now updating)** |
| 8 | 63.57 | 730.3 | 0.0 | 0.5 |
| 9 | 55.36 | 636.0 | 0.0 | 0.5 |
| 10 | **0.0** | **81.0** | 0.0 | 0.5 |
| 11-12 | 0.0 | 81.0 | 0.0 | 0.5 |

**While the actor was frozen** (critic-only warm-up), the BC actor drove genuinely
well: real, sustained episodes (up to 881 of a possible 1000 steps, i.e. ~44s),
positive returns in the 25-77 range. This independently confirms the BC actor is
NOT fundamentally broken -- the earlier live-diagnostics finding (constant ~-20%
left steer causing an instant wall crash) was evidently specific to that particular
starting observation/orientation, not universal; across many real resets it drives
adequately most of the time.

**Within ~5 actor gradient updates of the critic warm-up ending, performance
collapsed completely** to the exact same degenerate pattern seen with a fully
random-initialized actor (0 return, 81-step episodes = the reward function's own
`MIN_STEPS=70 + FAILURE_COUNTDOWN=10` stall-timeout). This was not a one-off blip:
3 consecutive post-collapse rounds all showed the identical 0.0/81.0 pattern.

## Root cause (diagnosis, not yet fixed)

This is a well-documented failure mode when fine-tuning a behavior-cloned policy
with off-policy actor-critic RL: `critic_warmup_steps=1000` was intended to let the
critics learn accurate Q-values under the frozen BC policy before the actor is
allowed to move -- but 1000 critic updates against a small, freshly-initialized
replay buffer (only ~1000-1500 samples accumulated by the time warm-up ended) is
evidently nowhere near enough for the twin critics to have an accurate value
estimate of the BC policy. Once the actor starts following `-Q(s, pi(s))`, it is
being pulled toward whatever the still-inaccurate critic *currently* thinks is
good -- which is not the same as what is actually good -- and the policy
collapses almost immediately (within ~1000 additional critic updates / a handful
of the delayed actor updates).

This matches the exact concern the project's own `critic_warmup_steps` mechanism
was designed to prevent (see `td3/agent.py`'s module comments), but the current
warm-up duration (1000 steps) is insufficient in practice, at least on this track
with this reward density.

## Candidate fixes for a future attempt (not yet implemented; needs its own testing)

1. **Much longer critic warm-up** (e.g. 5,000-20,000 steps) so the critic has a
   larger, more representative replay buffer before the actor moves.
2. **BC regularization term in the actor loss** (TD3+BC style: `actor_loss =
   -Q(s, pi(s)) + alpha * MSE(pi(s), a_human)`), so the actor is never allowed to
   drift arbitrarily far from the demonstrated behavior, only nudged by the RL
   signal.
3. **Lower initial actor learning rate** (or an LR warm-up schedule) immediately
   after `critic_warmup` ends, so early (likely noisy) policy gradients cause
   smaller steps.
4. **Smaller `policy_noise`/`noise_clip`** during this early post-warm-up phase,
   since target policy smoothing on top of an already-fragile actor may compound
   the instability.

None of these have been implemented yet -- this manifest documents the diagnosis
only. Implementing and re-testing one of these is the natural next step before
attempting a longer, multi-hour Phase 6 baseline run.
