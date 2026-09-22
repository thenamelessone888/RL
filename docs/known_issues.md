# Known issues

## Deterministic policy gets stuck at a fixed point on `tmrl-test` (2026-09-19)

**Status**: root-caused, not yet fixed. Affects the current
`TD3_CLEAN_HUMAN_WARMSTART` checkpoint (post Phase 6 extended baseline,
`experiments/phase6_baseline_v3_extended/`). Full 6-episode trace:
`experiments/known_issue_stuck_at_wall/trained_deterministic_trace.csv`.

### Symptom

`python train_td3.py trained-eval --episodes 6 --max-steps 700` (deterministic,
`test=True`, no exploration noise) produces 6/6 near-identical short episodes:

| episode | max speed | at step | episode length | return |
|---|---|---|---|---|
| 0 | 45.0 | 77 | 101 | 9.44 |
| 1 | 46.3 | 82 | 98 | 9.83 |
| 2 | 46.3 | 81 | 96 | 9.89 |
| 3 | 46.2 | 81 | 96 | 10.01 |
| 4 | 45.7 | 80 | 96 | 9.63 |
| 5 | 46.3 | 81 | 98 | 9.99 |

Every episode: accelerate cleanly to ~45-46 km/h by step ~80, then decelerate to
near-zero and stay there until the reward function's stall-timeout ends the
episode (`MIN_STEPS=70 + FAILURE_COUNTDOWN=10`).

### Root cause (from `trained_deterministic_trace.csv`, episode 0, steps 81-100)

- `lidar_min == 0.0` on **every** step from 81 to 100 -- something is touching
  the car continuously.
- `lidar_mean`/`lidar_max` shrink monotonically (72 -> 9) -- the car's clear
  space keeps closing in, consistent with being wedged into a corner/wall.
- Actions during this window are inconsistent (raw gas swings 0.16-0.99, small
  and non-decisive steering) -- **no reverse, no deliberate escape turn**. The
  policy just keeps nudging forward into the obstacle until the episode times
  out.

This is a genuine capability gap, not a bug: the policy has no learned
recovery behavior for "stuck against a wall." It explains the Phase 6 extended
baseline's bimodal return distribution (`experiments/phase6_baseline_v3_extended/`)
completely -- during *training*, Gaussian exploration noise occasionally
perturbs the action enough to get the car past this exact point (producing the
observed 300-950-step "long" episodes), but the underlying deterministic
policy never learned to do this reliably itself.

### Why this makes sense given the data

The behavior-cloning warm-start actor was trained on `dataset_human`, i.e. a
single competent human driver's demonstrations. A competent driver does not
crash into this wall, so the dataset almost certainly contains no examples of
"how to recover once stuck against a wall" -- there is nothing for BC to
imitate here, and nothing in the reward function (pure forward-progress-along-
trajectory) specifically encourages learning a reverse-and-reorient recovery
maneuver either.

### Candidate next steps (not yet implemented; needs its own design decision)

1. **Reward shaping**: a small penalty for `lidar_min` staying near 0 for
   several consecutive steps (encourages the critic to value getting unstuck),
   or an explicit small reward for meaningful lidar-based clearance increase
   after a stuck state.
2. **Targeted demonstration**: record a short additional human demonstration
   clip specifically of recovering from being stuck at this exact point on
   `tmrl-test`, and include it in behavior-cloning pretraining.
3. **Increased exploration noise locally**: since noise is what currently
   rescues some episodes, a state-dependent exploration boost when `lidar_min`
   stays at 0 for N steps could increase the rate of successful recoveries
   during training, giving TD3 more (s, a, r) recovery examples to learn from.
4. **Curriculum**: if this obstacle is a specific sharp turn/wall early on
   `tmrl-test`, consider whether an easier practice track without this exact
   feature would let the policy consolidate general driving skill first
   (Section 10 of the research plan), returning to `tmrl-test` afterward.

None of these have been implemented. This is a genuine design decision (which
approach best fits the project's reward/curriculum philosophy) rather than a
one-line fix, and per Section 12/14 of the research plan should not be picked
speculatively without deciding which mechanism the project wants to rely on.
