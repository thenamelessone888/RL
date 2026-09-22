# Phase 6, attempt 3: extended single-track baseline (human-warmstart TD3 + BC-anchor)

**Purpose**: extend attempt 2 (which only had ~10-15 minutes of post-warmup data)
with a longer run and, for the first time, structured metrics logging + a plot,
to properly assess "does the agent learn something" rather than eyeballing log
scrollback.

## Reproducibility

- Date: 2026-09-19
- Git commit: see next commit after this manifest (includes
  `scripts/log_to_metrics_csv.py`, `scripts/plot_results.py`)
- Config: unchanged from attempt 2 (`bc_reg_alpha=2.5`)
- This run **resumed attempt 2's checkpoint** (epoch 1) rather than starting
  over -- it is a direct continuation, not a separate seed.
- Duration: ~50 minutes combined with attempt 2 (attempt 2: ~10-15 min, this
  extension: ~35 min), 90 rounds of metrics captured in this extension alone
  (`results/metrics/phase6_baseline_v3_metrics.csv`)
- Plot: `results/plots/phase6_baseline_v3_training_curves.png` (committed
  alongside its CSV as this phase's key evidence, as an explicit exception to
  the usual `.gitignore` exclusion of generated results)

## Result: no collapse, but a noisy plateau -- not yet a clear learning trend

See the plot. Episode return over 90 rounds is **bimodal**:
- A recurring "floor" around return 8-10, episode length ~95-120 steps --
  notably higher than attempt 1's true collapse floor (return exactly 0.0,
  length exactly 81 -- the reward function's own stall-timeout firing with zero
  progress). This floor represents the car making *some* early progress before
  failing at what is very likely a specific, recurring point on the track
  (consistent episode lengths in a narrow ~95-120 band across many different
  rounds strongly suggests a fixed obstacle, not random variation).
- Recurring peaks of return 40-90, episode length 300-950 steps, appearing
  roughly every 5-10 rounds throughout the *entire* run (not concentrated
  early or late) -- the policy clearly *can* drive well past that point some of
  the time.
- `bc_reg_term` climbed gradually from ~0.010 to a plateau around 0.025-0.028
  and stayed there (not runaway) -- the actor has drifted a bounded, controlled
  amount from the frozen BC anchor and stabilized.
- `loss_critic` crept up mildly (0.0 to ~0.1) but stayed bounded; `loss_actor`
  stayed essentially flat (~-2.47 to -2.49) for the whole run.

**Conclusion**: the BC-anchor fix's core goal -- preventing catastrophic,
persistent collapse -- holds over an extended run (90 rounds, ~35 minutes,
continuing an already-successful checkpoint). However, this data does **not**
yet show unambiguous continued improvement: the bimodal pattern looks more like
a plateau than a trend, and 90 rounds is still a small sample for a real-time
robotic RL task. Per Section 30 of the research plan, this should be reported
exactly as that -- "no collapse, plateaued at a noisy but non-degenerate level"
-- not oversold as "the car learned to drive."

## Most actionable finding for future work

The tight, recurring ~95-120-step failure band is a concrete, investigable
lead: it suggests the policy has a specific weak point (likely a particular
turn or track feature) rather than a general driving deficiency. Before running
more undifferentiated training time, it would be more productive to:

1. Inspect a `bc-eval`/live trace of a "short" (~100-step) episode against a
   "long" (~500+ step) episode side by side (LIDAR/speed/action at the point of
   divergence) to identify what specifically differs.
2. Consider whether this points to a curriculum opportunity (Section 10) --
   e.g. if the failure point is a sharp turn, an easier/straighter practice
   track first might let the policy consolidate general driving before facing
   this specific obstacle.

Neither of these has been done yet; this manifest documents the extended
baseline run's results and this diagnosis only.
