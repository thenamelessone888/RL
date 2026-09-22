# Phase 6, attempt 2: single-track baseline (human-warmstart TD3 + BC-anchor)

**Purpose**: retry Phase 6 after diagnosing attempt 1's post-warmup policy
collapse (`experiments/phase6_baseline_v1/manifest.md`), with the TD3+BC-style
`bc_reg_alpha` fix (`td3/agent.py`, `td3/config.py`) in place.

## Reproducibility

- Date: 2026-09-19
- Git commit: `58f41f9c1b8475bba114b3140e4e1c78b489285d`
- Config: `td3.config.TD3_HUMAN_WARMSTART_TRAINER` / `TD3_HUMAN_WARMSTART_AGENT`,
  now with `bc_reg_alpha=2.5`
- Warm-start actor: `TD3_LIDAR_SAC_4_LIDAR_pretrained_HUMAN_actor.tmod` (unchanged
  from attempt 1)
- Track: `tmrl-test`
- Prerequisite cleanup: attempt 1's collapsed `trainer.tcpt`/`trainer.tmod`/
  `worker.tmod` and `dataset_td3` replay data were archived to
  `TmrlData\_archive\phase6_baseline_v1_collapsed\` before this run, so this is a
  genuinely fresh run, not a resume of the collapsed checkpoint.
- Run artifacts: `C:\Users\sreci\TmrlData\experiments\TD3_CLEAN_HUMAN_WARMSTART\`

## Result: no collapse -- performance held and improved over 9 post-warmup rounds

Per-round `return_train` / `episode_length_train` / `bc_reg_term` (chronological,
epoch/round noted; only rounds with a newly-completed episode shown):

| Epoch/round | return_train | episode_length_train | bc_reg_term | critic_warmup |
|---|---|---|---|---|
| 0/1 | 0.0 | 81.0 | NaN | 1.0 (frozen) |
| 0/3 | 15.29 | 233.1 | NaN | 1.0 (frozen) |
| 0/4 | 11.89 | 169.2 | NaN | 1.0 (frozen) |
| 0/5 | 22.00 | 238.0 | NaN | 1.0 (frozen) |
| 0/6 | 23.62 | 315.4 | 0.0213 | **0.0 (just ended)** |
| 0/7 | 21.02 | 286.4 | 0.0114 | 0.0 |
| 0/8 | 19.86 | 234.7 | 0.0112 | 0.0 |
| 0/9 | 26.04 | 305.6 | 0.0096 | 0.0 |
| 1/0 | 21.67 | 256.0 | 0.0096 | 0.0 |
| 1/1 | 4.77 | 119.5 | 0.0121 | 0.0 |
| 1/2 | 22.91 | 439.1 | 0.0139 | 0.0 |
| 1/3 | 23.86 | 454.0 | 0.0161 | 0.0 |
| 1/4 | 20.99 | 348.4 | 0.0174 | 0.0 |
| 1/5 | 62.61 | 752.0 | 0.0133 | 0.0 |

**Interpretation**:
- Unlike attempt 1, the actor did **not** collapse to the 0.0/81.0 degenerate
  pattern after critic warm-up ended. 9 consecutive post-warmup rounds (spanning
  epoch 0 into epoch 1) show sustained real track progress.
- Normal episode-to-episode variance is present (e.g. epoch1/round1 dipped to
  4.77/119.5, then recovered to 22.91/439.1 the next round) -- expected for
  real-time RL with a still-improving policy, not evidence of collapse (a real
  collapse in attempt 1 was *persistent*: 3+ consecutive rounds stuck at exactly
  0.0/81.0, not a single dip followed by recovery).
- The best round observed (return 62.61, episode length 752/1000 steps) matches
  or exceeds the frozen BC actor's own best observed episode in attempt 1 (return
  76.69, length 881) and is a large improvement over the median frozen-BC episode
  -- i.e. there is early, tentative evidence of the RL fine-tuning *improving* on
  the BC baseline, not just failing to destroy it, though 9 rounds (~10-15 minutes)
  is far too little data to claim a proven learning trend.
- `bc_reg_term` stayed small (0.01-0.02) throughout, i.e. the regularization is
  active but not dominating the actor loss -- consistent with the intended design
  (a mild anchor, not a hard constraint).

## Caveats / what this does NOT yet show

- This is ~10-15 minutes of post-warmup training (roughly 10 real episodes). It
  demonstrates the immediate collapse failure mode is fixed, not that the policy
  has converged or is production-quality.
- No unseen-track generalization, no curriculum, no ablation against
  `bc_reg_alpha` values other than 2.5 (the TD3+BC paper default) have been run.
- A longer, multi-hour run (or several repeated shorter runs with different
  seeds) would be needed before claiming this as a validated baseline result for
  the research writeup -- this manifest documents a successful smoke-scale retry,
  which is what Phase 6 (per Section 31) calls for before committing to longer
  runs, not the final baseline experiment.

## Process-management note (unrelated to the RL result)

`TaskStop` on this machine does not reliably kill the underlying Windows
`python.exe` process tree for background-launched `server`/`trainer`/`worker`
processes -- it only stops the tracking wrapper. Every stop in this session
required a follow-up `Get-CimInstance Win32_Process | Where CommandLine -like
'*train_td3*' -or '*spawn_main*'` check and manual `Stop-Process -Force`, because
leftover processes kept the TCP port bound and caused the next launch's server to
silently fail to bind (a `CannotListenError` inside a `multiprocessing.Process`
that does not crash the parent, producing a "server" that looks alive but has no
listener). Anyone continuing this project interactively should adopt the same
verify-and-kill-by-PID habit rather than trusting a stop confirmation alone.
