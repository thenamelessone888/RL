# Phase 5 smoke test: TD3 + TMRL gradient-update verification

**Purpose**: prove `Server` + `Trainer` + `RolloutWorker` collect real live TrackMania
data and produce actual TD3 gradient updates, before committing to any longer
baseline run (Section 31, Phase 5).

## Reproducibility

- Date: 2026-09-19
- Git commit: `b08399291a7ffd5b5486c81bc2a2c7ae8dfd6785`
- Algorithm: TD3 (`td3/`), plain init (no human warm-start), config: `td3.config.TD3_TRAINER` / `TD3_AGENT`
- TMRL: 0.7.1, Python 3.12.6, PyTorch 2.14.0+cu126, CUDA available
- Track: `tmrl-test` (config.json `RUN_NAME=SAC_4_LIDAR_pretrained`, `RTGYM_INTERFACE=TM20LIDAR`)
- Run artifacts: `C:\Users\sreci\TmrlData\experiments\TD3_CLEAN_RANDOM\` (worker.tmod, trainer.tmod, trainer.tcpt), `C:\Users\sreci\TmrlData\dataset_td3\`
- Duration: ~50s wall-clock (server/trainer/worker started sequentially, stopped after first training round logged)

## Result: PASS

Trainer log, epoch 0 round 0:

```
memory_len                      241
loss_actor                      NaN
loss_critic                0.083813
loss_critic1                0.03872
loss_critic2               0.045093
q1_mean                    0.467529
q2_mean                     0.46682
target_q_mean              0.479268
critic_grad_norm           3.477299
actor_grad_norm                 NaN
policy_updated                  0.0
critic_warmup                   1.0
total_updates                 100.5
return_test                     0.0
return_train                    0.0
episode_length_test            81.0
episode_length_train            81.0
```

**Interpretation**:
- `memory_len 241 > start_training=200` — real replay samples were collected from
  live TrackMania and training began.
- `loss_critic`/`critic_grad_norm` finite and non-zero — real backprop occurred on
  live data.
- `loss_actor`/`actor_grad_norm` = NaN, `policy_updated=0.0`, `critic_warmup=1.0` —
  **expected, not a bug**: `TD3Agent.critic_warmup_steps=1000` intentionally freezes
  the actor for the first 1000 critic updates (see `td3/agent.py`); `total_updates=100.5`
  confirms training is still within that warmup window.
- `episode_length_train/test = 81.0`, `return = 0.0` — matches the reward function's
  own stall-detection (`MIN_STEPS=70 + FAILURE_COUNTDOWN=10 = 80` steps) firing on an
  untrained random policy, as extensively characterized during live Phase 2
  debugging (see conversation history / `docs/tmrl_integration.md` once written) —
  not a new issue.

## Prior live-debugging context (Phase 2, same session)

Before this smoke test, extensive live debugging established:
- TrackMania's gamepad control scheme required manual axis binding (Accelerate/Brake/Steering)
  in Settings > Input, since the virtual ViGEm controller must exist before TM's controller
  list is populated. One-time environment setup issue, not a code bug.
- The full env/gamepad/reward/termination pipeline was verified end-to-end with a fixed
  throttle action (both via a raw `TM2020InterfaceLidar` and via the actual wrapped
  `TD3_ENV_CLS`): car accelerated cleanly 0 -> ~45 km/h over ~3.5s before a wall/off-track
  collision, exactly matching the reward function's stall-detection timing.
- The TD3 human-BC-warmstart actor (`TD3_LIDAR_SAC_4_LIDAR_pretrained_HUMAN_actor.tmod`)
  was found to command a **constant ~-20% left steer from the very first frame**, driving
  the car into a wall almost immediately from every standing start -- a real behavior-cloning
  distributional-shift finding (looked good in the offline action-MSE audit against recorded
  human data, fails once actually deployed closed-loop), not an infrastructure bug. Worth
  revisiting when refining the human warm-start (Phase 7/13).
