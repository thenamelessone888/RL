# Autonomous TrackMania Driving: TD3 + Curriculum Learning on TMRL

Research project building an independently-implemented TD3 (Fujimoto, Hoof & Meger,
2018) continuous-control agent on top of [TMRL](https://github.com/trackmania-rl/tmrl)
0.7.1's TrackMania 2020 real-time environment, using LIDAR/state observations,
human-demonstration behavior-cloning warm-start, and (planned) curriculum learning
across tracks of increasing difficulty.

**This is not TMRL's own SAC/REDQ.** TD3 (twin critics, clipped double-Q targets,
delayed policy updates, target policy smoothing) is implemented from scratch in
[`td3/`](td3/), reusing only TMRL's infrastructure (environment, networking,
replay-memory format, `TrainingAgent`/`ActorModule` extension points).

## Three-directory layout (do not confuse these)

| Directory | What it is |
|---|---|
| `C:\Users\sreci\RL\tmrl_custom_algo_implementaion` (this repo) | All source code, tests, configs, docs, experiment metadata. |
| `C:\Users\sreci\RL\tmrl-env` | The Python virtual environment. `tmrl==0.7.1` is installed here; treated as read-only reference — never modified. |
| `C:\Users\sreci\TmrlData` | TMRL's own runtime data: checkpoints, weights, reward files, replay datasets, human demonstrations. Not part of this git repo. |

## Project structure

```
td3/                  TD3 agent, actor/twin-critic models, config, diagnostics
                       (the project's current, primary RL algorithm)
demonstrations/        Shared, algorithm-agnostic human-demonstration recording,
                       inspection, and dataset tooling (used by TD3's behavior-
                       cloning warm-start; not TD3-specific by design)
tests/                 pytest suite (`pytest tests/ -v`) - 9 tests, all passing
archive/ddpg_legacy/   The project's original DDPG implementation, superseded by
                       TD3 (see archive/ddpg_legacy/README.md for why)
configs/, docs/,
results/, experiments/ Scaffolding for later phases - see each folder's README.md
train_td3.py           CLI launcher: server / trainer / worker / record /
                       pretrain-human / inspect-human-data / diagnose / eval modes
```

## Setup

```bash
# Activate the existing venv (already has tmrl installed)
C:\Users\sreci\RL\tmrl-env\Scripts\activate

cd C:\Users\sreci\RL\tmrl_custom_algo_implementaion
pip install -r requirements.txt   # only needed if the venv is rebuilt from scratch
```

## Running

Three separate terminals, from this directory, with `tmrl-env` activated:

```bash
python train_td3.py server
python train_td3.py trainer
python train_td3.py worker
```

Human demonstration workflow (shared with any future algorithm, not TD3-specific):

```bash
python train_td3.py record --episodes 5       # record human driving
python train_td3.py inspect-human-data         # validate the recorded dataset
python train_td3.py pretrain-human             # behavior-clone the TD3 actor from it
python train_td3.py trainer --human-warmstart  # train TD3 starting from that actor
python train_td3.py worker --human-warmstart
```

Diagnostics:

```bash
python train_td3.py diagnose
python train_td3.py verify-bc
python train_td3.py bc-eval --episodes 1
python train_td3.py random-eval --episodes 1
```

## Testing

```bash
python -m pytest tests/ -v
```

All 9 tests are offline (no TrackMania/OpenPlanet required): TD3 agent/model/config/
checkpoint/demonstration unit tests, plus two shared tests for the human-demonstration
dataset format and the underlying `MemoryTMLidar` replay format.

## Status

See the project owner's working notes for current phase status against the research
plan's 12-phase development order (Phase 0: inspection through Phase 12: final
experiments). As of this cleanup pass: TD3 is implemented and unit-tested; human
behavior-cloning warm-start is implemented and unit-tested; curriculum learning,
multi-track training, and generalization evaluation are not yet implemented.
