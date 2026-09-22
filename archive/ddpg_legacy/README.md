# Archived: DDPG implementation

This is the project's original, independently-implemented DDPG (single critic,
Ornstein-Uhlenbeck exploration noise) agent. It was superseded by the TD3
implementation in [`td3/`](../../td3/) as the project's primary research algorithm
(TD3 addresses DDPG's overestimation bias via twin critics + clipped double-Q
targets, delayed policy updates, and target policy smoothing).

Nothing here was deleted — it is kept for reference and as a potential ablation
baseline ("our algorithm without twin-critic/delayed-update improvements").

## Why this is archived rather than left in place

- The master research plan (Section 7) requires a single primary RL algorithm
  independently implemented from an established paper; TD3 was selected after this
  DDPG implementation revealed the random/erratic-behaviour problems that motivated
  moving to a more stable off-policy algorithm.
- Keeping both `ddpg/` and `td3/` live at the project root made it ambiguous which
  package was "the" algorithm for a reader (e.g. a professor) inspecting the repo.

## Restoring this code

This package assumes it lives at the project root (its modules use absolute imports
like `from ddpg.config import ...`, and `ddpg/human_interface.py` /
`ddpg/human_recorder.py` / `ddpg/human_dataset.py` are self-contained DDPG-coupled
copies of what is now the shared, algorithm-agnostic
[`demonstrations/`](../../demonstrations/) package). To run it again:

1. Move `archive/ddpg_legacy/ddpg/`, `archive/ddpg_legacy/train_ddpg.py`, and
   `archive/ddpg_legacy/tests/*.py` back to the project root.
2. Run `python train_ddpg.py <mode>` as before.

Its recorded-human-data dependency (`HUMAN_DATASET_PATH`) still points at
`TmrlData/dataset_human`, the same dataset TD3's behavior-cloning warm-start uses —
recording is not algorithm-specific, so no data needs to be re-recorded.
