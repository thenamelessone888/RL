"""
Validates and reports statistics on the recorded human demonstration dataset.

Reuses a real tmrl.custom.custom_memories.MemoryTMLidar instance both to read data.pkl
(Memory.__init__'s own load-if-exists logic) and to decode/reconstruct full transitions via
its own get_transition() (the same LIDAR/action-history reconstruction logic used for
DDPG/SAC training) -- we do not re-implement history reconstruction (Problem 4 lesson).

Raw per-step statistics (action range, NaNs, episode boundaries) are computed directly from
memory.data's 9 parallel columns, confirmed via inspection of MemoryTMLidar.append_buffer():
  data[0]=index  data[1]=action  data[2]=speed  data[3]=lidar(19,)  data[4]=eoe(bool)
  data[5]=reward data[6]=info    data[7]=terminated(bool)           data[8]=truncated(bool)
"""

import numpy as np

import tmrl.config.config_constants as cfg
from ddpg.human_interface import HUMAN_DATASET_PATH


def _load_memory():
    from tmrl.custom.custom_memories import MemoryTMLidar
    return MemoryTMLidar(memory_size=100_000_000,
                          batch_size=1,
                          dataset_path=HUMAN_DATASET_PATH,
                          imgs_obs=cfg.IMG_HIST_LEN,
                          act_buf_len=cfg.ACT_BUF_LEN,
                          nb_steps=1,
                          sample_preprocessor=None,
                          crc_debug=False,
                          device="cpu")


def _episode_lengths(eoe):
    lengths = []
    cur = 0
    for e in eoe:
        cur += 1
        if e:
            lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)  # trailing, not-yet-terminated episode
    return lengths


def inspect_human_data(verbose=True):
    memory = _load_memory()
    n_raw_steps = len(memory.data[0]) if len(memory.data) > 0 else 0

    if n_raw_steps == 0:
        if verbose:
            print(f"No data found at: {HUMAN_DATASET_PATH}")
            print("Run `python train_ddpg.py record` first.")
        return {"n_raw_steps": 0, "n_transitions": 0, "n_episodes": 0}

    actions = np.stack(memory.data[1])          # (N, 3)
    speeds = np.stack(memory.data[2])            # (N, 1)
    lidars = np.stack(memory.data[3])             # (N, 19)
    rewards = np.array(memory.data[5], dtype=np.float32)
    eoe = np.array(memory.data[4], dtype=bool)
    terminated = np.array(memory.data[7], dtype=bool)
    truncated = np.array(memory.data[8], dtype=bool)

    n_episodes = int(eoe.sum())
    ep_lengths = _episode_lengths(eoe)
    n_transitions = len(memory)  # usable (prev_obs, act, rew, obs, done) transitions

    problems = []

    if not np.isfinite(actions).all():
        problems.append("NaN/Inf found in recorded actions")
    if actions.min() < -1.0001 or actions.max() > 1.0001:
        problems.append(f"actions outside [-1, 1]: min={actions.min():.4f} max={actions.max():.4f}")
    if not np.isfinite(speeds).all():
        problems.append("NaN/Inf found in recorded speeds")
    if not np.isfinite(lidars).all():
        problems.append("NaN/Inf found in recorded LIDAR")
    if not np.isfinite(rewards).all():
        problems.append("NaN/Inf found in recorded rewards")
    if eoe.dtype != bool or terminated.dtype != bool or truncated.dtype != bool:
        problems.append("episode-boundary flags are not boolean")
    if not np.array_equal(eoe, terminated | truncated):
        problems.append("eoe flag inconsistent with (terminated OR truncated)")

    # Decode-path sanity check: reconstruct one real transition via MemoryTMLidar's own logic.
    obs_shapes = None
    decode_error = None
    if n_transitions > 0:
        try:
            last_obs, new_act, rew, new_obs, term, trunc, info = memory.get_transition(0)
            obs_shapes = [np.asarray(x).shape for x in new_obs]
        except Exception as e:
            decode_error = str(e)
            problems.append(f"get_transition() decode failed: {decode_error}")
    else:
        problems.append(
            f"only {n_raw_steps} raw steps recorded, below the minimum "
            f"({max(cfg.IMG_HIST_LEN, cfg.ACT_BUF_LEN) + 1}) needed to reconstruct one transition"
        )

    stats = {
        "dataset_path": HUMAN_DATASET_PATH,
        "n_raw_steps": n_raw_steps,
        "n_episodes": n_episodes,
        "n_transitions": n_transitions,
        "episode_lengths": ep_lengths,
        "action_gas": (float(actions[:, 0].min()), float(actions[:, 0].max()), float(actions[:, 0].mean())),
        "action_brake": (float(actions[:, 1].min()), float(actions[:, 1].max()), float(actions[:, 1].mean())),
        "action_steer": (float(actions[:, 2].min()), float(actions[:, 2].max()), float(actions[:, 2].mean())),
        "reward": (float(rewards.mean()), float(rewards.min()), float(rewards.max())),
        "obs_shapes": obs_shapes,
        "problems": problems,
    }

    if verbose:
        print("=" * 60)
        print("HUMAN DEMONSTRATION DATASET")
        print("=" * 60)
        print(f"Dataset path: {HUMAN_DATASET_PATH}")
        print()
        print(f"Episodes: {n_episodes}")
        print(f"Raw recorded steps: {n_raw_steps}")
        print(f"Usable transitions: {n_transitions}")
        print()
        print("Action statistics:")
        for name, (lo, hi, mean) in [("gas", stats["action_gas"]),
                                      ("brake", stats["action_brake"]),
                                      ("steer", stats["action_steer"])]:
            print(f"  {name}:")
            print(f"    min:  {lo:.4f}")
            print(f"    max:  {hi:.4f}")
            print(f"    mean: {mean:.4f}")
        print()
        print("Reward:")
        print(f"  mean: {stats['reward'][0]:.4f}")
        print(f"  min:  {stats['reward'][1]:.4f}")
        print(f"  max:  {stats['reward'][2]:.4f}")
        print()
        if ep_lengths:
            print("Episode lengths:")
            print(f"  mean: {np.mean(ep_lengths):.1f}")
            print(f"  min:  {int(np.min(ep_lengths))}")
            print(f"  max:  {int(np.max(ep_lengths))}")
        print()
        if obs_shapes is not None:
            labels = ["speed", "lidar", "previous action", "previous action"][:len(obs_shapes)]
            print("Observation (reconstructed via MemoryTMLidar.get_transition):")
            for label, shape in zip(labels, obs_shapes):
                print(f"  {label} shape: {shape}")
        print()
        if problems:
            print("VALIDATION: PROBLEMS FOUND")
            for p in problems:
                print(f"  [FAIL] {p}")
        else:
            print("VALIDATION: OK (no NaNs, actions in range, episode boundaries consistent)")
        print()

    return stats


if __name__ == "__main__":
    inspect_human_data()
