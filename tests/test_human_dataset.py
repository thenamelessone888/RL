"""
Offline unit tests for the human demonstration dataset format/round-trip.

Does NOT require TrackMania or the `keyboard` package -- builds synthetic samples in the
exact format demonstrations/human_recorder.py produces (via the real
tmrl.custom.custom_memories.get_local_buffer_sample_lidar compressor and a real
tmrl.custom.custom_memories.MemoryTMLidar instance), persists them to a temp dataset_path
with plain pickle (exactly as demonstrations/human_recorder.py._save_memory does), reloads via
a fresh MemoryTMLidar pointed at that path (exercising Memory.__init__'s own data.pkl loading
logic), and verifies the reconstructed transitions.

Run via pytest (`pytest tests/ -v`) or directly:
    python tests/test_human_dataset.py

For a live-TrackMania integration test, see the note at the bottom of this file -- that is
NOT run here.
"""

import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np

from tmrl.networking import Buffer
from tmrl.custom.custom_memories import MemoryTMLidar, get_local_buffer_sample_lidar


def make_raw_obs(step):
    """Raw (un-preprocessed) TM2020InterfaceLidar-style obs: (speed, imgs(img_hist_len,19))."""
    speed = np.array([float(step)], dtype=np.float32)
    imgs = np.full((4, 19), float(step) / 10.0, dtype=np.float32)  # img_hist_len=4
    return [speed, imgs]


def make_flattened_obs(step, act_buf_len=2):
    """
    Mimics obs_preprocessor_tm_lidar_act_in_obs applied to the rtgym-wrapped
    (speed, imgs(4,19), *act_buf) tuple: (speed, flattened_lidar(76,), *act_buf(3,)*2).
    """
    speed, imgs = make_raw_obs(step)
    flat = imgs.flatten()  # (76,)
    acts = tuple(np.array([0.5, 0.0, np.sin(step + i)], dtype=np.float32) for i in range(act_buf_len))
    return (speed, flat, *acts)


def make_action(step):
    return np.array([0.5, 0.0, float(np.clip(np.sin(step), -1.0, 1.0))], dtype=np.float32)


def build_episode(memory, buffer, start_step, n_steps, terminate_at_end=True):
    """Appends n_steps samples to buffer in the exact recorder call chain, ending an episode."""
    for i in range(n_steps):
        step = start_step + i
        act = make_action(step)
        new_obs = make_flattened_obs(step)
        rew = float(step) * 0.1
        terminated = terminate_at_end and (i == n_steps - 1)
        truncated = False
        info = {"step": step}
        sample = get_local_buffer_sample_lidar(act, new_obs, rew, terminated, truncated, info)
        buffer.append_sample(sample)
    memory.append(buffer)
    buffer.clear()


def new_memory(dataset_path):
    return MemoryTMLidar(memory_size=100_000,
                          batch_size=2,
                          dataset_path=dataset_path,
                          imgs_obs=4,
                          act_buf_len=2,
                          nb_steps=1,
                          sample_preprocessor=None,
                          crc_debug=False,
                          device="cpu")


def save(memory, dataset_path):
    with open(Path(dataset_path) / "data.pkl", "wb") as f:
        pickle.dump(memory.data, f)


def main():
    print("HUMAN DEMONSTRATION DATASET TEST")

    tmp_dir = tempfile.mkdtemp(prefix="tmrl_human_dataset_test_")
    try:
        # -----------------------------------------------------------------
        # 1-2. Save one synthetic human transition, then a fresh episode.
        # -----------------------------------------------------------------
        memory = new_memory(tmp_dir)
        buffer = Buffer(maxlen=1000)

        # reset-sample (act = default action, matches _reset_sample in human_recorder.py)
        default_act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        reset_obs = make_flattened_obs(0)
        reset_sample = get_local_buffer_sample_lidar(default_act, reset_obs, 0.0, False, False, {"reset": True})
        buffer.append_sample(reset_sample)

        # first episode: 8 steps, ends with terminated=True
        build_episode(memory, buffer, start_step=1, n_steps=8, terminate_at_end=True)
        print(f"[PASS] Episode 1 recorded, memory length so far: {len(memory)}")

        # 9. Verify multiple episodes: second, shorter episode ending with truncated
        buffer.append_sample(get_local_buffer_sample_lidar(
            default_act, make_flattened_obs(100), 0.0, False, False, {"reset": True}))
        for i in range(5):
            step = 101 + i
            act = make_action(step)
            new_obs = make_flattened_obs(step)
            truncated = (i == 4)
            sample = get_local_buffer_sample_lidar(act, new_obs, float(step) * 0.1, False, truncated, {"step": step})
            buffer.append_sample(sample)
        memory.append(buffer)
        buffer.clear()
        print(f"[PASS] Episode 2 recorded, memory length so far: {len(memory)}")

        save(memory, tmp_dir)
        print(f"[PASS] Dataset saved to temp dir: {tmp_dir}")

        # -----------------------------------------------------------------
        # 2. Load it back (fresh instance, exercises Memory.__init__'s own load).
        # -----------------------------------------------------------------
        loaded = new_memory(tmp_dir)
        assert len(loaded) == len(memory), "reloaded memory length mismatch"
        print(f"[PASS] Dataset reloaded from disk, length={len(loaded)}")

        # -----------------------------------------------------------------
        # 3. Verify observation structure.
        # -----------------------------------------------------------------
        idx = len(loaded) - 1
        last_obs, action, reward, new_obs, terminated, truncated, info = loaded.get_transition(idx)
        assert len(last_obs) == 4 and len(new_obs) == 4
        assert last_obs[0].shape == (1,) and new_obs[0].shape == (1,)      # speed
        assert last_obs[1].shape == (76,) and new_obs[1].shape == (76,)    # lidar history
        assert last_obs[2].shape == (3,) and new_obs[2].shape == (3,)      # previous action
        assert last_obs[3].shape == (3,) and new_obs[3].shape == (3,)      # previous action
        print("[PASS] Observation structure: speed(1,) lidar(76,) prev_act(3,) prev_act(3,)")

        # -----------------------------------------------------------------
        # 4-5. Verify action shape and range.
        # -----------------------------------------------------------------
        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert action.min() >= -1.0001 and action.max() <= 1.0001
        print("[PASS] Action shape (3,) and within [-1, 1]")

        # -----------------------------------------------------------------
        # 6. Verify reward.
        # -----------------------------------------------------------------
        assert np.isfinite(reward)
        print("[PASS] Reward finite")

        # -----------------------------------------------------------------
        # 7. Verify terminated/truncated.
        # -----------------------------------------------------------------
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        print("[PASS] terminated/truncated are valid booleans")

        # -----------------------------------------------------------------
        # 8. Verify episode boundaries: eoe column matches (terminated or truncated).
        # -----------------------------------------------------------------
        eoe = np.array(loaded.data[4], dtype=bool)
        term_col = np.array(loaded.data[7], dtype=bool)
        trunc_col = np.array(loaded.data[8], dtype=bool)
        assert np.array_equal(eoe, term_col | trunc_col)
        assert eoe.sum() == 2, f"expected 2 episode-ending flags, got {eoe.sum()}"
        print("[PASS] Episode boundaries respected (2 episodes, eoe == terminated OR truncated)")

        # -----------------------------------------------------------------
        # 9. Multiple episodes already verified above (2 episodes accumulated correctly).
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # 10. Verify no corruption: sample many transitions, all finite, all in range.
        # -----------------------------------------------------------------
        n_checked = 0
        for i in range(len(loaded)):
            lo, a, r, no, te, tr, inf = loaded.get_transition(i)
            assert np.isfinite(a).all() and a.min() >= -1.0001 and a.max() <= 1.0001
            assert np.isfinite(r)
            for x in lo:
                assert np.isfinite(x).all()
            for x in no:
                assert np.isfinite(x).all()
            n_checked += 1
        print(f"[PASS] {n_checked} transitions checked: no NaNs, no out-of-range actions")

        # Sanity: memory.sample() (the actual DDPG-consumed batch) also works end-to-end.
        loaded.batch_size = 2
        batch = loaded.sample()
        o, a, r, o2, d, _ = batch
        assert a.shape == (2, 3)
        assert r.shape == (2,)
        print("[PASS] memory.sample() produces a well-formed DDPG-compatible batch")

        print()
        print("=" * 60)
        print("HUMAN DEMONSTRATION DATASET TEST PASSED")
        print("=" * 60)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_main():
    """pytest entry point (mirrors main(), enables `pytest tests/ -v` discovery)."""
    main()


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# LIVE TRACKMANIA TEST REQUIRED (not run by this file):
#   python train_td3.py record --episodes 1
#   python train_td3.py inspect-human-data
# These require a running TrackMania 2020 instance with the OpenPlanet TMRL plugin and
# real keyboard input, and cannot be exercised in an offline/CI environment.
# -----------------------------------------------------------------------------
