import numpy as np

from tmrl.networking import Buffer
from tmrl.custom.custom_memories import MemoryTMLidar


def make_sample(step):
    speed = np.array([float(step)], dtype=np.float32)

    # One LIDAR frame = 19 beams.
    lidar = np.full((19,), float(step), dtype=np.float32)

    observation = (speed, lidar)

    action = np.array(
        [
            0.5,
            0.0,
            np.sin(step),
        ],
        dtype=np.float32,
    )

    reward = float(step)

    terminated = False
    truncated = False

    info = {
        "step": step,
    }

    return (
        action,
        observation,
        reward,
        terminated,
        truncated,
        info,
    )


def main():
    print("SHARED MemoryTMLidar TEST (used by both ddpg/ and td3/)")

    # 4 LIDAR frames + 2 previous actions.
    memory = MemoryTMLidar(
        memory_size=100,
        batch_size=2,
        dataset_path="",
        imgs_obs=4,
        act_buf_len=2,
        nb_steps=1,
        device="cuda",
    )

    print("[PASS] MemoryTMLidar created")

    buffer = Buffer(maxlen=100)

    # Enough sequential samples for history reconstruction.
    for step in range(10):
        buffer.memory.append(make_sample(step))

    buffer.stat_train_return = 123.0
    buffer.stat_train_steps = 10

    memory.append(buffer)

    memory_len = len(memory)

    print(f"[INFO] Raw buffer samples: {len(buffer.memory)}")
    print(f"[INFO] Valid memory samples: {memory_len}")

    assert memory_len > 0
    assert memory_len <= len(buffer.memory)

    print(f"[PASS] Buffer appended: {memory_len} valid samples")

    # Inspect one reconstructed transition.
    transition_index = memory_len - 1
    transition = memory.get_transition(transition_index)

    assert len(transition) == 7

    last_obs, action, reward, new_obs, terminated, truncated, info = transition

    print("[PASS] Transition reconstructed")

    # Observation should be:
    # speed (1)
    # 4 LIDAR frames * 19 = 76
    # 2 previous actions * 3 = 6
    #
    # Total = 83 values.
    assert len(last_obs) == 4
    assert len(new_obs) == 4

    assert last_obs[0].shape == (1,)
    assert last_obs[1].shape == (76,)
    assert last_obs[2].shape == (3,)
    assert last_obs[3].shape == (3,)

    assert new_obs[0].shape == (1,)
    assert new_obs[1].shape == (76,)
    assert new_obs[2].shape == (3,)
    assert new_obs[3].shape == (3,)

    print("[PASS] Observation history shape = (1, 76, 3, 3)")

    # Verify action/reward.
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.isfinite(reward)

    print("[PASS] Action/reward valid")

    # Verify collated batch.
    batch = memory.sample()

    o, a, r, o2, d, _ = batch

    assert len(o) == 4
    assert len(o2) == 4

    assert o[0].shape[0] == 2
    assert o[1].shape == (2, 76)
    assert o[2].shape == (2, 3)
    assert o[3].shape == (2, 3)

    assert o2[0].shape[0] == 2
    assert o2[1].shape == (2, 76)
    assert o2[2].shape == (2, 3)
    assert o2[3].shape == (2, 3)

    assert a.shape == (2, 3)
    assert r.shape == (2,)
    assert d.shape == (2,)

    print("[PASS] Memory.sample() produces a well-formed (obs, act, rew, obs2, done) batch")

    # Verify everything is finite.
    for tensor in list(o) + list(o2) + [a, r, d]:
        assert np.isfinite(tensor.detach().cpu().numpy()).all()

    print("[PASS] Batch values are finite")

    print()
    print("MEMORY TEST PASSED")


def test_main():
    """pytest entry point (mirrors main(), enables `pytest tests/ -v` discovery)."""
    main()


if __name__ == "__main__":
    main()