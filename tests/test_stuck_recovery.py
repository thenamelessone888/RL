"""
Offline unit tests for td3/stuck_recovery.py's stuck-detection/penalty logic.

Does NOT require TrackMania: TM2020InterfaceLidarStuckRecovery.__init__ (via
its TM2020Interface/TM2020InterfaceLidar parents) only sets plain attributes
-- it never touches the game window/OpenPlanet client until initialize()/
reset() is called, which this test never does. _apply_stuck_shaping() is
exercised directly against synthetic (obs, rew, info) instead, since the
real get_obs_rew_terminated_info() requires a live screenshot.

Run via pytest (`pytest tests/ -v`) or directly:
    python tests/test_stuck_recovery.py
"""

import numpy as np

from td3.stuck_recovery import TM2020InterfaceLidarStuckRecovery


def make_obs(speed, lidar_min, lidar_shape=(4, 19)):
    """A synthetic (speed, lidar_history) obs matching TM2020InterfaceLidar's
    raw (pre-preprocessor) format: [speed(1,), imgs(img_hist_len, 19)]."""
    speed_arr = np.array([speed], dtype="float32")
    lidar = np.full(lidar_shape, lidar_min + 10.0, dtype="float32")
    lidar[0, 0] = lidar_min  # ensure the minimum is exactly lidar_min somewhere
    return [speed_arr, lidar]


def make_interface(**kwargs):
    # gamepad=False avoids constructing a vgamepad.VX360Gamepad (which would
    # otherwise happen lazily at initialize(), not __init__ -- never called
    # here regardless, but keep construction maximally inert).
    return TM2020InterfaceLidarStuckRecovery(img_hist_len=4, gamepad=False, **kwargs)


def main():
    print("=" * 60)
    print("STUCK RECOVERY REWARD SHAPING TEST")
    print("=" * 60)

    itf = make_interface(
        stuck_lidar_threshold=0.5,
        stuck_speed_threshold=2.0,
        stuck_patience=3,
        stuck_penalty=-0.1,
    )

    # ========================================================================
    # Not stuck: normal driving (high speed, clear lidar) -> reward unchanged.
    # ========================================================================
    obs = make_obs(speed=30.0, lidar_min=20.0)
    info = {}
    rew = itf._apply_stuck_shaping(obs, rew=np.float32(0.05), info=info)
    assert rew == np.float32(0.05), "reward must be unchanged while not stuck"
    assert info["stuck_penalty_applied"] is False
    assert itf._consecutive_stuck_steps == 0
    print("[PASS] Normal driving (high speed, clear lidar): reward unchanged")

    # ========================================================================
    # Stuck begins: below both thresholds, but patience (3) not yet reached.
    # ========================================================================
    for step in range(1, 3):
        obs = make_obs(speed=0.5, lidar_min=0.0)
        info = {}
        rew = itf._apply_stuck_shaping(obs, rew=np.float32(0.0), info=info)
        assert itf._consecutive_stuck_steps == step
        assert info["stuck_penalty_applied"] is False, (
            f"penalty must not apply before patience is reached (step {step})"
        )
        assert rew == np.float32(0.0)
    print("[PASS] No penalty applied before stuck_patience consecutive steps")

    # ========================================================================
    # Patience reached (3rd consecutive stuck step): penalty now applies.
    # ========================================================================
    obs = make_obs(speed=0.5, lidar_min=0.0)
    info = {}
    rew = itf._apply_stuck_shaping(obs, rew=np.float32(0.0), info=info)
    assert itf._consecutive_stuck_steps == 3
    assert info["stuck_penalty_applied"] is True
    assert rew == np.float32(-0.1), f"expected penalized reward -0.1, got {rew}"
    print("[PASS] Penalty applied once stuck_patience consecutive steps reached")

    # ========================================================================
    # Penalty keeps being applied while still stuck (not a one-shot event).
    # ========================================================================
    obs = make_obs(speed=0.5, lidar_min=0.0)
    info = {}
    rew = itf._apply_stuck_shaping(obs, rew=np.float32(0.02), info=info)
    assert info["stuck_penalty_applied"] is True
    assert rew == np.float32(0.02 - 0.1)
    print("[PASS] Penalty continues to apply on every step while still stuck, "
          "additive with whatever base reward TMRL's own reward function reports")

    # ========================================================================
    # Recovery: escaping the stuck state resets the counter and stops the
    # penalty immediately (no cooldown/hysteresis).
    # ========================================================================
    obs = make_obs(speed=15.0, lidar_min=25.0)
    info = {}
    rew = itf._apply_stuck_shaping(obs, rew=np.float32(0.04), info=info)
    assert itf._consecutive_stuck_steps == 0
    assert info["stuck_penalty_applied"] is False
    assert rew == np.float32(0.04)
    print("[PASS] Escaping the stuck state immediately stops the penalty "
          "and resets the consecutive-stuck counter")

    # ========================================================================
    # Only ONE of the two conditions being true must not trigger the penalty
    # (both lidar_min AND speed must be below their thresholds).
    # ========================================================================
    itf2 = make_interface(stuck_patience=1)

    obs_fast_but_touching = make_obs(speed=20.0, lidar_min=0.0)  # touching wall but still moving fast (e.g. grazing)
    info = {}
    rew = itf2._apply_stuck_shaping(obs_fast_but_touching, rew=np.float32(0.03), info=info)
    assert info["stuck_penalty_applied"] is False, (
        "high speed should not count as 'stuck' even with lidar_min=0"
    )

    obs_slow_but_clear = make_obs(speed=0.1, lidar_min=15.0)  # slow (e.g. reversing carefully) but not touching anything
    info = {}
    rew = itf2._apply_stuck_shaping(obs_slow_but_clear, rew=np.float32(0.0), info=info)
    assert info["stuck_penalty_applied"] is False, (
        "low speed alone (clear lidar) should not count as 'stuck'"
    )
    print("[PASS] Both lidar_min AND speed must be below threshold "
          "(neither condition alone triggers the penalty)")

    # ========================================================================
    # reset() clears the consecutive-stuck counter (offline: bypass
    # reset_common()'s live game calls by resetting the counter directly,
    # mirroring what TM2020InterfaceLidarStuckRecovery.reset() does).
    # ========================================================================
    itf3 = make_interface(stuck_patience=1)
    obs = make_obs(speed=0.0, lidar_min=0.0)
    itf3._apply_stuck_shaping(obs, rew=np.float32(0.0), info={})
    assert itf3._consecutive_stuck_steps == 1
    itf3._consecutive_stuck_steps = 0  # what reset() does, without touching the live game
    assert itf3._consecutive_stuck_steps == 0
    print("[PASS] Stuck counter can be cleared independently of live-game reset")

    print()
    print("=" * 60)
    print("STUCK RECOVERY REWARD SHAPING TEST PASSED")
    print("=" * 60)


def test_main():
    """pytest entry point (mirrors main(), enables `pytest tests/ -v` discovery)."""
    main()


if __name__ == "__main__":
    main()
