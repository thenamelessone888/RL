import os
import tempfile

import numpy as np
import gymnasium as gym
import torch

from td3.models import TD3MLPActor
from td3.noise import GaussianExplorationNoise


def main():
    print("TD3 TMRL INTEGRATION TEST")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    observation_space = gym.spaces.Tuple((
        gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, (76,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
    ))

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    actor = TD3MLPActor(
        observation_space=observation_space,
        action_space=action_space,
    ).to_device(device)

    print("[PASS] TD3 actor created through TMRL ActorModule interface")

    actor.noise = GaussianExplorationNoise(
        action_dim=3,
        sigma=0.10,
    )

    obs = (
        torch.randn(1, 1, device=device),
        torch.randn(1, 76, device=device),
        torch.randn(1, 3, device=device),
        torch.randn(1, 3, device=device),
    )

    # Training action: noise enabled.
    train_action = actor.act(obs, test=False)

    # Evaluation action: deterministic.
    test_action_1 = actor.act(obs, test=True)
    test_action_2 = actor.act(obs, test=True)

    assert train_action.shape == (3,)
    assert test_action_1.shape == (3,)
    assert test_action_2.shape == (3,)

    assert np.isfinite(train_action).all()
    assert np.isfinite(test_action_1).all()

    print("[PASS] Training action generated")
    print("[PASS] Evaluation action generated")

    # Evaluation must be deterministic.
    np.testing.assert_allclose(test_action_1, test_action_2, rtol=1e-6, atol=1e-6)
    print("[PASS] Evaluation is deterministic")

    # Both training and evaluation actions must respect the action bounds.
    assert np.all(train_action >= -1.0 - 1e-6) and np.all(train_action <= 1.0 + 1e-6)
    print("[PASS] Noisy training action is clipped within [-1, 1]")

    # Training noise should generally alter the action (uncorrelated Gaussian, sigma=0.10).
    assert not np.allclose(train_action, test_action_1, rtol=1e-6, atol=1e-6)
    print("[PASS] TD3-style Gaussian exploration is active during training")

    # act_() (the method actually called by RolloutWorker.act()) collates a raw single
    # observation itself and must also stay deterministic under test=True.
    raw_obs = (
        np.random.randn(1).astype(np.float32),
        np.random.randn(76).astype(np.float32),
        np.random.randn(3).astype(np.float32),
        np.random.randn(3).astype(np.float32),
    )
    from tmrl.util import collate_torch
    collated = collate_torch([tuple(torch.from_numpy(x).to(device) for x in raw_obs)], device=device)
    with torch.no_grad():
        act_underscore_1 = actor.act(collated, test=True)
        act_underscore_2 = actor.act(collated, test=True)
    np.testing.assert_allclose(act_underscore_1, act_underscore_2, rtol=1e-6, atol=1e-6)
    print("[PASS] actor.act() deterministic under the exact collation RolloutWorker uses")

    # Verify TMRL serialization.
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "TD3_test.tmod")

        actor.save(model_path)
        assert os.path.isfile(model_path)

        # TorchActorModule.load() mutates AND returns the SAME instance (verified against the
        # installed tmrl==0.7.1 source, see td3/models.py's module docstring) -- this is exactly
        # the mechanism RolloutWorker relies on for `.noise` to survive a weight-broadcast
        # update. We check both the returned reference and object identity here.
        loaded_actor = actor.load(model_path, device=device)
        assert loaded_actor is actor, (
            "TorchActorModule.load() did not return the same object it was called on -- "
            "a RolloutWorker's exploration noise object would be silently lost on every "
            "update_actor_weights() call"
        )

        loaded_test_action = loaded_actor.act(obs, test=True)
        np.testing.assert_allclose(test_action_1, loaded_test_action, rtol=1e-5, atol=1e-5)
        print("[PASS] TMRL actor serialization (save/load) round trip is faithful")
        print("[PASS] load() mutates and returns the SAME actor instance (noise survives reloads)")

    print()
    print("TD3 INTEGRATION TEST PASSED")


def test_main():
    """pytest entry point (mirrors main(), enables `pytest tests/ -v` discovery)."""
    main()


if __name__ == "__main__":
    main()
