import os
import tempfile

import numpy as np
import gymnasium as gym
import torch

from ddpg.models import DDPGMLPActor
from ddpg.noise import OrnsteinUhlenbeckNoise


def main():
    print("DDPG TMRL INTEGRATION TEST")

    observation_space = gym.spaces.Tuple((
        gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, (76,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
    ))

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    actor = DDPGMLPActor(
        observation_space=observation_space,
        action_space=action_space,
    ).to_device("cuda")

    print("[PASS] DDPG actor created through TMRL ActorModule interface")

    actor.noise = OrnsteinUhlenbeckNoise(
        action_dim=3,
        theta=0.15,
        sigma=0.20,
    )

    obs = (
        torch.randn(1, 1, device="cuda"),
        torch.randn(1, 76, device="cuda"),
        torch.randn(1, 3, device="cuda"),
        torch.randn(1, 3, device="cuda"),
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
    np.testing.assert_allclose(
        test_action_1,
        test_action_2,
        rtol=1e-6,
        atol=1e-6,
    )

    print("[PASS] Evaluation is deterministic")

    # Training noise should generally alter the action.
    # We don't require every random sample to differ, but with this noise
    # configuration it should differ from the deterministic action.
    assert not np.allclose(
        train_action,
        test_action_1,
        rtol=1e-6,
        atol=1e-6,
    )

    print("[PASS] OU exploration active during training")

    # Verify TMRL serialization.
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "DDPG_test.tmod")

        actor.save(model_path)

        assert os.path.isfile(model_path)

        loaded_actor = actor.load(
            model_path,
            device="cuda",
        )

        loaded_test_action = loaded_actor.act(
            obs,
            test=True,
        )

        np.testing.assert_allclose(
            test_action_1,
            loaded_test_action,
            rtol=1e-5,
            atol=1e-5,
        )

        print("[PASS] TMRL actor serialization")

    print()
    print("PHASE 4 INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()