import copy

import numpy as np
import torch

from tmrl.util import collate_torch
from gymnasium.spaces import Box, Tuple

from ddpg.agent import DDPGAgent


def make_spaces():
    observation_space = Tuple((
        Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        ),
        Box(
            low=-np.inf,
            high=np.inf,
            shape=(76,),
            dtype=np.float32,
        ),
        Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
        Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        ),
    ))

    action_space = Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    return observation_space, action_space


def make_batch(batch_size=32):
    samples = []

    for _ in range(batch_size):
        observation = (
            torch.randn(1),
            torch.randn(76),
            torch.randn(3),
            torch.randn(3),
        )

        next_observation = (
            torch.randn(1),
            torch.randn(76),
            torch.randn(3),
            torch.randn(3),
        )

        action = torch.rand(3) * 2.0 - 1.0
        reward = torch.randn(())
        terminated = np.float32(0.0)
        truncated = np.float32(0.0)

        samples.append(
            (
                observation,
                action,
                reward,
                next_observation,
                terminated,
                truncated,
            )
        )

    return samples


def parameters_changed(before, after):
    for p_before, p_after in zip(before, after):
        if not torch.equal(p_before, p_after):
            return True
    return False


def parameters_finite(model):
    return all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )


def main():
    print("=" * 60)
    print("DDPG AGENT TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Device:", device)

    agent = DDPGAgent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )

    print("[PASS] DDPGAgent created")

    # ---------------------------------------------------------
    # Verify target networks start equal to online networks
    # ---------------------------------------------------------

    actor_before = copy.deepcopy(
        [p.detach().clone() for p in agent.model.actor.parameters()]
    )

    actor_target_before = copy.deepcopy(
        [p.detach().clone() for p in agent.model_target.actor.parameters()]
    )

    critic_before = copy.deepcopy(
        [p.detach().clone() for p in agent.model.critic.parameters()]
    )

    critic_target_before = copy.deepcopy(
        [p.detach().clone() for p in agent.model_target.critic.parameters()]
    )

    assert all(
        torch.equal(a, b)
        for a, b in zip(actor_before, actor_target_before)
    )

    assert all(
        torch.equal(a, b)
        for a, b in zip(critic_before, critic_target_before)
    )

    print("[PASS] Target actor initialized from online actor")
    print("[PASS] Target critic initialized from online critic")

    # ---------------------------------------------------------
    # Synthetic TMRL-style batch
    # ---------------------------------------------------------

    batch = make_batch()

    print("[PASS] Synthetic TMRL batch created")

    # ---------------------------------------------------------
    # Run DDPG update
    # ---------------------------------------------------------
    batch = collate_torch(batch, device=device)
    metrics = agent.train(batch)

    print()
    print("[INFO] Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    assert "loss_actor" in metrics
    assert "loss_critic" in metrics

    assert np.isfinite(metrics["loss_actor"])
    assert np.isfinite(metrics["loss_critic"])

    print("[PASS] Actor loss is finite")
    print("[PASS] Critic loss is finite")

    # ---------------------------------------------------------
    # Verify online networks changed
    # ---------------------------------------------------------

    actor_after = [
        p.detach().clone()
        for p in agent.model.actor.parameters()
    ]

    critic_after = [
        p.detach().clone()
        for p in agent.model.critic.parameters()
    ]

    assert parameters_changed(actor_before, actor_after)
    assert parameters_changed(critic_before, critic_after)

    print("[PASS] Actor parameters changed")
    print("[PASS] Critic parameters changed")

    # ---------------------------------------------------------
    # Verify target networks changed through Polyak update
    # ---------------------------------------------------------

    actor_target_after = [
        p.detach().clone()
        for p in agent.model_target.actor.parameters()
    ]

    critic_target_after = [
        p.detach().clone()
        for p in agent.model_target.critic.parameters()
    ]

    assert parameters_changed(
        actor_target_before,
        actor_target_after,
    )

    assert parameters_changed(
        critic_target_before,
        critic_target_after,
    )

    print("[PASS] Target actor updated")
    print("[PASS] Target critic updated")

    # ---------------------------------------------------------
    # Verify all parameters remain finite
    # ---------------------------------------------------------

    assert parameters_finite(agent.model)
    assert parameters_finite(agent.model_target)

    print("[PASS] Online network parameters are finite")
    print("[PASS] Target network parameters are finite")

    print()
    print("=" * 60)
    print("PHASE 2 DDPG UPDATE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()