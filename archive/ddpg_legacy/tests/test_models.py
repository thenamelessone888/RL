import numpy as np
import torch
from gymnasium.spaces import Box, Tuple

from ddpg.models import DDPGMLPActor, DDPGActorCritic


def main():
    print("=" * 60)
    print("DDPG MODEL TEST")
    print("=" * 60)

    # This mirrors the flattened TMRL LIDAR observation:
    #
    # speed       = 1
    # lidar       = 76
    # previous a  = 3
    # previous a  = 3
    #
    # total       = 83

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

    print("[INFO] Observation space:", observation_space)
    print("[INFO] Action space:", action_space)

    # ---------------------------------------------------------
    # Actor
    # ---------------------------------------------------------

    actor = DDPGMLPActor(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
    )

    print()
    print("[PASS] DDPG actor created")

    batch_size = 4

    observation = (
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 76),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 3),
    )

    with torch.no_grad():
        action = actor.forward(observation)

    print("[INFO] Actor output shape:", action.shape)
    print("[INFO] Actor output:")
    print(action)

    assert action.shape == (batch_size, 3)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)

    print("[PASS] Actor output shape = (4, 3)")
    print("[PASS] Actor actions inside [-1, 1]")

    # ---------------------------------------------------------
    # Critic
    # ---------------------------------------------------------

    model = DDPGActorCritic(
        observation_space=observation_space,
        action_space=action_space,
    )

    print()
    print("[PASS] DDPG Actor-Critic created")

    with torch.no_grad():
        actor_action = model.actor(observation)
        q_value = model.critic(observation, actor_action)

    print("[INFO] Critic output shape:", q_value.shape)
    print("[INFO] Critic output:")
    print(q_value)

    assert actor_action.shape == (batch_size, 3)
    assert q_value.shape == (batch_size,)

    assert torch.isfinite(actor_action).all()
    assert torch.isfinite(q_value).all()

    print("[PASS] Critic output shape = (4,)")
    print("[PASS] Critic output is finite")

    # ---------------------------------------------------------
    # Gradient test
    # ---------------------------------------------------------

    action = model.actor(observation)
    q_value = model.critic(observation, action)

    loss = -q_value.mean()
    loss.backward()

    actor_has_grad = any(
        p.grad is not None
        for p in model.actor.parameters()
    )

    critic_has_grad = any(
        p.grad is not None
        for p in model.critic.parameters()
    )

    assert actor_has_grad
    assert critic_has_grad

    print("[PASS] Actor gradients exist")
    print("[PASS] Critic gradients exist")

    print()
    print("=" * 60)
    print("PHASE 1 MODEL TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()