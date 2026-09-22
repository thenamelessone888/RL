import numpy as np
import torch
from gymnasium.spaces import Box, Tuple

from td3.models import TD3MLPActor, TD3ActorCritic


def make_spaces():
    observation_space = Tuple((
        Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(76,), dtype=np.float32),
        Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    ))
    action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return observation_space, action_space


def make_obs(batch_size):
    return (
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 76),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 3),
    )


def main():
    print("=" * 60)
    print("TD3 MODEL TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()
    print("[INFO] Observation space:", observation_space)
    print("[INFO] Action space:", action_space)

    # ---------------------------------------------------------
    # Actor: output shape / bounds
    # ---------------------------------------------------------
    actor = TD3MLPActor(observation_space=observation_space, action_space=action_space, device="cpu")
    print("[PASS] TD3 actor created")

    batch_size = 4
    observation = make_obs(batch_size)

    with torch.no_grad():
        action = actor.forward(observation)

    assert action.shape == (batch_size, 3)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    print("[PASS] Actor output shape = (4, 3)")
    print("[PASS] Actor actions inside [-1, 1]")

    # ---------------------------------------------------------
    # Deterministic actor inference: repeated calls, same input -> same output
    # ---------------------------------------------------------
    with torch.no_grad():
        action_a = actor.forward(observation)
        action_b = actor.forward(observation)
    assert torch.equal(action_a, action_b)
    print("[PASS] Actor forward pass is deterministic for identical input")

    # act() with test=True must also be deterministic and bounded.
    single_obs = tuple(x[:1] for x in observation)
    a1 = actor.act(single_obs, test=True)
    a2 = actor.act(single_obs, test=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)
    assert np.all(a1 >= -1.0 - 1e-6) and np.all(a1 <= 1.0 + 1e-6)
    print("[PASS] actor.act(test=True) is deterministic and bounded")

    # ---------------------------------------------------------
    # Actor-Critic: twin critics are independent
    # ---------------------------------------------------------
    model = TD3ActorCritic(observation_space=observation_space, action_space=action_space)
    print("[PASS] TD3 Actor-Critic created")

    with torch.no_grad():
        actor_action = model.actor(observation)
        q1_value = model.critic1(observation, actor_action)
        q2_value = model.critic2(observation, actor_action)

    assert actor_action.shape == (batch_size, 3)
    assert q1_value.shape == (batch_size,)
    assert q2_value.shape == (batch_size,)
    assert torch.isfinite(q1_value).all()
    assert torch.isfinite(q2_value).all()
    print("[PASS] Both critics' output shape = (4,)")
    print("[PASS] Both critics' output is finite")

    # Independence: the two critics must NOT be the same module / share weights, and must
    # generally produce DIFFERENT Q-values for the same (s, a) since they are independently
    # initialized (this is the entire point of TD3's twin-critic design -- if they always agreed
    # exactly, the min(Q1, Q2) trick would be a no-op).
    assert model.critic1 is not model.critic2
    params1 = list(model.critic1.parameters())
    params2 = list(model.critic2.parameters())
    assert len(params1) == len(params2)
    identical_init = all(torch.equal(p1, p2) for p1, p2 in zip(params1, params2))
    assert not identical_init, "critic1 and critic2 were initialized identically -- they are not independent"
    assert not torch.allclose(q1_value, q2_value), "critic1 and critic2 produced identical Q-values -- suspicious for independently-initialized twin critics"
    print("[PASS] critic1 and critic2 are independently-initialized, independent modules")

    # Gradient isolation: a critic1-only loss must not populate critic2's .grad, and vice versa.
    model.zero_grad()
    q1_value = model.critic1(observation, actor_action.detach())
    (q1_value.mean()).backward()
    critic1_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.critic1.parameters())
    critic2_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.critic2.parameters())
    assert critic1_has_grad
    assert not critic2_has_grad
    print("[PASS] Backpropagating through critic1 alone leaves critic2's gradients untouched")

    # ---------------------------------------------------------
    # Gradient test: actor loss through critic1 only (TD3 convention)
    # ---------------------------------------------------------
    model.zero_grad()
    action = model.actor(observation)
    q1_pi = model.critic1(observation, action)
    loss = -q1_pi.mean()
    loss.backward()

    actor_has_grad = any(p.grad is not None for p in model.actor.parameters())
    critic1_has_grad = any(p.grad is not None for p in model.critic1.parameters())
    critic2_has_grad = any(p.grad is not None for p in model.critic2.parameters())

    assert actor_has_grad
    assert critic1_has_grad
    assert not critic2_has_grad, "actor loss (TD3 convention: critic1 only) leaked gradients into critic2"
    print("[PASS] Actor gradients exist")
    print("[PASS] Critic1 gradients exist (actor loss source)")
    print("[PASS] Critic2 gradients are absent (not used for the actor loss, per TD3)")

    print()
    print("=" * 60)
    print("TD3 MODEL TEST PASSED")
    print("=" * 60)


def test_main():
    """pytest entry point (mirrors main(), enables `pytest tests/ -v` discovery)."""
    main()


if __name__ == "__main__":
    main()
