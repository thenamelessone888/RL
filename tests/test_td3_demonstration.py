import torch
from gymnasium.spaces import Box, Tuple

from td3.models import TD3MLPActor


def main():
    observation_space = Tuple((
        Box(low=-float("inf"), high=float("inf"), shape=(1,), dtype=float),
        Box(low=-float("inf"), high=float("inf"), shape=(76,), dtype=float),
        Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
        Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
    ))
    action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=float)

    actor = TD3MLPActor(observation_space, action_space, device="cpu")
    obs = (torch.randn(32, 1), torch.randn(32, 76), torch.randn(32, 3), torch.randn(32, 3))
    target = torch.rand(32, 3) * 2 - 1
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in actor.parameters()]
    pred = actor.forward(obs)
    loss = ((pred - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = [p.detach().clone() for p in actor.parameters()]

    assert pred.shape == (32, 3)
    assert torch.all(pred <= 1) and torch.all(pred >= -1)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    print("[PASS] BC forward/loss/update changes TD3 actor weights")
    print("TD3 DEMONSTRATION TEST PASSED")


def test_main():
    main()


if __name__ == "__main__":
    main()
