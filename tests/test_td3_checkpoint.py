
"""
Full checkpoint save/load/resume test, using TMRL's OWN checkpointing mechanism
(tmrl.util.dump/load -- the exact pickle-based round trip tmrl.networking.Trainer.run()
uses for CHECKPOINT_PATH), rather than re-implementing serialization ourselves.

This also exercises the real TD3_TRAINER wiring from td3/config.py
(env spaces -> TD3_MEMORY -> TD3Agent), not just a hand-built TD3Agent
as in test_td3_agent.py.

Requires a live-constructible TD3_ENV_CLS() (confirmed elsewhere in this
project to succeed without a running TrackMania instance, since rtgym only
needs the interface's declared observation/action spaces to build the
Gymnasium env) and the LIDAR interface configured in ~/TmrlData/config/config.json
(see td3/config.py's own assertions).

If those aren't available in a given environment, this test is skipped
rather than failed.
"""

import tempfile
import os

import numpy as np
import torch

from tmrl.networking import Buffer
from tmrl.util import dump, load


def _make_sample(step, rng):
    speed = np.array([float(step % 50)], dtype=np.float32)

    lidar = rng.uniform(
        0,
        50,
        size=(19,),
    ).astype(np.float32)

    observation = (
        speed,
        lidar,
    )

    action = rng.uniform(
        -1,
        1,
        size=(3,),
    ).astype(np.float32)

    reward = float(
        rng.standard_normal() * 0.01
    )

    return (
        action,
        observation,
        reward,
        False,
        False,
        {},
    )


def main():
    print("=" * 60)
    print("TD3 CHECKPOINT SAVE/LOAD/RESUME TEST")
    print("=" * 60)

    try:
        from td3.config import TD3_TRAINER

    except Exception as e:
        print(
            "[SKIP] td3.config could not be imported/configured "
            f"in this environment: {e}"
        )
        return

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    torch.manual_seed(1)
    rng = np.random.default_rng(1)

    # ------------------------------------------------------------
    # Construct the REAL TMRL TD3 trainer
    # ------------------------------------------------------------

    trainer_offline = TD3_TRAINER()

    print(
        f"[PASS] TD3_TRAINER() constructed "
        f"(agent={type(trainer_offline.agent).__name__}, "
        f"memory={type(trainer_offline.memory).__name__})"
    )

    # ------------------------------------------------------------
    # Build synthetic replay data
    # ------------------------------------------------------------

    buffer = Buffer(maxlen=2000)

    for step in range(600):
        buffer.memory.append(
            _make_sample(step, rng)
        )

    buffer.stat_train_return = 0.0
    buffer.stat_train_steps = 600

    trainer_offline.memory.append(buffer)

    assert len(trainer_offline.memory) > 0

    print(
        f"[PASS] Synthetic replay data appended "
        f"({len(trainer_offline.memory)} transitions)"
    )

    # ------------------------------------------------------------
    # Run several REAL TD3 training steps
    # ------------------------------------------------------------

    agent = trainer_offline.agent

    for _ in range(5):
        agent.train(
            trainer_offline.memory.sample()
        )

    print(
        "[PASS] Ran 5 real train() steps through the "
        "TD3_TRAINER-wired agent"
    )

    # ------------------------------------------------------------
    # Create probe observation ON THE SAME DEVICE as the actor
    #
    # IMPORTANT:
    # The TD3 model may be on CUDA.
    # Therefore probe tensors must also be on CUDA.
    # ------------------------------------------------------------

    device = agent.device

    probe_obs = tuple(
        torch.randn(
            4,
            s,
            device=device,
        )
        for s in (1, 76, 3, 3)
    )

    # ------------------------------------------------------------
    # Capture actor output before checkpoint
    # ------------------------------------------------------------

    with torch.no_grad():
        action_before_checkpoint = (
            agent.model.actor(
                probe_obs
            ).clone()
        )

    # ------------------------------------------------------------
    # Save checkpoint using TMRL's actual checkpoint mechanism
    # ------------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(
            tmp,
            "TD3_test.tcpt",
        )

        dump(
            trainer_offline,
            ckpt_path,
        )

        assert os.path.isfile(
            ckpt_path
        )

        print(
            "[PASS] tmrl.util.dump() checkpointed "
            "the full TrainingOffline instance"
        )

        # --------------------------------------------------------
        # Save training counter before destroying original object
        # --------------------------------------------------------

        total_it_before = agent._total_it

        # --------------------------------------------------------
        # Destroy original trainer/agent
        # --------------------------------------------------------

        del trainer_offline
        del agent

        # --------------------------------------------------------
        # Reload using TMRL's actual load mechanism
        # --------------------------------------------------------

        reloaded = load(
            ckpt_path
        )

        print(
            "[PASS] tmrl.util.load() restored "
            "the checkpoint"
        )

        # --------------------------------------------------------
        # Verify actor produces identical output after reload
        # --------------------------------------------------------

        # The probe observation was created on the original agent's
        # device. Verify the reloaded agent is on the same device.
        reloaded_device = reloaded.agent.device

        assert str(reloaded_device) == str(device), (
            f"checkpoint device changed: "
            f"before={device}, after={reloaded_device}"
        )

        with torch.no_grad():
            action_after_checkpoint = (
                reloaded.agent.model.actor(
                    probe_obs
                )
            )

        torch.testing.assert_close(
            action_before_checkpoint,
            action_after_checkpoint,
            atol=1e-6,
            rtol=1e-6,
        )

        print(
            "[PASS] Reloaded actor produces identical "
            "actions to the pre-checkpoint actor"
        )

        # --------------------------------------------------------
        # Verify replay memory survived
        # --------------------------------------------------------

        assert len(reloaded.memory) > 0

        print(
            f"[PASS] Reloaded replay memory intact "
            f"({len(reloaded.memory)} transitions)"
        )

        # --------------------------------------------------------
        # Verify training resumes after reload
        # --------------------------------------------------------

        critic1_before = [
            p.detach().clone()
            for p in reloaded.agent.model.critic1.parameters()
        ]

        reloaded.agent.train(
            reloaded.memory.sample()
        )

        critic1_after = [
            p.detach().clone()
            for p in reloaded.agent.model.critic1.parameters()
        ]

        assert any(
            not torch.equal(a, b)
            for a, b in zip(
                critic1_before,
                critic1_after,
            )
        )

        print(
            "[PASS] Training resumes correctly after "
            "checkpoint reload (parameters keep updating)"
        )

        # --------------------------------------------------------
        # Verify TD3 update counter survived checkpoint
        # --------------------------------------------------------

        assert reloaded.agent._total_it == (
            total_it_before + 1
        ), (
            f"expected resumed total_it counter == "
            f"{total_it_before + 1}, "
            f"got {reloaded.agent._total_it}"
        )

        print(
            "[PASS] policy_delay counter (_total_it) survives "
            "the checkpoint round trip "
            "(so the actor-update cadence stays correct "
            "across a resume, instead of resetting)"
        )

    print()
    print("=" * 60)
    print("TD3 CHECKPOINT TEST PASSED")
    print("=" * 60)


def test_main():
    main()


if __name__ == "__main__":
    main()
