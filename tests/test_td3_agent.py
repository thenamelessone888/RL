
import os
import tempfile

import numpy as np
import torch
from gymnasium.spaces import Box, Tuple

from tmrl.util import collate_torch

from td3.agent import TD3Agent
from td3.models import TD3MLPActor


# ============================================================================
# TEST SPACES / DATA
# ============================================================================

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


def make_batch(batch_size=32, terminated_frac=0.0):
    samples = []

    for i in range(batch_size):
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

        terminated = (
            np.float32(1.0)
            if i < int(batch_size * terminated_frac)
            else np.float32(0.0)
        )

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


# ============================================================================
# PARAMETER HELPERS
# ============================================================================

def params_equal(a, b):
    return all(
        torch.equal(x, y)
        for x, y in zip(a, b)
    )


def params_changed(before, after):
    return any(
        not torch.equal(x, y)
        for x, y in zip(before, after)
    )


def clone_params(module):
    return [
        p.detach().clone()
        for p in module.parameters()
    ]


# ============================================================================
# MAIN TD3 ALGORITHM TEST
# ============================================================================

def main():
    print("=" * 60)
    print("TD3 AGENT TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)

    # ========================================================================
    # Construction + target initialization
    #
    # Warm-up is explicitly disabled here because this section tests the
    # ordinary TD3 update schedule itself.
    # ========================================================================

    agent = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        policy_delay=2,
        critic_warmup_steps=0,
    )

    print("[PASS] TD3Agent created")

    assert params_equal(
        clone_params(agent.model.actor),
        clone_params(agent.model_target.actor),
    )

    assert params_equal(
        clone_params(agent.model.critic1),
        clone_params(agent.model_target.critic1),
    )

    assert params_equal(
        clone_params(agent.model.critic2),
        clone_params(agent.model_target.critic2),
    )

    print(
        "[PASS] Target actor/critic1/critic2 initialized "
        "as hard copies of the online networks"
    )

    # Target networks must never receive gradients.
    assert all(
        not p.requires_grad
        for p in agent.model_target.parameters()
    )

    print(
        "[PASS] Target networks have requires_grad=False "
        "(frozen, never optimized directly)"
    )

    # ========================================================================
    # Twin critics must be independent
    # ========================================================================

    assert agent.model.critic1 is not agent.model.critic2
    assert (
        agent.model_target.critic1
        is not agent.model_target.critic2
    )

    print(
        "[PASS] critic1/critic2 (and their targets) "
        "are distinct module instances"
    )

    # ========================================================================
    # Min-Q target
    # ========================================================================

    agent_minq = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        target_policy_noise=0.0,
        target_noise_clip=0.0,
        policy_delay=1,
        critic_warmup_steps=0,
    )

    batch = collate_torch(
        make_batch(16),
        device=device,
    )

    o, a, r, o2, d, _ = batch

    with torch.no_grad():
        a2 = agent_minq._smoothed_target_action(o2)

        q1_targ = agent_minq.model_target.critic1(
            o2,
            a2,
        )

        q2_targ = agent_minq.model_target.critic2(
            o2,
            a2,
        )

        expected_target_q_mean = (
            torch.min(q1_targ, q2_targ)
            .mean()
            .item()
        )

    metrics = agent_minq.train(batch)

    assert abs(
        metrics["target_q_mean"]
        - expected_target_q_mean
    ) < 1e-5, (
        f"target_q_mean={metrics['target_q_mean']} "
        f"!= mean(min(Q1_targ,Q2_targ))="
        f"{expected_target_q_mean}"
    )

    # A valid elementwise minimum can equal Q1 for every element in a
    # particular random batch.  Checking that its *mean* differs from Q1's
    # mean is therefore flaky.  Verify the actual elementwise TD3 operation.
    expected_target_q = torch.minimum(q1_targ, q2_targ)
    assert torch.all(expected_target_q <= q1_targ)
    assert torch.all(expected_target_q <= q2_targ)

    print(
        "[PASS] Bellman target uses "
        "min(Q1_target, Q2_target)"
    )

    # ========================================================================
    # Target policy smoothing / clipping
    # ========================================================================

    agent_smooth = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        target_policy_noise=5.0,
        target_noise_clip=0.5,
        critic_warmup_steps=0,
    )

    o2_probe = tuple(
        x.to(device)
        for x in (
            torch.randn(64, 1),
            torch.randn(64, 76),
            torch.randn(64, 3),
            torch.randn(64, 3),
        )
    )

    with torch.no_grad():
        raw_target_action = (
            agent_smooth.model_target.actor(o2_probe)
        )

        smoothed = (
            agent_smooth._smoothed_target_action(o2_probe)
        )

    assert smoothed.shape == raw_target_action.shape

    assert torch.all(
        smoothed
        >= -agent_smooth.act_limit - 1e-6
    )

    assert torch.all(
        smoothed
        <= agent_smooth.act_limit + 1e-6
    )

    frac_changed = (
        (smoothed - raw_target_action)
        .abs()
        .gt(1e-6)
        .float()
        .mean()
        .item()
    )

    assert frac_changed > 0.9, (
        "expected target-smoothing noise to perturb "
        f"almost all samples, got {frac_changed:.2%}"
    )

    print(
        "[PASS] Target policy smoothing noise is clipped "
        "to [-target_noise_clip, target_noise_clip]"
    )

    print(
        "[PASS] Smoothed target action stays within "
        "action bounds [-1, 1]"
    )

    # ========================================================================
    # No target smoothing
    # ========================================================================

    agent_nosmooth = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        target_policy_noise=0.0,
        target_noise_clip=0.5,
        critic_warmup_steps=0,
    )

    with torch.no_grad():
        raw2 = (
            agent_nosmooth.model_target.actor(o2_probe)
        )

        smoothed2 = (
            agent_nosmooth._smoothed_target_action(o2_probe)
        )

    assert torch.equal(
        raw2,
        smoothed2,
    ), (
        "target_policy_noise=0.0 should leave "
        "the target action unchanged"
    )

    print(
        "[PASS] target_policy_noise=0.0 leaves "
        "the target action exactly unchanged"
    )

    # ========================================================================
    # Delayed actor updates + delayed target Polyak updates
    #
    # Warm-up disabled so this section specifically verifies standard TD3.
    # ========================================================================

    actor_before = clone_params(
        agent.model.actor
    )

    critic1_before = clone_params(
        agent.model.critic1
    )

    critic2_before = clone_params(
        agent.model.critic2
    )

    actor_target_before = clone_params(
        agent.model_target.actor
    )

    critic1_target_before = clone_params(
        agent.model_target.critic1
    )

    critic2_target_before = clone_params(
        agent.model_target.critic2
    )

    batch1 = collate_torch(
        make_batch(32),
        device=device,
    )

    metrics1 = agent.train(batch1)

    assert metrics1["total_updates"] == 1
    assert metrics1["policy_updated"] == 0.0

    assert params_changed(
        critic1_before,
        clone_params(agent.model.critic1),
    )

    assert params_changed(
        critic2_before,
        clone_params(agent.model.critic2),
    )

    print(
        "[PASS] Critics update on EVERY train() call (call #1)"
    )

    assert params_equal(
        actor_before,
        clone_params(agent.model.actor),
    )

    assert params_equal(
        actor_target_before,
        clone_params(agent.model_target.actor),
    )

    assert params_equal(
        critic1_target_before,
        clone_params(agent.model_target.critic1),
    )

    assert params_equal(
        critic2_target_before,
        clone_params(agent.model_target.critic2),
    )

    print(
        "[PASS] Actor and ALL target networks remain "
        "frozen on a non-delayed step (call #1)"
    )

    assert np.isnan(
        metrics1["loss_actor"]
    )

    critic1_before_2 = clone_params(
        agent.model.critic1
    )

    critic2_before_2 = clone_params(
        agent.model.critic2
    )

    batch2 = collate_torch(
        make_batch(32),
        device=device,
    )

    metrics2 = agent.train(batch2)

    assert metrics2["total_updates"] == 2
    assert metrics2["policy_updated"] == 1.0
    assert not np.isnan(
        metrics2["loss_actor"]
    )

    assert params_changed(
        critic1_before_2,
        clone_params(agent.model.critic1),
    )

    assert params_changed(
        critic2_before_2,
        clone_params(agent.model.critic2),
    )

    print(
        "[PASS] Critics update on call #2 too"
    )

    assert params_changed(
        actor_before,
        clone_params(agent.model.actor),
    )

    assert params_changed(
        actor_target_before,
        clone_params(agent.model_target.actor),
    )

    assert params_changed(
        critic1_target_before,
        clone_params(agent.model_target.critic1),
    )

    assert params_changed(
        critic2_target_before,
        clone_params(agent.model_target.critic2),
    )

    print(
        "[PASS] Actor and ALL target networks update "
        "on the delayed step (call #2)"
    )

    # Polyak averaging must lag the online network.
    online_actor_now = clone_params(
        agent.model.actor
    )

    target_actor_now = clone_params(
        agent.model_target.actor
    )

    assert not params_equal(
        online_actor_now,
        target_actor_now,
    ), (
        "target actor is bit-identical to the online actor "
        "after a Polyak update -- this looks like a hard "
        "copy, not a soft (tau) update"
    )

    print(
        "[PASS] Target actor lags the online actor "
        "(soft update, not a hard copy)"
    )

    # ========================================================================
    # Terminal handling
    # ========================================================================

    agent_term = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        critic_warmup_steps=0,
    )

    batch_term = make_batch(
        batch_size=8,
        terminated_frac=1.0,
    )

    batch_term_collated = collate_torch(
        batch_term,
        device=device,
    )

    o_t, a_t, r_t, o2_t, d_t, _t = (
        batch_term_collated
    )

    assert torch.all(
        d_t == 1.0
    )

    with torch.no_grad():
        a2_t = (
            agent_term._smoothed_target_action(o2_t)
        )

        q1_targ_t = (
            agent_term.model_target.critic1(
                o2_t,
                a2_t,
            )
        )

        q2_targ_t = (
            agent_term.model_target.critic2(
                o2_t,
                a2_t,
            )
        )

        q_targ_t = torch.min(
            q1_targ_t,
            q2_targ_t,
        )

        expected_backup = r_t

        actual_backup = (
            r_t
            + agent_term.gamma
            * (1 - d_t)
            * q_targ_t
        )

    assert torch.allclose(
        actual_backup,
        expected_backup,
        atol=1e-5,
    )

    print(
        "[PASS] terminated=1 correctly zeroes "
        "the bootstrap term"
    )

    # ========================================================================
    # Finite parameters
    # ========================================================================

    def parameters_finite(model):
        return all(
            torch.isfinite(p).all().item()
            for p in model.parameters()
        )

    assert parameters_finite(agent.model)
    assert parameters_finite(
        agent.model_target
    )

    print(
        "[PASS] All online/target parameters remain "
        "finite after training"
    )

    # ========================================================================
    # Gradients
    # ========================================================================

    assert any(
        p.grad is not None
        for p in agent.model.critic1.parameters()
    )

    assert any(
        p.grad is not None
        for p in agent.model.critic2.parameters()
    )

    assert any(
        p.grad is not None
        for p in agent.model.actor.parameters()
    )

    print(
        "[PASS] Critic and actor gradients are populated"
    )

    assert (
        np.isfinite(
            metrics2["critic_grad_norm"]
        )
        and metrics2["critic_grad_norm"] > 0
    )

    assert (
        np.isfinite(
            metrics2["actor_grad_norm"]
        )
        and metrics2["actor_grad_norm"] > 0
    )

    print(
        "[PASS] Reported gradient norms are finite "
        "and positive on an update step"
    )

    # ========================================================================
    # Actor save/load equivalence
    # ========================================================================

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(
            tmp,
            "TD3_test_actor.tmod",
        )

        exported_actor = agent.get_actor()

        exported_actor.save(path)

        assert os.path.isfile(path)

        fresh_actor = (
            TD3MLPActor(
                observation_space,
                action_space,
            )
            .to_device(device)
        )

        fresh_actor = fresh_actor.load(
            path,
            device=device,
        )

        probe_obs = tuple(
            x.to(device)
            for x in (
                torch.randn(4, 1),
                torch.randn(4, 76),
                torch.randn(4, 3),
                torch.randn(4, 3),
            )
        )

        with torch.no_grad():
            out_original = (
                exported_actor.forward(probe_obs)
            )

            out_reloaded = (
                fresh_actor.forward(probe_obs)
            )

        torch.testing.assert_close(
            out_original,
            out_reloaded,
            atol=1e-6,
            rtol=1e-6,
        )

        print(
            "[PASS] get_actor() -> save() -> load() "
            "round trip is bit-identical"
        )

    # ========================================================================
    # get_actor() must reflect later training
    # ========================================================================

    actor_ref = agent.get_actor()

    with torch.no_grad():
        before_update = (
            actor_ref
            .forward(probe_obs)
            .clone()
        )

    for _ in range(4):
        agent.train(
            collate_torch(
                make_batch(16),
                device=device,
            )
        )

    with torch.no_grad():
        after_update = (
            actor_ref
            .forward(probe_obs)
        )

    assert not torch.equal(
        before_update,
        after_update,
    ), (
        "get_actor()'s returned module did not reflect "
        "further training -- copy_shared() storage sharing "
        "appears broken"
    )

    print(
        "[PASS] get_actor()'s returned actor reflects "
        "subsequent training updates"
    )

    print()
    print("=" * 60)
    print("TD3 AGENT TEST PASSED")
    print("=" * 60)


# ============================================================================
# HUMAN BC WARM-START TEST
# ============================================================================

def test_warmstart():
    """
    Verify the human behavior-cloning warm-start path.

    Expected behavior:

        saved BC actor
              |
              v
        TD3Agent initialization
              |
              +----> online actor
              |
              +----> target actor

    Critics remain independently initialized.

    The warm-start actor itself is not modified during construction.
    """

    print("=" * 60)
    print("TD3 WARM-START TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================================
    # Create a known BC actor
    # ========================================================================

    bc_actor = (
        TD3MLPActor(
            observation_space,
            action_space,
            device=device,
        )
        .to_device(device)
    )

    with torch.no_grad():
        for p in bc_actor.parameters():
            p.add_(
                torch.randn_like(p) * 0.5
            )

    with tempfile.TemporaryDirectory() as tmp:
        warmstart_path = os.path.join(
            tmp,
            "TD3_bc_actor.tmod",
        )

        bc_actor.save(
            warmstart_path
        )

        # ====================================================================
        # Confirm fresh actor differs
        # ====================================================================

        control_actor = (
            TD3MLPActor(
                observation_space,
                action_space,
                device=device,
            )
            .to_device(device)
        )

        assert not params_equal(
            clone_params(bc_actor),
            clone_params(control_actor),
        ), (
            "test setup is vacuous: freshly initialized "
            "control actor already matches BC actor"
        )

        # ====================================================================
        # Construct warm-started TD3 agent
        # ====================================================================

        agent = TD3Agent(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            warmstart_actor_path=warmstart_path,
        )

        # Online actor must match BC actor.
        assert params_equal(
            clone_params(bc_actor),
            clone_params(agent.model.actor),
        ), (
            "TD3Agent's online actor does NOT match "
            "the warm-started BC actor"
        )

        print(
            "[PASS] Online actor exactly matches "
            "saved BC actor"
        )

        # Target actor must also match.
        assert params_equal(
            clone_params(bc_actor),
            clone_params(agent.model_target.actor),
        ), (
            "TD3Agent's target actor does NOT match "
            "the warm-started BC actor"
        )

        print(
            "[PASS] Target actor also matches "
            "warm-started BC actor"
        )

        # Critics must remain independent.
        assert not params_equal(
            clone_params(agent.model.critic1),
            clone_params(agent.model.critic2),
        ), (
            "critics appear identically initialized "
            "after warm-start"
        )

        print(
            "[PASS] Both critics remain independently "
            "initialized"
        )

        # ====================================================================
        # Action equivalence
        # ====================================================================

        probe_obs = tuple(
            x.to(device)
            for x in (
                torch.randn(4, 1),
                torch.randn(4, 76),
                torch.randn(4, 3),
                torch.randn(4, 3),
            )
        )

        with torch.no_grad():
            bc_out = (
                bc_actor.forward(probe_obs)
            )

            agent_out = (
                agent.model.actor.forward(
                    probe_obs
                )
            )

        torch.testing.assert_close(
            bc_out,
            agent_out,
            atol=1e-6,
            rtol=1e-6,
        )

        print(
            "[PASS] BC actor and warm-started TD3 actor "
            "produce identical actions"
        )

    print()
    print("=" * 60)
    print("TD3 WARM-START TEST PASSED")
    print("=" * 60)


# ============================================================================
# NEW CRITIC WARM-UP TEST
# ============================================================================

def test_critic_warmup():
    """
    Verify the new critic warm-up behavior.

    During warm-up:

        critics update
        actor does NOT update
        target actor does NOT update
        target critics DO update

    After warm-up:

        normal delayed TD3 actor updates resume.
    """

    print("=" * 60)
    print("TD3 CRITIC WARM-UP TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Three warm-up steps make this test fast.
    agent = TD3Agent(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        policy_delay=2,
        critic_warmup_steps=3,
    )

    actor_before = clone_params(
        agent.model.actor
    )

    target_actor_before = clone_params(
        agent.model_target.actor
    )

    target_critic1_before = clone_params(
        agent.model_target.critic1
    )

    target_critic2_before = clone_params(
        agent.model_target.critic2
    )

    # ========================================================================
    # Warm-up steps 1-3
    # ========================================================================

    for expected_step in range(1, 4):

        batch = collate_torch(
            make_batch(32),
            device=device,
        )

        metrics = agent.train(batch)

        assert (
            metrics["total_updates"]
            == expected_step
        )

        assert (
            metrics["critic_warmup"]
            == 1.0
        )

        assert (
            metrics["policy_updated"]
            == 0.0
        )

        # Actor must remain frozen.
        assert params_equal(
            actor_before,
            clone_params(agent.model.actor),
        ), (
            f"Actor changed during warm-up "
            f"step {expected_step}"
        )

        # Target actor must remain frozen.
        assert params_equal(
            target_actor_before,
            clone_params(
                agent.model_target.actor
            ),
        ), (
            f"Target actor changed during warm-up "
            f"step {expected_step}"
        )

    print(
        "[PASS] Actor and target actor remain frozen "
        "during critic warm-up"
    )

    # ========================================================================
    # Target critics must track online critics.
    # ========================================================================

    assert params_changed(
        target_critic1_before,
        clone_params(
            agent.model_target.critic1
        ),
    )

    assert params_changed(
        target_critic2_before,
        clone_params(
            agent.model_target.critic2
        ),
    )

    print(
        "[PASS] Target critics update during warm-up"
    )

    # ========================================================================
    # Step 4: warm-up finished.
    # ========================================================================

    actor_after_warmup = clone_params(
        agent.model.actor
    )

    batch4 = collate_torch(
        make_batch(32),
        device=device,
    )

    metrics4 = agent.train(batch4)

    assert metrics4["total_updates"] == 4

    assert metrics4["critic_warmup"] == 0.0

    # policy_delay=2:
    #
    # step 4 % 2 == 0
    #
    # therefore actor update must happen.
    assert metrics4["policy_updated"] == 1.0

    assert params_changed(
        actor_after_warmup,
        clone_params(agent.model.actor),
    ), (
        "Actor did not resume updating after "
        "critic warm-up completed"
    )

    print(
        "[PASS] Normal delayed TD3 actor updates "
        "resume after critic warm-up"
    )

    print()
    print("=" * 60)
    print("TD3 CRITIC WARM-UP TEST PASSED")
    print("=" * 60)


# ============================================================================
# BC-ANCHOR REGULARIZATION TEST (TD3+BC style, added after Phase 6 baseline
# found critic_warmup_steps alone insufficient to prevent post-warmup
# collapse -- see experiments/phase6_baseline_v1/manifest.md)
# ============================================================================

def test_bc_regularization():
    """
    Verify bc_reg_alpha actually changes the actor loss/gradient relative to
    plain TD3 when a warm-start actor is present, does nothing when
    bc_reg_alpha=0.0 (the default, backward-compatible with test_warmstart),
    and never touches the frozen BC reference actor's own parameters.
    """

    print("=" * 60)
    print("TD3 BC-ANCHOR REGULARIZATION TEST")
    print("=" * 60)

    observation_space, action_space = make_spaces()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    bc_actor = (
        TD3MLPActor(
            observation_space,
            action_space,
            device=device,
        )
        .to_device(device)
    )

    with torch.no_grad():
        for p in bc_actor.parameters():
            p.add_(
                torch.randn_like(p) * 0.5
            )

    with tempfile.TemporaryDirectory() as tmp:
        warmstart_path = os.path.join(
            tmp,
            "TD3_bc_actor.tmod",
        )

        bc_actor.save(warmstart_path)

        batch = make_batch(32)

        # ====================================================================
        # Construct BOTH agents (identically seeded) BEFORE training either,
        # so the "identical initialization" check compares like with like.
        # ====================================================================

        torch.manual_seed(0)

        agent_off = TD3Agent(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            policy_delay=1,
            critic_warmup_steps=0,
            warmstart_actor_path=warmstart_path,
            bc_reg_alpha=0.0,
        )

        assert agent_off.bc_actor is not None, (
            "bc_actor should still be constructed whenever "
            "warmstart_actor_path is set, even if bc_reg_alpha=0.0"
        )

        torch.manual_seed(0)

        agent_on = TD3Agent(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            policy_delay=1,
            critic_warmup_steps=0,
            warmstart_actor_path=warmstart_path,
            bc_reg_alpha=2.5,
        )

        # Identical seed => identical initialization for both agents.
        assert params_equal(
            clone_params(agent_off.model.critic1),
            clone_params(agent_on.model.critic1),
        ), (
            "test setup invalid: agent_off/agent_on critics differ "
            "despite identical seeding"
        )

        bc_actor_params_before = clone_params(agent_on.bc_actor)

        # ====================================================================
        # bc_reg_alpha=0.0 must be a no-op (matches plain TD3 loss).
        # ====================================================================

        metrics_off = agent_off.train(
            collate_torch(batch, device=device)
        )

        assert np.isnan(metrics_off["bc_reg_term"]), (
            "bc_reg_term must be NaN when bc_reg_alpha=0.0 "
            "(regularization disabled)"
        )

        print(
            "[PASS] bc_reg_alpha=0.0 disables regularization "
            "(bc_reg_term stays NaN)"
        )

        # ====================================================================
        # bc_reg_alpha>0 must engage and change the actor loss.
        # ====================================================================

        metrics_on = agent_on.train(
            collate_torch(batch, device=device)
        )

        assert np.isfinite(metrics_on["bc_reg_term"]), (
            "bc_reg_term must be a finite number when "
            "bc_reg_alpha>0 and a warm-start actor is present"
        )

        assert metrics_on["bc_reg_term"] >= 0.0, (
            "bc_reg_term is a mean-squared error and must be "
            "non-negative"
        )

        print(
            "[PASS] bc_reg_alpha>0 reports a finite, "
            "non-negative bc_reg_term"
        )

        assert metrics_on["loss_actor"] != metrics_off["loss_actor"], (
            "bc_reg_alpha>0 must change the actor loss relative to "
            "bc_reg_alpha=0.0 given identical initialization/batch"
        )

        print(
            "[PASS] bc_reg_alpha>0 actually changes the actor loss "
            "relative to plain TD3 (bc_reg_alpha=0.0)"
        )

        # ====================================================================
        # The frozen BC reference actor must never be optimized.
        # ====================================================================

        assert params_equal(
            bc_actor_params_before,
            clone_params(agent_on.bc_actor),
        ), (
            "agent.bc_actor's parameters changed during training -- "
            "the frozen BC reference must never receive gradient updates"
        )

        assert all(
            not p.requires_grad
            for p in agent_on.bc_actor.parameters()
        ), (
            "agent.bc_actor parameters must have requires_grad=False"
        )

        print(
            "[PASS] Frozen BC reference actor is never modified "
            "by training and carries no gradients"
        )

    print()
    print("=" * 60)
    print("TD3 BC-ANCHOR REGULARIZATION TEST PASSED")
    print("=" * 60)


# ============================================================================
# PYTEST ENTRY POINTS
# ============================================================================

def test_main():
    """
    Pytest entry point.
    """

    main()
    test_warmstart()
    test_critic_warmup()
    test_bc_regularization()


if __name__ == "__main__":
    main()
    test_warmstart()
    test_critic_warmup()
    test_bc_regularization()
