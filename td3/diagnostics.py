"""
TD3 HUMAN BC ACTOR AUDIT + SAVE/LOAD DIAGNOSTICS

This diagnostic answers the most important question before starting TD3:

    Does the behavior-cloned actor actually reproduce human driving
    when given REAL observations from dataset_human?

It performs four checks:

1. REAL HUMAN DATA AUDIT
   Human observations -> BC actor -> predicted actions
   compared directly against recorded human actions.

2. BC ACTION STATISTICS
   Compares human and BC distributions for:
       gas / throttle
       brake / reverse
       steering

3. SAVE/LOAD EQUIVALENCE
   Verifies the saved BC actor and worker actor produce identical
   outputs on the SAME REAL human observations.

4. EXPLORATION CHECK
   Shows deterministic BC action vs action after exploration noise.

IMPORTANT:
This diagnostic does NOT modify training, replay buffers, checkpoints,
or the human dataset.

Run:

    python train_td3.py diagnose
"""

from pathlib import Path

import numpy as np
import torch

from td3.models import TD3MLPActor
from td3.noise import GaussianExplorationNoise
from td3.config import (
    TD3_ENV_CLS,
    TD3_HUMAN_WARMSTART_ACTOR_PATH,
    TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
)
from td3.demonstration import _load_human_pairs


# --------------------------------------------------------------------------------------
# Environment spaces
# --------------------------------------------------------------------------------------

def _synthetic_spaces():
    from gymnasium.spaces import Box, Tuple

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


def _get_spaces():
    try:
        with TD3_ENV_CLS() as env:
            return (
                env.observation_space,
                env.action_space,
                "live TD3_ENV_CLS()",
            )
    except Exception as e:
        print(
            f"[INFO] Could not construct live TrackMania environment "
            f"({type(e).__name__}: {e})."
        )
        print("[INFO] Using synthetic spaces with the known LIDAR shapes.")

        observation_space, action_space = _synthetic_spaces()

        return (
            observation_space,
            action_space,
            "synthetic fallback spaces",
        )


# --------------------------------------------------------------------------------------
# Tensor helpers
# --------------------------------------------------------------------------------------

def _to_device_obs(obs_np, device):
    """
    Convert the real human observations loaded by _load_human_pairs()
    into the exact batched tensor tuple expected by TD3MLPActor.forward().
    """

    return tuple(
        torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        for x in obs_np
    )


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def _print_action_statistics(name, actions):
    actions = np.asarray(actions, dtype=np.float32)

    print()
    print(f"{name} ACTION STATISTICS")
    print("-" * 60)

    labels = ["gas/reverse", "brake/reverse", "steering"]

    for i, label in enumerate(labels):
        x = actions[:, i]

        print(
            f"{label:16s} "
            f"min={x.min(): .4f}  "
            f"max={x.max(): .4f}  "
            f"mean={x.mean(): .4f}  "
            f"std={x.std(): .4f}"
        )


def _compare_actions(human_actions, predicted_actions):
    human = np.asarray(human_actions, dtype=np.float32)
    pred = np.asarray(predicted_actions, dtype=np.float32)

    diff = pred - human

    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)

    print()
    print("REAL-DATA BC PERFORMANCE")
    print("-" * 60)

    labels = ["gas/reverse", "brake/reverse", "steering"]

    for i, label in enumerate(labels):
        h = human[:, i]
        p = pred[:, i]

        if np.std(h) > 1e-8 and np.std(p) > 1e-8:
            correlation = float(np.corrcoef(h, p)[0, 1])
        else:
            correlation = float("nan")

        print(
            f"{label:16s} "
            f"MSE={mse[i]:.6f}  "
            f"MAE={mae[i]:.6f}  "
            f"corr={correlation:.4f}"
        )

    overall_mse = float(np.mean((pred - human) ** 2))
    overall_mae = float(np.mean(np.abs(pred - human)))

    print()
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")

    return {
        "mse": mse,
        "mae": mae,
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
    }


# --------------------------------------------------------------------------------------
# Main diagnostic
# --------------------------------------------------------------------------------------

def run_diagnostics(human_warmstart=True, device=None, seed=0):
    del human_warmstart
    del seed

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TD3 HUMAN BC ACTOR AUDIT")
    print("=" * 70)
    print(f"Device: {device}")

    # ----------------------------------------------------------------------------------
    # 1. Load REAL human observations/actions
    # ----------------------------------------------------------------------------------

    print()
    print("[1/4] Loading REAL human demonstration data...")

    obs_np, human_actions = _load_human_pairs()

    print(f"Human transitions: {len(human_actions)}")

    if len(human_actions) == 0:
        raise RuntimeError("No human transitions available.")

    # ----------------------------------------------------------------------------------
    # 2. Load environment spaces
    # ----------------------------------------------------------------------------------

    observation_space, action_space, space_source = _get_spaces()

    print(f"Spaces: {space_source}")
    print(f"Observation space: {observation_space}")
    print(f"Action space:      {action_space}")

    # ----------------------------------------------------------------------------------
    # 3. Load saved BC actor
    # ----------------------------------------------------------------------------------

    print()
    print("[2/4] Loading saved BC actor...")

    if not Path(TD3_HUMAN_WARMSTART_ACTOR_PATH).is_file():
        raise FileNotFoundError(
            "BC actor not found:\n"
            f"{TD3_HUMAN_WARMSTART_ACTOR_PATH}\n\n"
            "Run first:\n"
            "python train_td3.py pretrain-human --force"
        )

    bc_actor = TD3MLPActor(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    ).to_device(device)

    bc_actor = bc_actor.load(
        TD3_HUMAN_WARMSTART_ACTOR_PATH,
        device=device,
    )

    bc_actor.eval()

    print(f"Loaded: {TD3_HUMAN_WARMSTART_ACTOR_PATH}")

    # ----------------------------------------------------------------------------------
    # 4. Run BC actor over REAL human observations
    # ----------------------------------------------------------------------------------

    print()
    print("[3/4] Running BC actor over REAL human observations...")

    predicted_chunks = []

    obs_tensors = _to_device_obs(obs_np, device)

    batch_size = 512

    with torch.no_grad():
        for start in range(0, len(human_actions), batch_size):
            end = min(start + batch_size, len(human_actions))

            batch_obs = tuple(
                x[start:end]
                for x in obs_tensors
            )

            predicted = bc_actor.forward(batch_obs)

            predicted_chunks.append(
                predicted.detach().cpu().numpy()
            )

    predicted_actions = np.concatenate(
        predicted_chunks,
        axis=0,
    ).astype(np.float32)

    predicted_actions = np.clip(
        predicted_actions,
        -1.0,
        1.0,
    )

    print(
        f"Predicted actions: {predicted_actions.shape}"
    )

    # ----------------------------------------------------------------------------------
    # Human vs BC statistics
    # ----------------------------------------------------------------------------------

    _print_action_statistics(
        "HUMAN",
        human_actions,
    )

    _print_action_statistics(
        "BC ACTOR",
        predicted_actions,
    )

    comparison = _compare_actions(
        human_actions,
        predicted_actions,
    )

    # ----------------------------------------------------------------------------------
    # 5. Check save/load using REAL observations
    # ----------------------------------------------------------------------------------

    print()
    print("[4/4] Checking BC save/load equivalence on REAL observations...")

    if not Path(TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER).is_file():
        raise FileNotFoundError(
            "BC worker model not found:\n"
            f"{TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER}"
        )

    worker_actor = TD3MLPActor(
        observation_space=observation_space,
        action_space=action_space,
    ).to_device(device)

    identity_before = id(worker_actor)

    worker_actor = worker_actor.load(
        TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
        device=device,
    )

    identity_after = id(worker_actor)

    print(
        f"RolloutWorker-style load mutates same object: "
        f"{identity_before == identity_after}"
    )

    # Use a deterministic subset of REAL observations.
    check_count = min(1000, len(human_actions))

    real_obs_check = tuple(
        x[:check_count]
        for x in obs_tensors
    )

    with torch.no_grad():
        bc_check = bc_actor.forward(real_obs_check)
        worker_check = worker_actor.forward(real_obs_check)

    max_diff = (
        bc_check - worker_check
    ).abs().max().item()

    mean_diff = (
        bc_check - worker_check
    ).abs().mean().item()

    print(
        f"Max BC-vs-worker action difference: "
        f"{max_diff:.8e}"
    )

    print(
        f"Mean BC-vs-worker action difference: "
        f"{mean_diff:.8e}"
    )

    save_load_ok = max_diff < 1e-5

    print(
        f"Save/load equivalence: {save_load_ok}"
    )

    # ----------------------------------------------------------------------------------
    # Exploration sanity check
    # ----------------------------------------------------------------------------------

    single_obs = tuple(
        x[:1]
        for x in obs_tensors
    )

    deterministic_action = worker_actor.act(
        single_obs,
        test=True,
    )

    worker_actor.noise = GaussianExplorationNoise(
        action_dim=3,
        sigma=0.1,
    )

    exploring_action = worker_actor.act(
        single_obs,
        test=False,
    )

    deterministic_again = worker_actor.act(
        single_obs,
        test=True,
    )

    print()
    print("EXPLORATION SANITY CHECK")
    print("-" * 60)
    print(
        f"Real human observation deterministic action: "
        f"{deterministic_action}"
    )
    print(
        f"Same observation + exploration noise:         "
        f"{exploring_action}"
    )
    print(
        f"Same observation + test=True again:           "
        f"{deterministic_again}"
    )

    noise_changes_action = not np.allclose(
        deterministic_action,
        exploring_action,
    )

    test_stays_deterministic = np.allclose(
        deterministic_action,
        deterministic_again,
        atol=1e-6,
    )

    print(
        f"Exploration noise changes action: "
        f"{noise_changes_action}"
    )

    print(
        f"test=True ignores exploration noise: "
        f"{test_stays_deterministic}"
    )

    # ----------------------------------------------------------------------------------
    # Final verdict
    # ----------------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DIAGNOSTIC VERDICT")
    print("=" * 70)

    if not save_load_ok:
        print("CONFIRMED BUG: BC save/load mismatch.")
        print(
            "The saved actor and worker actor produce different "
            "actions on identical REAL observations."
        )

    elif not test_stays_deterministic:
        print("CONFIRMED BUG: exploration noise leaks into test=True.")

    elif comparison["overall_mse"] < 0.05:
        print("BC POLICY LOOKS GOOD ON REAL HUMAN DATA.")
        print()
        print(
            "The actor is successfully reproducing the human "
            "demonstration observations/actions."
        )
        print(
            "If gameplay becomes random after TD3 starts, "
            "the next suspect is TD3 optimization destabilizing "
            "the warm-start actor."
        )

    elif comparison["overall_mse"] < 0.15:
        print("BC POLICY IS PARTIALLY FITTED.")
        print()
        print(
            "The actor learned some human behavior, but the "
            "real-data action error is still significant."
        )
        print(
            "Do NOT start TD3 yet. We should inspect BC training "
            "before modifying the RL algorithm."
        )

    else:
        print("BC POLICY LOOKS WEAK ON REAL HUMAN DATA.")
        print()
        print(
            "The actor is not reproducing the human demonstrations "
            "well enough to trust the warm-start."
        )
        print(
            "Do NOT change TD3 yet. Fix/inspect behavior cloning first."
        )

    print("=" * 70)

    return {
        "transitions": len(human_actions),
        "overall_mse": comparison["overall_mse"],
        "overall_mae": comparison["overall_mae"],
        "save_load_ok": save_load_ok,
        "noise_changes_action": noise_changes_action,
        "test_stays_deterministic": test_stays_deterministic,
    }


if __name__ == "__main__":
    run_diagnostics()