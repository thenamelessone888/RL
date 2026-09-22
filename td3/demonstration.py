"""
Human-demonstration behavior-cloning warm start for the TD3 actor.

Reuses the shared recorded human dataset (demonstrations.human_interface.HUMAN_DATASET_PATH)
-- the recorded (obs, action) pairs are just human driving telemetry, entirely independent of
which RL algorithm will later be warm-started from them. We do not re-record a second human
dataset for TD3 (see td3/config.py's module docstring for the same point re: dataset paths).
Only the OUTPUT of this script -- the behavior-cloned TD3MLPActor's weights -- gets
TD3-specific paths, since TD3MLPActor has its own (non-interchangeable) state_dict namespace.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from td3.config import (
    TD3_ENV_CLS,
    TD3_HUMAN_WARMSTART_ACTOR_PATH,
    TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
)
from demonstrations.human_interface import HUMAN_DATASET_PATH
from td3.models import TD3MLPActor


def _load_human_pairs():
    from tmrl.custom.custom_memories import MemoryTMLidar
    import tmrl.config.config_constants as cfg

    memory = MemoryTMLidar(
        memory_size=100_000_000,
        batch_size=1,
        dataset_path=HUMAN_DATASET_PATH,
        imgs_obs=cfg.IMG_HIST_LEN,
        act_buf_len=cfg.ACT_BUF_LEN,
        nb_steps=1,
        sample_preprocessor=None,
        crc_debug=False,
        device="cpu",
    )

    if len(memory) == 0:
        raise RuntimeError(
            f"No usable human transitions found in {HUMAN_DATASET_PATH}. "
            "Record demonstrations first (`python train_td3.py record`)."
        )

    obs_parts = [[] for _ in range(4)]
    actions = []

    for i in range(len(memory)):
        obs, action, _, _, _, _, _ = memory.get_transition(i)
        for j in range(4):
            obs_parts[j].append(np.asarray(obs[j], dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))

    obs_arrays = tuple(np.stack(x) for x in obs_parts)
    actions = np.stack(actions).astype(np.float32)

    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"Expected human actions with shape (N,3), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Human action dataset contains NaN/Inf")
    actions = np.clip(actions, -1.0, 1.0)

    return obs_arrays, actions


def pretrain_human(epochs=20, batch_size=256, learning_rate=1e-4, device=None, force=False):
    """Behavior-clone the TD3 actor from human demonstrations and save actor weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not force and (
        Path(TD3_HUMAN_WARMSTART_ACTOR_PATH).exists()
        or Path(TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER).exists()
        ):
        raise FileExistsError(
            f"Warm-start actor already exists: {TD3_HUMAN_WARMSTART_ACTOR_PATH}. "
            "Use --force to intentionally replace it."
        )

    print("=" * 60)
    print("HUMAN DEMONSTRATION BEHAVIOR CLONING (TD3 actor)")
    print("=" * 60)
    print(f"Dataset: {HUMAN_DATASET_PATH}")
    print(f"Device:  {device}")

    if force:
        # Only remove files belonging to the dedicated TD3 human-warm-start run.
        for path in (
            TD3_HUMAN_WARMSTART_ACTOR_PATH,
            TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
            ):
            Path(path).unlink(missing_ok=True)

    obs_np, actions_np = _load_human_pairs()
    print(f"Usable demonstration transitions: {len(actions_np)}")

    # The actor expects a tuple of batched tensors matching the TMRL LIDAR policy input.
    tensors = [torch.from_numpy(x) for x in obs_np]
    target = torch.from_numpy(actions_np)
    dataset = TensorDataset(*tensors, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Use the real environment spaces so the model exactly matches the TD3 worker.
    with TD3_ENV_CLS() as env:
        observation_space = env.observation_space
        action_space = env.action_space

    actor = TD3MLPActor(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    ).to_device(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)

    actor.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        for *obs_batch, action_batch in loader:
            obs_batch = tuple(x.to(device) for x in obs_batch)
            action_batch = action_batch.to(device)

            pred = actor.forward(obs_batch)
            loss = F.mse_loss(pred, action_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = action_batch.shape[0]
            total_loss += loss.detach().item() * n
            count += n

        mean_loss = total_loss / max(count, 1)
        print(f"[BC] epoch {epoch:03d}/{epochs}  loss={mean_loss:.6f}")

    actor.eval()
    for path in (
        TD3_HUMAN_WARMSTART_ACTOR_PATH,
        TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        actor.save(path)

    print()
    print("[DONE] Human warm-start TD3 actor saved:")
    print(f"  actor:   {TD3_HUMAN_WARMSTART_ACTOR_PATH}")
    print(f"  worker:  {TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER}")
    print("  trainer: initialized directly from the canonical actor")
    print()
    print("Both critics were NOT pretrained. Normal TD3 will train them from dataset_td3.")
    print("Use the human-warm-start trainer/worker mode for a fresh TD3 run.")

    return {
        "transitions": len(actions_np),
        "epochs": epochs,
        "device": device,
        "actor_path": TD3_HUMAN_WARMSTART_ACTOR_PATH,
    }
