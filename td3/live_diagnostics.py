"""Standalone, deterministic TrackMania policy-forensics utilities.

These diagnostics intentionally do not construct a Trainer, do not connect to
the Server, do not send replay samples, and never attach exploration noise.
They answer one narrow question: which actor and actions control the car.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from tmrl.networking import RolloutWorker

from td3.config import (
    TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER,
    TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER,
    TD3_ENV_CLS,
    TD3_HUMAN_WARMSTART_AGENT,
    TD3_HUMAN_WARMSTART_ACTOR_PATH,
    TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
    TD3_OBS_PREPROCESSOR,
    TD3_SAMPLE_COMPRESSOR,
)
from td3.models import TD3MLPActor


def actor_fingerprint(actor):
    """Stable SHA-256 digest of actor parameter names, metadata and bytes."""
    digest = hashlib.sha256()
    parameter_count = 0
    for name, tensor in actor.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        parameter_count += tensor.numel()
    return {
        "sha256": digest.hexdigest(),
        "parameter_count": parameter_count,
    }


def _make_actor(path, device):
    """Load an actor exactly as RolloutWorker does, without a network endpoint."""
    with TD3_ENV_CLS() as env:
        actor = TD3MLPActor(env.observation_space, env.action_space)
    return actor.to_device(device).load(path, device=device)


def verify_artifacts(device=None):
    """Prove canonical/worker/trainer actor parameter and action identity."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = {
        "canonical": TD3_HUMAN_WARMSTART_ACTOR_PATH,
        "worker": TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
    }
    actors = {name: _make_actor(path, device) for name, path in paths.items()}

    # A Trainer does not load a seed `.tmod`: it builds TD3Agent and loads the
    # canonical actor through `warmstart_actor_path`. Its `.tmod` path is a
    # mutable broadcast output and must never be used as a BC identity check.
    trainer_seed = TD3_HUMAN_WARMSTART_AGENT(
        observation_space=actors["canonical"].observation_space,
        action_space=actors["canonical"].action_space,
    )
    actors["trainer_seed"] = trainer_seed.model.actor
    fingerprints = {name: actor_fingerprint(actor) for name, actor in actors.items()}

    from td3.demonstration import _load_human_pairs
    observations, _ = _load_human_pairs()
    obs = tuple(torch.from_numpy(x[:1000]).to(device) for x in observations)
    with torch.no_grad():
        outputs = {name: actor(obs) for name, actor in actors.items()}

    reference = outputs["canonical"]
    action_deltas = {
        name: {
            "max_abs": float((output - reference).abs().max().item()),
            "mean_abs": float((output - reference).abs().mean().item()),
            "mse": float(torch.mean((output - reference) ** 2).item()),
        }
        for name, output in outputs.items()
        if name != "canonical"
    }
    print("BC ACTOR IDENTITY")
    for name in paths:
        print(f"{name:9s} {fingerprints[name]['sha256']} "
              f"parameters={fingerprints[name]['parameter_count']}")
    print("ACTION DELTAS ON 1,000 HUMAN OBSERVATIONS")
    for name, delta in action_deltas.items():
        print(f"{name:9s} max={delta['max_abs']:.8e} "
              f"mean={delta['mean_abs']:.8e} mse={delta['mse']:.8e}")
    if len({item["sha256"] for item in fingerprints.values()}) != 1:
        raise RuntimeError("BC artifact fingerprints differ.")
    if any(delta["max_abs"] > 1e-6 for delta in action_deltas.values()):
        raise RuntimeError("BC artifact actions differ.")
    return fingerprints, action_deltas


def run_deterministic_policy(policy, episodes=1, max_steps=200, trace_steps=100):
    """Run BC, random, or the current trained policy directly against TrackMania
    without RL machinery. 'trained' loads the trainer's live worker.tmod
    (TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER), i.e. whatever the most recent
    training run last broadcast -- used to inspect the actor's *current*
    behavior (e.g. comparing a short/failed episode against a long/successful
    one), as opposed to 'bc' which is always the original frozen BC actor.
    """
    if policy not in {"bc", "random", "trained"}:
        raise ValueError("policy must be 'bc', 'random', or 'trained'")

    if policy == "bc":
        model_path = TD3_HUMAN_WARMSTART_ACTOR_PATH
        folder = Path(TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER)
    elif policy == "trained":
        from td3.config import TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER
        model_path = TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER
        folder = Path(TD3_CLEAN_HUMAN_EXPERIMENT_FOLDER)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained worker checkpoint found: {model_path}. "
                "Run the trainer/worker (--human-warmstart) first."
            )
    else:
        folder = Path(TD3_CLEAN_RANDOM_EXPERIMENT_FOLDER)
        model_path = str(folder / "DIAGNOSTIC_RANDOM_UNTRAINED_DO_NOT_CREATE.tmod")
        if Path(model_path).exists():
            raise RuntimeError("Random diagnostic path unexpectedly exists; refusing stale model.")

    trace_path = folder / f"{policy}_deterministic_trace.csv"
    rows = []

    worker = RolloutWorker(
        env_cls=TD3_ENV_CLS,
        actor_module_cls=TD3MLPActor,
        sample_compressor=TD3_SAMPLE_COMPRESSOR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=model_path,
        obs_preprocessor=TD3_OBS_PREPROCESSOR,
        standalone=True,
    )

    worker.actor.trace_callback = rows.append
    print(f"POLICY={policy}")
    print(f"MODEL_PATH={model_path}")
    print(f"FINGERPRINT={actor_fingerprint(worker.actor)}")
    print("EXPLORATION=disabled (all calls use test=True)")

    summaries = []
    try:
        for episode in range(episodes):
            obs, _ = worker.reset(collect_samples=False)
            total_reward = 0.0
            for step in range(max_steps):
                obs, reward, terminated, truncated, info = worker.step(
                    obs=obs,
                    test=True,
                    collect_samples=False,
                )
                row = rows[-1]
                row.update({
                    "episode": episode,
                    "step": step,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                total_reward += reward
                if terminated or truncated:
                    break
            summaries.append({
                "episode": episode,
                "return": float(total_reward),
                "steps": step + 1,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        try:
            worker.env.unwrapped.wait()
        except Exception:
            pass

    fields = [
        "episode", "step", "test", "speed", "lidar_min", "lidar_mean",
        "lidar_max", "previous_actions", "raw_action", "applied_noise",
        "final_action", "reward", "terminated", "truncated",
    ]
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[:trace_steps])

    print("EPISODE SUMMARIES")
    for summary in summaries:
        print(summary)
    print(f"TRACE={trace_path}")
    return summaries, trace_path
