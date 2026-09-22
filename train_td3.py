
"""
TD3 + TMRL 0.7.1 launcher.

Run three separate PowerShell windows:

    python train_td3.py server

    python train_td3.py trainer

    python train_td3.py worker

The existing SAC configuration/model/checkpoint files, and this project's existing DDPG
run (train_ddpg.py), are never used as TD3 output paths -- see td3/config.py's isolation
asserts, which are executed at import time.

Human demonstration recording/inspection use the shared, algorithm-agnostic tooling in
demonstrations/ (demonstrations/human_recorder.py, demonstrations/human_dataset.py) since
recorded human driving data is not algorithm-specific -- see td3/demonstration.py's module
docstring. Only `pretrain-human` is algorithm-specific (it behavior-clones a TD3MLPActor).
"""

import argparse
import time

from tmrl.networking import Server, Trainer, RolloutWorker

import tmrl.config.config_constants as cfg

from td3.config import (
    TD3_TRAINER,
    TD3_ENV_CLS,
    TD3_SAMPLE_COMPRESSOR,
    TD3_OBS_PREPROCESSOR,

    TD3_MODEL_PATH_WORKER,
    TD3_MODEL_PATH_SAVE_HISTORY,
    TD3_MODEL_PATH_TRAINER,
    TD3_CHECKPOINT_PATH,
    TD3_HUMAN_WARMSTART_TRAINER,
    TD3_HUMAN_WARMSTART_CHECKPOINT_PATH,
    TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER,
    TD3_HUMAN_WARMSTART_MODEL_PATH_TRAINER,
    TD3_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY,

    TD3_CURRICULUM_TRAINER,
    TD3_CURRICULUM_CHECKPOINT_PATH,
    TD3_CURRICULUM_MODEL_PATH_WORKER,
    TD3_CURRICULUM_MODEL_PATH_TRAINER,
    TD3_CURRICULUM_MODEL_PATH_SAVE_HISTORY,
)

from td3.models import TD3MLPActor
from td3.noise import GaussianExplorationNoise


# ============================================================================
# SERVER
# ============================================================================

def run_server():
    print("=" * 60)
    print("TD3 SERVER")
    print("=" * 60)

    print(f"Port: {cfg.PORT}")
    print(f"Workers allowed: {cfg.NB_WORKERS}")

    server = Server(
        port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_SERVER,
        header_size=cfg.HEADER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        max_workers=cfg.NB_WORKERS,
    )

    # Keep the Server process alive.
    #
    # Server internally owns the networking relay. We simply keep this
    # process alive so workers and trainer can connect to it.

    print()
    print("[READY] TD3 Server is running.")
    print("[INFO] Start the trainer in another PowerShell window.")
    print("[INFO] Then start the worker.")
    print("[INFO] Press CTRL+C to stop.")
    print()

    try:
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
        print("[INFO] TD3 Server stopped.")


# ============================================================================
# TRAINER
# ============================================================================

def run_trainer(human_warmstart=False, curriculum=False):
    print("=" * 60)
    print("TD3 TRAINER" + (" (CURRICULUM)" if curriculum else ""))
    print("=" * 60)

    if curriculum:
        from curriculum.manager import CurriculumManager
        from curriculum import reward_registry
        manager = CurriculumManager()
        print(manager.status_string())
        print()
        reward_registry.activate(manager.current_stage.name)
        training_cls = TD3_CURRICULUM_TRAINER
        trainer_model = TD3_CURRICULUM_MODEL_PATH_TRAINER
        trainer_checkpoint = TD3_CURRICULUM_CHECKPOINT_PATH
        dataset_path = cfg.TMRL_FOLDER / "dataset_curriculum"
    elif human_warmstart:
        training_cls = TD3_HUMAN_WARMSTART_TRAINER
        trainer_model = TD3_HUMAN_WARMSTART_MODEL_PATH_TRAINER
        trainer_checkpoint = TD3_HUMAN_WARMSTART_CHECKPOINT_PATH
        dataset_path = cfg.TMRL_FOLDER / "dataset_td3"
    else:
        training_cls = TD3_TRAINER
        trainer_model = TD3_MODEL_PATH_TRAINER
        trainer_checkpoint = TD3_CHECKPOINT_PATH
        dataset_path = cfg.TMRL_FOLDER / "dataset_td3"

    print("Trainer model:")
    print(f"  {trainer_model}")

    print("Checkpoint:")
    print(f"  {trainer_checkpoint}")

    print(f"Dataset:")
    print(f"  {dataset_path}")

    print()

    trainer = Trainer(
        training_cls=training_cls,

        server_ip=cfg.SERVER_IP_FOR_TRAINER,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,

        local_com_port=cfg.LOCAL_PORT_TRAINER,

        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,

        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,

        # CRITICAL:
        # Explicit TD3 paths.
        #
        # Do NOT omit these because TMRL's defaults point at the
        # currently configured SAC paths.

        model_path=trainer_model,
        checkpoint_path=trainer_checkpoint,
    )

    print("[READY] TD3 Trainer connecting to Server...")
    print()

    trainer.run()


# ============================================================================
# WORKER
# ============================================================================

def run_pretrain_human(epochs, batch_size, learning_rate, force=False):
    from td3.demonstration import pretrain_human
    pretrain_human(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force=force,
    )


def run_worker(human_warmstart=False, curriculum=False):
    print("=" * 60)
    print("TD3 ROLLOUT WORKER" + (" (CURRICULUM)" if curriculum else ""))
    print("=" * 60)

    if curriculum:
        from curriculum.manager import CurriculumManager
        from curriculum import reward_registry
        manager = CurriculumManager()
        print(manager.status_string())
        print()
        reward_registry.activate(manager.current_stage.name)
        worker_model = TD3_CURRICULUM_MODEL_PATH_WORKER
        worker_history = TD3_CURRICULUM_MODEL_PATH_SAVE_HISTORY
    elif human_warmstart:
        worker_model = TD3_HUMAN_WARMSTART_MODEL_PATH_WORKER
        worker_history = TD3_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY
    else:
        worker_model = TD3_MODEL_PATH_WORKER
        worker_history = TD3_MODEL_PATH_SAVE_HISTORY

    print("Worker model:")
    print(f"  {worker_model}")

    print("Model history:")
    print(f"  {worker_history}")

    print()

    worker = RolloutWorker(
        env_cls=TD3_ENV_CLS,

        # Our actor, not TMRL SAC and not this project's DDPG.
        actor_module_cls=TD3MLPActor,

        # Existing TMRL LIDAR compressor (algorithm-agnostic).
        sample_compressor=TD3_SAMPLE_COMPRESSOR,

        device=(
            "cuda"
            if cfg.CUDA_INFERENCE
            else "cpu"
        ),

        # CRITICAL:
        # Explicit TD3 model path.
        model_path=worker_model,

        # Existing LIDAR observation preprocessing (algorithm-agnostic).
        obs_preprocessor=TD3_OBS_PREPROCESSOR,

        crc_debug=cfg.CRC_DEBUG,

        # TD3-specific model history.
        model_path_history=worker_history,

        model_history=cfg.MODEL_HISTORY,

        standalone=False,

        server_ip=cfg.SERVER_IP_FOR_WORKER,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,

        local_port=cfg.LOCAL_PORT_WORKER,

        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,

        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
    )

    # ========================================================================
    # TD3 EXPLORATION
    # ========================================================================

    # Noise is attached to OUR actor. TD3 uses decaying, temporally-uncorrelated Gaussian
    # exploration noise (NOT the project's existing DDPG Ornstein-Uhlenbeck noise -- see
    # td3/noise.py's module docstring).
    #
    # RolloutWorker:
    #
    #   training episode -> test=False
    #                    -> actor.act(..., test=False)
    #                    -> Gaussian noise enabled
    #
    #   evaluation       -> test=True
    #                    -> actor.act(..., test=True)
    #                    -> deterministic action

    worker.actor.noise = GaussianExplorationNoise(
        action_dim=3,
        sigma=0.1,           # TD3 paper default for a [-1, 1]-bounded action space
        sigma_min=0.02,
        decay_steps=200_000,
        decay_type="linear",
    )

    print("[READY] TD3 Worker created.")
    print("[INFO] Gaussian exploration enabled (TD3-style, decaying).")
    print("[INFO] Training episodes use exploration.")
    print("[INFO] Evaluation episodes are deterministic.")
    print()
    print("[START] Collecting TrackMania training episodes...")
    print()

    worker.run(
        test_episode_interval=10,
        nb_episodes=float("inf"),
        verbose=True,
    )


# ============================================================================
# HUMAN DEMONSTRATION RECORDING / INSPECTION
# ============================================================================
# Reuses the shared, algorithm-agnostic recorder/inspector (see module docstring above and
# td3/demonstration.py).

def run_record(episodes, minutes):
    from demonstrations.human_recorder import run_recorder
    run_recorder(episodes=episodes, minutes=minutes)


def run_inspect_human_data():
    from demonstrations.human_dataset import inspect_human_data
    inspect_human_data()


def run_diagnose():
    from td3.diagnostics import run_diagnostics
    run_diagnostics()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TMRL 0.7.1 custom TD3 launcher"
    )

    parser.add_argument(
        "mode",
        choices=[
            "server",
            "trainer",
            "worker",
            "record",
            "inspect-human-data",
            "pretrain-human",
            "diagnose",
            "verify-bc",
            "bc-eval",
            "random-eval",
            "trained-eval",
            "curriculum-trainer",
            "curriculum-worker",
            "curriculum-status",
            "curriculum-watch",
            "record-track-reward",
        ],
        help="Process to start",
    )

    parser.add_argument(
        "stage_name",
        nargs="?",
        default=None,
        help="(record-track-reward) name of the curriculum stage to record "
             "a reward trajectory for, e.g. tmrl_train_harder",
    )

    parser.add_argument(
        "--trainer-log",
        type=str,
        default=None,
        help="(curriculum-watch) path to the curriculum-trainer's stdout log file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="(record mode) stop after this many recorded episodes",
    )

    parser.add_argument(
        "--human-warmstart",
        action="store_true",
        help="(trainer/worker) use the human-demonstration warm-start run and checkpoint",
    )

    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=20,
        help="(pretrain-human) behavior-cloning epochs",
    )

    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=256,
        help="(pretrain-human) behavior-cloning batch size",
    )

    parser.add_argument(
        "--bc-lr",
        type=float,
        default=1e-4,
        help="(pretrain-human) behavior-cloning learning rate",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="(pretrain-human) overwrite an existing human warm-start actor",
    )

    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="(record mode) stop after this many minutes",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="(bc-eval/random-eval) maximum deterministic steps per episode",
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_server()

    elif args.mode == "trainer":
        run_trainer(human_warmstart=args.human_warmstart)

    elif args.mode == "worker":
        run_worker(human_warmstart=args.human_warmstart)

    elif args.mode == "curriculum-trainer":
        run_trainer(curriculum=True)

    elif args.mode == "curriculum-worker":
        run_worker(curriculum=True)

    elif args.mode == "curriculum-status":
        from curriculum.manager import CurriculumManager
        from curriculum import reward_registry
        print(CurriculumManager().status_string())
        print()
        reward_registry.status()

    elif args.mode == "curriculum-watch":
        if not args.trainer_log:
            raise SystemExit(
                "curriculum-watch requires --trainer-log <path to the "
                "curriculum-trainer process's stdout log file>"
            )
        import sys
        from curriculum.watch_and_promote import main as watch_main
        sys.argv = [sys.argv[0], args.trainer_log]
        watch_main()

    elif args.mode == "record-track-reward":
        if not args.stage_name:
            raise SystemExit(
                "record-track-reward requires a stage name, e.g.:\n"
                "  python train_td3.py record-track-reward tmrl_train_harder"
            )
        from curriculum.reward_registry import record_stage_reward
        record_stage_reward(args.stage_name, force=args.force)

    elif args.mode == "record":
        run_record(episodes=args.episodes, minutes=args.minutes)

    elif args.mode == "inspect-human-data":
        run_inspect_human_data()

    elif args.mode == "pretrain-human":
        run_pretrain_human(
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_lr,
            force=args.force,
        )

    elif args.mode == "diagnose":
        run_diagnose()

    elif args.mode == "verify-bc":
        from td3.live_diagnostics import verify_artifacts
        verify_artifacts()

    elif args.mode in {"bc-eval", "random-eval", "trained-eval"}:
        from td3.live_diagnostics import run_deterministic_policy
        policy = {
            "bc-eval": "bc",
            "random-eval": "random",
            "trained-eval": "trained",
        }[args.mode]
        run_deterministic_policy(
            policy=policy,
            episodes=args.episodes or 1,
            max_steps=args.max_steps,
            trace_steps=args.max_steps * (args.episodes or 1),
        )


if __name__ == "__main__":
    main()
