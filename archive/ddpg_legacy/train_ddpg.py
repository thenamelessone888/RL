
"""
DDPG + TMRL 0.7.1 launcher.

Run three separate PowerShell windows:

    python train_ddpg.py server

    python train_ddpg.py trainer

    python train_ddpg.py worker

The existing SAC configuration/model/checkpoint files are never used
as DDPG output paths.
"""

import argparse
import time

from tmrl.networking import Server, Trainer, RolloutWorker

import tmrl.config.config_constants as cfg

from ddpg.config import (
    DDPG_TRAINER,
    DDPG_ENV_CLS,
    DDPG_SAMPLE_COMPRESSOR,
    DDPG_OBS_PREPROCESSOR,

    DDPG_MODEL_PATH_WORKER,
    DDPG_MODEL_PATH_SAVE_HISTORY,
    DDPG_MODEL_PATH_TRAINER,
    DDPG_CHECKPOINT_PATH,
    DDPG_HUMAN_WARMSTART_TRAINER,
    DDPG_HUMAN_WARMSTART_CHECKPOINT_PATH,
    DDPG_HUMAN_WARMSTART_MODEL_PATH_WORKER,
    DDPG_HUMAN_WARMSTART_MODEL_PATH_TRAINER,
    DDPG_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY,
)

from ddpg.models import DDPGMLPActor
from ddpg.noise import OrnsteinUhlenbeckNoise


# ============================================================================
# SERVER
# ============================================================================

def run_server():
    print("=" * 60)
    print("DDPG SERVER")
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
    print("[READY] DDPG Server is running.")
    print("[INFO] Start the trainer in another PowerShell window.")
    print("[INFO] Then start the worker.")
    print("[INFO] Press CTRL+C to stop.")
    print()

    try:
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
        print("[INFO] DDPG Server stopped.")


# ============================================================================
# TRAINER
# ============================================================================

def run_trainer(human_warmstart=False):
    print("=" * 60)
    print("DDPG TRAINER")
    print("=" * 60)

    trainer_model = DDPG_HUMAN_WARMSTART_MODEL_PATH_TRAINER if human_warmstart else DDPG_MODEL_PATH_TRAINER
    trainer_checkpoint = DDPG_HUMAN_WARMSTART_CHECKPOINT_PATH if human_warmstart else DDPG_CHECKPOINT_PATH
    print("Trainer model:")
    print(f"  {trainer_model}")

    print("Checkpoint:")
    print(f"  {trainer_checkpoint}")

    print(f"Dataset:")
    print(f"  {cfg.TMRL_FOLDER / 'dataset_ddpg'}")

    print()

    trainer = Trainer(
        training_cls=(DDPG_HUMAN_WARMSTART_TRAINER if human_warmstart else DDPG_TRAINER),

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
        # Explicit DDPG paths.
        #
        # Do NOT omit these because TMRL's defaults point at the
        # currently configured SAC paths.

        model_path=(DDPG_HUMAN_WARMSTART_MODEL_PATH_TRAINER if human_warmstart else DDPG_MODEL_PATH_TRAINER),
        checkpoint_path=(DDPG_HUMAN_WARMSTART_CHECKPOINT_PATH if human_warmstart else DDPG_CHECKPOINT_PATH),
    )

    print("[READY] DDPG Trainer connecting to Server...")
    print()

    trainer.run()


# ============================================================================
# WORKER
# ============================================================================

def run_pretrain_human(epochs, batch_size, learning_rate, force=False):
    from ddpg.demonstration import pretrain_human
    pretrain_human(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        force=force,
    )


def run_worker(human_warmstart=False):
    print("=" * 60)
    print("DDPG ROLLOUT WORKER")
    print("=" * 60)

    worker_model = DDPG_HUMAN_WARMSTART_MODEL_PATH_WORKER if human_warmstart else DDPG_MODEL_PATH_WORKER
    worker_history = DDPG_HUMAN_WARMSTART_MODEL_PATH_SAVE_HISTORY if human_warmstart else DDPG_MODEL_PATH_SAVE_HISTORY
    print("Worker model:")
    print(f"  {worker_model}")

    print("Model history:")
    print(f"  {worker_history}")

    print()

    worker = RolloutWorker(
        env_cls=DDPG_ENV_CLS,

        # Our actor, not TMRL SAC.
        actor_module_cls=DDPGMLPActor,

        # Existing TMRL LIDAR compressor.
        sample_compressor=DDPG_SAMPLE_COMPRESSOR,

        device=(
            "cuda"
            if cfg.CUDA_INFERENCE
            else "cpu"
        ),

        # CRITICAL:
        # Explicit DDPG model path.
        model_path=(DDPG_HUMAN_WARMSTART_MODEL_PATH_WORKER if human_warmstart else DDPG_MODEL_PATH_WORKER),

        # Existing LIDAR observation preprocessing.
        obs_preprocessor=DDPG_OBS_PREPROCESSOR,

        crc_debug=cfg.CRC_DEBUG,

        # DDPG-specific model history.
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
    # DDPG EXPLORATION
    # ========================================================================

    # Noise is attached to OUR actor.
    #
    # RolloutWorker:
    #
    #   training episode -> test=False
    #                    -> actor.act(..., test=False)
    #                    -> OU noise enabled
    #
    #   evaluation       -> test=True
    #                    -> actor.act(..., test=True)
    #                    -> deterministic action

    worker.actor.noise = OrnsteinUhlenbeckNoise(
        action_dim=3,
        theta=0.15,
        sigma=0.20,
        dt=1e-2,
    )

    print("[READY] DDPG Worker created.")
    print("[INFO] OU exploration enabled.")
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
# MAIN
# ============================================================================

# ============================================================================
# HUMAN DEMONSTRATION RECORDING
# ============================================================================
# Separate path from the DDPG server/trainer/worker pipeline above. Does not load the
# pretrained SAC model or the DDPG actor. Does not feed DDPG_MEMORY / dataset_ddpg.
# See ddpg/human_interface.py, ddpg/human_recorder.py, ddpg/human_dataset.py.

def run_record(episodes, minutes):
    from ddpg.human_recorder import run_recorder
    run_recorder(episodes=episodes, minutes=minutes)


def run_inspect_human_data():
    from ddpg.human_dataset import inspect_human_data
    inspect_human_data()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TMRL 0.7.1 custom DDPG launcher"
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
        ],
        help="Process to start",
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

    args = parser.parse_args()

    if args.mode == "server":
        run_server()

    elif args.mode == "trainer":
        run_trainer(human_warmstart=args.human_warmstart)

    elif args.mode == "worker":
        run_worker(human_warmstart=args.human_warmstart)

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


if __name__ == "__main__":
    main()

