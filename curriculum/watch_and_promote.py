"""
Tails a curriculum trainer's stdout (same log format as
scripts/log_to_metrics_csv.py -- TMRL's own per-round
logging.info(stats[-1].to_string())) and feeds each round's
return_train/episode_length_train into the CurriculumManager.

When the current stage's promotion criteria are met (Section 10), this
prints a STAGE COMPLETE banner with the exact next steps and exits. It does
NOT automatically continue training the next stage: there is no way to
programmatically load a different TrackMania map (verified: no such API in
the installed tmrl==0.7.1 / its OpenPlanet plugins), so a human must load the
next stage's map before curriculum training can continue.

Usage (run alongside `python train_td3.py curriculum-trainer`):
    python curriculum/watch_and_promote.py <trainer_log_path>
"""

import re
import sys
import time

from curriculum.manager import CurriculumManager
from curriculum import reward_registry

EPOCH_ROUND_RE = re.compile(r"=== epoch (\d+)/\d+ === round (\d+)/\d+ =")
LOG_PREFIX_RE = re.compile(r"^[A-Z]+:\S*?:")
KV_RE = re.compile(r"^(\S+)\s+(\S+)$")

IDLE_FLUSH_SECONDS = 5.0
POLL_INTERVAL = 0.5


def parse_numeric(raw):
    if raw in ("NaN", "nan"):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return None


def follow(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                yield None
                time.sleep(POLL_INTERVAL)
                continue
            yield line.rstrip("\n")


def print_stage_complete_banner(manager, completed_stage):
    print()
    print("=" * 70)
    print("CURRICULUM STAGE COMPLETE")
    print("=" * 70)
    print(f"Finished: {completed_stage.name} ({completed_stage.map_file})")

    if manager.stage_idx > 0 and manager.promotion_log:
        last = manager.promotion_log[-1]
        print(f"  env_steps trained: {last['env_steps']}")
        if last["final_avg_return"] is not None:
            print(f"  final moving-avg return: {last['final_avg_return']:.2f}")

    next_stage = manager.current_stage
    print()
    print(f"NEXT STAGE: {next_stage.name}")
    print(f"  1. Stop the current trainer/worker (Ctrl+C both).")
    print(f"  2. In TrackMania, load: {next_stage.map_file}")
    if not next_stage.ready:
        print(f"  3. Record its reward trajectory (one-time, only if not done "
              f"already):")
        print(f"       python train_td3.py record-track-reward {next_stage.name}")
        print(f"  4. Activate it and resume training:")
    else:
        print(f"  3. Resume training (the SAME policy/checkpoint continues, "
              f"only the active reward file changes):")
    print(f"       python train_td3.py curriculum-trainer")
    print(f"       python train_td3.py curriculum-worker")
    print(f"     (curriculum-trainer/worker automatically activate "
          f"{next_stage.name}'s reward file on startup.)")
    print("=" * 70)


def main():
    # When stdout is redirected to a file (e.g. a backgrounded process
    # watched via `Read`/`tail`), Python defaults to block-buffering instead
    # of line-buffering, so status prints and the STAGE COMPLETE banner
    # would sit invisible in the buffer until the process exits normally.
    # Since this script is meant to be watched live, force line buffering.
    sys.stdout.reconfigure(line_buffering=True)

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <trainer_log_path>")
        sys.exit(1)

    log_path = sys.argv[1]
    manager = CurriculumManager()

    print(f"Watching: {log_path}")
    print(manager.status_string())
    print()

    current_kv = {}
    pending_flush = False
    last_line_time = time.time()

    def flush_round():
        nonlocal pending_flush
        if not current_kv:
            return
        episode_length = current_kv.get("episode_length_train")
        return_train = current_kv.get("return_train")
        manager.record_round(episode_length, return_train)
        pending_flush = False

        print(f"[curriculum] stage={manager.current_stage.name} "
              f"steps={manager.env_steps_this_stage}/{manager.current_stage.min_env_steps} "
              f"return={return_train} window={len(manager.return_window)}/"
              f"{manager.current_stage.promotion_window}")

        if manager.should_promote():
            if manager.is_last_stage:
                print()
                print("=" * 70)
                print(f"CURRICULUM COMPLETE: final stage "
                      f"({manager.current_stage.name}) has met its "
                      f"promotion criteria.")
                print("The agent has now trained through every configured "
                      "stage. Consider an unseen-track generalization eval "
                      "next (Section 19).")
                print("=" * 70)
                sys.exit(0)
            completed_stage = manager.current_stage
            next_stage = manager.promote()
            print_stage_complete_banner(manager, completed_stage)
            sys.exit(0)

    for line in follow(log_path):
        if line is None:
            if pending_flush and (time.time() - last_line_time) >= IDLE_FLUSH_SECONDS:
                flush_round()
            continue

        last_line_time = time.time()

        header_match = EPOCH_ROUND_RE.search(line)
        if header_match:
            flush_round()
            current_kv = {}
            pending_flush = False
            continue

        stripped = LOG_PREFIX_RE.sub("", line).strip()
        kv_match = KV_RE.match(stripped)
        if kv_match:
            key, raw_value = kv_match.groups()
            value = parse_numeric(raw_value)
            if value is not None:
                current_kv[key] = value
                pending_flush = True


if __name__ == "__main__":
    main()
