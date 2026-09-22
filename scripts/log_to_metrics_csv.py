"""
Tails a train_td3.py trainer's stdout log and converts each round's
`logging.info(stats[-1].to_string())` block (a pandas Series printed as
"key<whitespace>value" lines, per tmrl.training_offline.TrainingOffline) into
one row of a CSV under results/metrics/, per Section 22 of the research plan.

Decoupled from TMRL/project internals on purpose: it only parses the trainer's
own stdout, so it works unmodified regardless of which TD3 config produced the
log. Run alongside a trainer process:

    python scripts/log_to_metrics_csv.py <trainer_log_path> <run_name>

Writes to results/metrics/<run_name>_metrics.csv (created if missing, appended
to otherwise). A row is flushed either when the next round's header appears,
or after IDLE_FLUSH_SECONDS of no new log lines (so the last round of a run
that gets stopped mid-idle is not silently dropped).
"""

import csv
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

IDLE_FLUSH_SECONDS = 5.0
POLL_INTERVAL = 0.5

EPOCH_ROUND_RE = re.compile(r"=== epoch (\d+)/\d+ === round (\d+)/\d+ =")

# Python's logging prefixes only the FIRST line of a multi-line message
# (e.g. "INFO:root:memory_len   241"); continuation lines have no prefix.
LOG_PREFIX_RE = re.compile(r"^[A-Z]+:\S*?:")

KV_RE = re.compile(r"^(\S+)\s+(\S+)$")


def parse_numeric(raw):
    """Returns a float (NaN-aware) or None if `raw` is not a real metric value."""
    if raw in ("NaN", "nan"):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return None


def follow(path):
    """Yield new lines appended to `path` (like `tail -f`), or None on idle."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # start at end of file: only new rounds, not old ones
        while True:
            line = f.readline()
            if not line:
                yield None
                time.sleep(POLL_INTERVAL)
                continue
            yield line.rstrip("\n")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <trainer_log_path> <run_name>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    run_name = sys.argv[2]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / f"{run_name}_metrics.csv"

    fieldnames = []
    writer = None
    csv_file = None

    current_epoch = None
    current_round = None
    current_kv = {}
    pending_flush = False
    last_line_time = time.time()

    def flush_row():
        nonlocal fieldnames, writer, csv_file, pending_flush
        if current_epoch is None or not current_kv:
            return
        row = {"epoch": current_epoch, "round": current_round}
        row.update(current_kv)
        row["wall_time"] = time.time()

        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

        if writer is None:
            new_file = not csv_path.exists()
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
        else:
            writer.fieldnames = fieldnames

        writer.writerow({k: row.get(k, "") for k in fieldnames})
        csv_file.flush()
        pending_flush = False
        print(f"[metrics] epoch={current_epoch} round={current_round} -> {csv_path}")

    print(f"Watching: {log_path}")
    print(f"Writing:  {csv_path}")

    for line in follow(log_path):
        if line is None:
            if pending_flush and (time.time() - last_line_time) >= IDLE_FLUSH_SECONDS:
                flush_row()
            continue

        last_line_time = time.time()

        header_match = EPOCH_ROUND_RE.search(line)
        if header_match:
            flush_row()
            current_epoch = int(header_match.group(1))
            current_round = int(header_match.group(2))
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
