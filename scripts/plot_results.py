"""
Plots training curves from a metrics CSV produced by scripts/log_to_metrics_csv.py
(Section 23 of the research plan).

Usage:
    python scripts/plot_results.py <run_name>

Reads results/metrics/<run_name>_metrics.csv, writes
results/plots/<run_name>_training_curves.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed to save a PNG
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <run_name>")
        sys.exit(1)

    run_name = sys.argv[1]
    csv_path = METRICS_DIR / f"{run_name}_metrics.csv"

    if not csv_path.exists():
        print(f"No metrics file found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["step_idx"] = range(len(df))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"{run_name}_training_curves.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"TD3 training curves: {run_name}")

    ax = axes[0, 0]
    if "return_train" in df.columns:
        ax.plot(df["step_idx"], df["return_train"], marker="o", ms=3)
    ax.set_title("Episode return (train)")
    ax.set_xlabel("round")
    ax.set_ylabel("return")

    ax = axes[0, 1]
    if "episode_length_train" in df.columns:
        ax.plot(df["step_idx"], df["episode_length_train"], marker="o", ms=3, color="tab:orange")
    ax.set_title("Episode length (train)")
    ax.set_xlabel("round")
    ax.set_ylabel("steps")

    ax = axes[1, 0]
    if "loss_critic" in df.columns:
        ax.plot(df["step_idx"], df["loss_critic"], label="loss_critic", color="tab:green")
    if "loss_actor" in df.columns:
        ax.plot(df["step_idx"], df["loss_actor"], label="loss_actor", color="tab:red")
    ax.set_title("Losses")
    ax.set_xlabel("round")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[1, 1]
    if "bc_reg_term" in df.columns:
        ax.plot(df["step_idx"], df["bc_reg_term"], marker="o", ms=3, color="tab:purple")
    ax.set_title("BC-anchor regularization term")
    ax.set_xlabel("round")
    ax.set_ylabel("MSE(pi, bc_pi)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
