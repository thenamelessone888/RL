# results/

- `metrics/` — per-run CSV/JSON training and evaluation metrics (Section 22).
- `plots/` — matplotlib figures generated from `metrics/` (Section 23).
- `evaluations/` — independent (exploration-disabled) evaluation results, including
  seen-vs-unseen-track generalization runs (Sections 18-19).

Empty until the first real training run produces something to log; `.gitignore`
excludes the generated CSV/JSON/PNG files themselves so this directory structure is
versioned without committing large or frequently-changing run output.
