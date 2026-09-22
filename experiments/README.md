# experiments/

One subfolder per named experiment run, each holding a reproducibility manifest
(seed, algorithm, hyperparameters, git commit hash, TMRL/Python/PyTorch/CUDA
versions, track list — Section 26) plus pointers to that run's artifacts in
`C:\Users\sreci\TmrlData\weights\`, `checkpoints\`, and `experiments\<run_name>\`.

The actual weights/checkpoints are never duplicated here — only metadata about
where to find them (see Section 4 of the research plan). Empty until the first
tracked experiment is run.
