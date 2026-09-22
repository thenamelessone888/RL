# configs/

Reserved for externalized experiment configuration (JSON), per Section 8 of the
research plan: TD3's hyperparameters (`actor_lr`, `critic_lr`, `gamma`, `tau`,
`policy_noise`, `noise_clip`, `policy_delay`, `batch_size`, `replay_capacity`,
`learning_starts`, `exploration_noise`, `gradient_steps`, ...) currently live as
hard-coded `partial(...)` keyword arguments in [`td3/config.py`](../td3/config.py).

Externalizing them into `baseline.json` / `curriculum.json` / `evaluation.json` /
`debug.json` here is deferred to Phase 3/4 (algorithm finalization + unit tests) of
the research plan, rather than done as part of the Phase 1 cleanup pass, so that the
already-tested `td3/` implementation isn't changed without accompanying test
verification.
