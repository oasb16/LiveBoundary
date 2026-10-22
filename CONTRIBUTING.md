# Contributing

Create a virtual environment, install `-e '.[dev]'`, and run `ruff check .`, `pytest`, and
`python -m build`. New behavior needs deterministic unit tests. Empirical tests must use fixed seeds,
bounded budgets, machine-readable raw results, and may not make a claimed accuracy threshold the sole
CI gate.

Do not add hosted-service requirements to the core. Do not silently accept unverified targets, read
replay during inference, tune against final-test labels, or describe synthetic results as production
reliability. Contributions are submitted under Apache-2.0.
