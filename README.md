# LiveBoundary

[![CI](https://github.com/oasb16/LiveBoundary/actions/workflows/ci.yml/badge.svg)](https://github.com/oasb16/LiveBoundary/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

LiveBoundary is a small research library for **constrained online adaptation**: it updates a
model's existing PyTorch parameters from verified examples while adding differentiable losses for
explicit business boundaries. Candidate updates are evaluated and either atomically promoted or
rejected. Each business has an independent accepted-version lineage.

This project demonstrates a mechanism on a tiny offline model. It does not establish frontier-model
effectiveness, database reliability, autonomous lifelong learning, generic policy compilation,
guaranteed deletion or unlearning, or a new learning algorithm. “Anti-training” is only informal
motivation.

## Why this exists

Most “memory” systems leave the model unchanged and recover facts from runtime context. LiveBoundary
tests a narrower, harder question: can verified facts and explicit operating boundaries be written
into native model parameters, survive a fresh-process restart with all training context removed, and
still pass retention and usefulness gates?

The first synthetic experiment publishes the failure as well as the success:

| Strategy | New facts | Prior retention | Stable behavior | Accepted updates |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.056 | 1.000 | 1.000 | N/A |
| Task only | 1.000 | 0.000 | 0.833 | 2/3 |
| Boundary guided | 1.000 | 0.000 | 0.833 | 2/3 |
| Boundary + preservation/replay | 0.944 | 1.000 | 1.000 | 3/3 |

Raw per-seed results and rejected candidates are checked in. See
[`RESULTS.md`](RESULTS.md) and [`benchmark-results.json`](benchmark-results.json).

## What you can inspect

- Existing PyTorch parameters change through real gradients; per-parameter change magnitudes are
  recorded.
- A closed constraint language supports allowed/forbidden categories, implication, conjunction,
  disjunction, negation, and trusted-field conditions.
- Candidate updates are independently cloned, evaluated, accepted or rejected, atomically
  checkpointed, and rollable to an earlier accepted version.
- Strict evaluation starts a fresh process that receives only a checkpoint and a structured question.
- The baseline remains an immutable recovery point and is never treated as truth for new facts.

## Quickstart

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/quickstart.py
pytest
```

The core uses no hosted service and needs no network after installation. The optional causal-LM
adapter loads both model and tokenizer with `local_files_only=True`:

```bash
python -m pip install -e '.[hf]'
```

`LocalCausalLMAdapter` scores declared answer candidates. Its softmax is a distribution over those
candidates, not the full language distribution. Model downloads are intentionally outside the core.

## API shape

```python
learner = BusinessLearner(
    business_id="acme",
    model=model,
    baseline=baseline,
    constraints=constraints,
    config=config,
    checkpoint_dir=path,
)
report = learner.adapt(verified_batch, validation_suite)
result = learner.predict(query)
learner.rollback(version_id)
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete run and
[`docs/architecture.md`](docs/architecture.md) for state boundaries and assumptions.

## Synthetic benchmarks

```bash
liveboundary-benchmark --seeds 3,17,41 --output benchmark-results.json
```

The JSON contains every seed, accepted and rejected update counts, bounded step budgets, latency,
checkpoint size, and aggregate metrics for a frozen baseline, task-only updates, boundary-guided
updates, and boundary plus preservation/replay. All names and data are synthetic. The final-test
split is checked for identifier overlap and is not used for acceptance.

## Scope and provenance

The implementation is original and does not copy Pylon or paper implementations. Pylon's repository
was inspected as Apache-2.0, but LiveBoundary has no code dependency on it and has no affiliation
with either project named Pylon. See [`docs/related-work.md`](docs/related-work.md),
[`docs/data-contracts.md`](docs/data-contracts.md), and [`LIMITATIONS.md`](LIMITATIONS.md).

Licensed under Apache-2.0.
