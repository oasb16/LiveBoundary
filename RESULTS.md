# Bounded benchmark results

These results were measured locally on 2026-10-21 with Python 3.9.6, PyTorch 2.8.0 on macOS,
three fixed seeds (`3,17,41`), 100 baseline pretraining steps, and 80 adaptation steps. Raw per-seed
records are in [`benchmark-results.json`](benchmark-results.json). All data are synthetic.

| Strategy | Accepted | New facts | Prior retention | Historical | Violations | Useful accuracy | Stable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0/0 | 0.056 | 1.000 | 1.000 | 0.033 | 0.000 | 1.000 |
| Task only | 2/3 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.833 |
| Boundary guided | 2/3 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.833 |
| Boundary + preservation/replay | 3/3 | 0.944 | 1.000 | 1.000 | 0.000 | 0.889 | 1.000 |

The task-only and boundary-only candidates learned the new/corrected values but completely forgot the
two prior probes. At seed 17 each also regressed stable accuracy to 0.5, so the explicit zero-regression
gate rejected that candidate and preserved the accepted model. Preservation plus two replayed prior
examples retained the prior and stable probes for all seeds, but seed 41 missed one new/useful example.
This is the expected plasticity-retention tradeoff, not a universal ranking.

Mean candidate update latency was 0.118 seconds for task only, 0.121 seconds for boundary guided, and
0.156 seconds with preservation/replay. Accepted checkpoints were 15,173 bytes. Memory usage was not
reported because this minimal environment had no reliable cross-platform process-memory measurement
dependency. No result here establishes production reliability or effectiveness at language-model scale.
