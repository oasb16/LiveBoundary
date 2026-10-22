# Limitations

- The offline result demonstrates native-weight change in a tiny structured classifier, not useful
  adaptation in a frontier language model.
- Finite categorical enumeration is exact only for the declared output space and capped size. The
  rule language cannot represent arbitrary business policy or prove a natural-language interpretation.
- Acceptance quality depends on representative, non-leaking probes and thresholds. A passing
  synthetic suite is not a production safety guarantee.
- Replay consumes storage and may retain sensitive training facts. This MVP keeps it in process and
  excludes it from strict inference, but provides no encryption, retention policy, or data deletion
  guarantee.
- Parameter updates do not enforce tenant authorization. Filesystem/process access control remains an
  application responsibility.
- Rollback restores an accepted model version, not effects already produced in external systems.
- “Correction” trains a newer scoped label; it does not prove removal of every representation of the
  old value. No guaranteed machine unlearning is claimed.
- The optional Hugging Face adapter is local-only and candidate-normalized, but is not yet connected
  to the transactional checkpoint loader or benchmarked here.
- Memory usage is omitted when no reliable cross-platform measurement dependency is present.
- The benchmark uses small synthetic categories and fixed budgets. Failures and rejected candidates
  are reported, but results may vary by PyTorch/platform version.

Open questions include scalable constraint scoring, adversarial and distribution-shift evaluation,
privacy-aware replay, safe optimizer-state continuity, temporal representation, edit locality, probe
coverage, and human-governed verification workflows.
