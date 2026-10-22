# Data contracts and update semantics

All public contracts are dataclasses with dictionary serialization.

- `BusinessEvent` carries business/event identifiers, observation time, optional effective time,
  source, verification state, event kind, payload, and correction/supersession references.
- `VerifiedExample` can only be constructed with `verification_status=verified`; it carries inputs,
  target, provenance, and optional event/effective-time links.
- `Constraint` stores versioned explicit rule/condition dictionaries and source.
- `AdaptationConfig` stores objective weights, replay behavior, native parameter selection, finite
  enumeration limit, acceptance thresholds, and seed.
- `EvaluationReport`, `AdaptationReport`, and `ModelVersion` record measured outcomes and lineage.

`EventKind` distinguishes a new fact, correction, effective-dated change, and behavioral boundary.
`EventLedger` requires the caller to choose an unverified-data policy: `reject` or `ignore`. It never
silently turns an unverified statement into a target. Temporal resolution selects the latest verified
event effective no later than the query time. Corrections must reference an existing event. Historical
and current training queries therefore need explicit time scope; missing temporal scope must not be
expanded into conflicting targets.

The baseline is a recovery/reference model, not an oracle for new facts. Preservation uses baseline
outputs only on caller-designated stable probes. New business truth comes from verified labels and
explicit rules.
