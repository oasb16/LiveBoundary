# Architecture

## State flow

1. The caller creates an immutable baseline and a business-scoped learner.
2. `adapt` rejects cross-business examples, independently clones the last accepted live model, and
   selects either every native parameter or explicit matching native parameters/modules.
3. Adam minimizes weighted verified-task, boundary, and optional preservation losses. A bounded,
   uniformly sampled replay buffer can add earlier verified examples. Replay is stored in process
   and is never consulted by `predict` or strict inference.
4. Validation measures new knowledge, retention, corrections, temporal history, boundaries, useful
   action, stable behavior, and finiteness. Final-test examples are explicitly excluded and checked
   for split overlap.
5. A candidate passing every configured threshold is written to a temporary checkpoint, renamed,
   and then recorded as accepted in an atomically replaced manifest. A rejection leaves the accepted
   model untouched. Optimizer state is not persisted in this MVP, so there is no hidden optimizer
   mutation to restore.
6. Rollback moves the manifest pointer only to a known accepted version and reloads it. Startup uses
   that pointer for recovery.

Baseline weights and buffers are copied and compared after training. Candidate and accepted tensors
are also compared for storage isolation. Business directory names separate version histories; this
is state isolation, not an access-control claim about neural parameters.

## Boundary semantics

The supported language contains categorical membership, negation, conjunction, disjunction and
implication. Rules may be conditioned on equality, inequality, membership and non-membership tests
over trusted structured input fields. It is not arbitrary Python and is not a natural-language
policy compiler.

For a categorical random variable `Y` over explicitly listed values and active rules whose jointly
valid set is `V`, the exact boundary loss is:

`-log P(Y in V) = logsumexp(all logits) - logsumexp(logits in V)`

Every output is enumerated, up to `max_enumerated_outputs` (256 by default). If `V` is empty, training
raises `UnsatisfiableConstraints` with active rule identifiers. Log-sum-exp avoids avoidable numerical
underflow; non-finite loss, gradient, or parameters fail the candidate.

## Parametric-memory boundary

`strict_predict` starts a new Python process and supplies exactly two JSON keys: the accepted model
checkpoint and one structured question. The child rejects extra keys and loads no event ledger,
replay buffer, conversation, answer, retrieval index, entity lookup, or training-time cache. The
outer evaluator retains ground truth.

The tiny model's question explicitly encodes named structured fields such as entity, query type and
temporal scope. The expected category is absent and must be represented by learned parameters.
Held-out wording is a distinct input value, so results test modest compositional generalization, not
free-form language understanding.

## Checkpoint contents

Accepted records include parent version, configuration and constraint hashes, architecture identity,
baseline SHA-256, seed, selected parameters, checkpoint path, and adaptation metadata. The checkpoint
contains architecture configuration, parameter and buffer state, and version metadata. Tokenizer
identity belongs in a future generalized checkpoint implementation; the optional LM adapter is an
integration preview and does not yet participate in transactional persistence.
