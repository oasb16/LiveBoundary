"""Transactional constrained online adaptation."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.nn import functional as F

from .checkpoints import CheckpointStore, canonical_hash
from .constraints import BoundaryLossEngine, assert_finite
from .contracts import (
    AdaptationConfig,
    AdaptationReport,
    Constraint,
    EvaluationReport,
    ModelVersion,
    VerifiedExample,
    utc_now,
)
from .models import TinyModelAdapter, state_deltas, states_equal


@dataclass
class EvaluationSuite:
    new_examples: Sequence[VerifiedExample] = field(default_factory=list)
    prior_examples: Sequence[VerifiedExample] = field(default_factory=list)
    correction_examples: Sequence[VerifiedExample] = field(default_factory=list)
    historical_examples: Sequence[VerifiedExample] = field(default_factory=list)
    stable_examples: Sequence[VerifiedExample] = field(default_factory=list)
    useful_examples: Sequence[VerifiedExample] = field(default_factory=list)
    final_test_examples: Sequence[VerifiedExample] = field(default_factory=list)

    def validation_examples(self) -> List[VerifiedExample]:
        groups = [
            self.new_examples,
            self.prior_examples,
            self.correction_examples,
            self.historical_examples,
            self.stable_examples,
            self.useful_examples,
        ]
        seen = set()
        result = []
        for example in (item for group in groups for item in group):
            if example.example_id not in seen:
                result.append(example)
                seen.add(example.example_id)
        final_ids = {example.example_id for example in self.final_test_examples}
        overlap = final_ids.intersection(seen)
        if overlap:
            raise ValueError(f"Validation and final-test splits overlap: {sorted(overlap)}")
        return result


def _accuracy(model: TinyModelAdapter, examples: Sequence[VerifiedExample]) -> float:
    if not examples:
        return 0.0
    predicted = model.predict([example.inputs for example in examples])
    return sum(p == example.target for p, example in zip(predicted, examples)) / len(examples)


class BusinessLearner:
    """Maintains one independently checkpointed live model for one business."""

    def __init__(
        self,
        business_id: str,
        model: TinyModelAdapter,
        baseline: TinyModelAdapter,
        constraints: Sequence[Constraint],
        config: AdaptationConfig,
        checkpoint_dir: Path,
        resume: bool = True,
    ) -> None:
        self.business_id = business_id
        self.config = config
        self.constraints = list(constraints)
        self.baseline = baseline.clone()
        self._baseline_state = self.baseline.state_snapshot()
        self.store = CheckpointStore(Path(checkpoint_dir), business_id)
        self.baseline_identity = self.store.initialize_baseline(self.baseline)
        existing = self.store.accepted_version() if resume else None
        self.model = (
            TinyModelAdapter.load(Path(existing.checkpoint_path)) if existing else model.clone()
        )
        self.current_version = existing
        self.boundary = BoundaryLossEngine(
            self.model.categories, self.constraints, config.max_enumerated_outputs
        )
        self.replay: List[VerifiedExample] = []
        self._rng = random.Random(config.seed)

    def _check_business(self, examples: Sequence[VerifiedExample]) -> None:
        mismatched = sorted({e.business_id for e in examples if e.business_id != self.business_id})
        if mismatched:
            raise ValueError(
                f"Learner for {self.business_id!r} cannot train on businesses {mismatched!r}"
            )

    def predict(self, query: Mapping[str, Any]) -> str:
        return self.model.predict([query])[0]

    def evaluate(self, model: TinyModelAdapter, suite: EvaluationSuite) -> EvaluationReport:
        examples = suite.validation_examples()
        if examples:
            with torch.no_grad():
                loss_tensor = model.task_loss(examples)
            loss = float(loss_tensor)
            predicted = model.predict([e.inputs for e in examples])
            violation = self.boundary.violation_rate(predicted, [e.inputs for e in examples])
        else:
            loss = 0.0
            violation = 0.0
        useful_predictions = (
            model.predict([e.inputs for e in suite.useful_examples])
            if suite.useful_examples
            else []
        )
        useful_coverage = (
            sum(value not in {"abstain", "none", "no_action"} for value in useful_predictions)
            / len(useful_predictions)
            if useful_predictions
            else 0.0
        )
        finite = math.isfinite(loss) and all(
            bool(torch.isfinite(parameter).all()) for parameter in model.network.parameters()
        )
        return EvaluationReport(
            loss=loss,
            accuracy=_accuracy(model, examples),
            new_fact_accuracy=_accuracy(model, suite.new_examples),
            prior_fact_retention=_accuracy(model, suite.prior_examples),
            correction_accuracy=_accuracy(model, suite.correction_examples),
            historical_query_accuracy=_accuracy(model, suite.historical_examples),
            boundary_violation_rate=violation,
            useful_action_accuracy=_accuracy(model, suite.useful_examples),
            useful_action_coverage=useful_coverage,
            stable_task_accuracy=_accuracy(model, suite.stable_examples),
            finite=finite,
            counts={
                "validation": len(examples),
                "final_test_excluded": len(suite.final_test_examples),
            },
        )

    def adapt(
        self, verified_batch: Sequence[VerifiedExample], validation_suite: EvaluationSuite
    ) -> AdaptationReport:
        start = time.perf_counter()
        batch = list(verified_batch)
        if not batch:
            raise ValueError("adapt requires at least one verified example")
        self._check_business(batch)
        validation_suite.validation_examples()
        torch.manual_seed(self.config.seed)
        before_accepted = self.model.state_snapshot()
        candidate = self.model.clone()
        candidate_before = candidate.state_snapshot()
        selected = candidate.select_parameters(
            self.config.trainable, self.config.trainable_patterns
        )
        optimizer = torch.optim.Adam(
            [parameter for _, parameter in selected], lr=self.config.learning_rate
        )
        replay_used: List[VerifiedExample] = []
        if self.config.replay_sample_size and self.replay:
            count = min(self.config.replay_sample_size, len(self.replay))
            replay_used = self._rng.sample(self.replay, count)
        train_examples = batch + replay_used
        stable = list(validation_suite.stable_examples)
        baseline_stable_logits = None
        if stable and self.config.preservation_weight:
            with torch.no_grad():
                baseline_stable_logits = self.baseline.logits([e.inputs for e in stable]).detach()
        for _ in range(self.config.gradient_steps):
            optimizer.zero_grad(set_to_none=True)
            task_loss = candidate.task_loss(train_examples)
            boundary_loss = self.boundary.loss(
                candidate.logits([e.inputs for e in train_examples]),
                [e.inputs for e in train_examples],
            )
            preservation_loss = task_loss * 0.0
            if baseline_stable_logits is not None:
                candidate_logits = candidate.logits([e.inputs for e in stable])
                preservation_loss = F.kl_div(
                    F.log_softmax(candidate_logits, dim=-1),
                    F.softmax(baseline_stable_logits, dim=-1),
                    reduction="batchmean",
                )
            loss = (
                self.config.task_weight * task_loss
                + self.config.boundary_weight * boundary_loss
                + self.config.preservation_weight * preservation_loss
            )
            assert_finite(loss, "training loss")
            loss.backward()
            for name, parameter in selected:
                if parameter.grad is None:
                    raise FloatingPointError(f"Selected parameter {name} received no gradient")
                assert_finite(parameter.grad, f"gradient for {name}")
            optimizer.step()
            for name, parameter in selected:
                assert_finite(parameter, f"parameter {name}")
        evaluation = self.evaluate(candidate, validation_suite)
        baseline_stable = _accuracy(self.baseline, validation_suite.stable_examples)
        stable_regression = max(0.0, baseline_stable - evaluation.stable_task_accuracy)
        reasons = []
        checks = [
            (evaluation.finite, "candidate contains non-finite loss or parameters"),
            (
                evaluation.new_fact_accuracy >= self.config.min_new_accuracy,
                "new-fact accuracy below threshold",
            ),
            (
                evaluation.prior_fact_retention >= self.config.min_prior_accuracy,
                "prior retention below threshold",
            ),
            (
                evaluation.useful_action_accuracy >= self.config.min_useful_accuracy,
                "useful-action accuracy below threshold",
            ),
            (
                evaluation.boundary_violation_rate <= self.config.max_boundary_violation_rate,
                "boundary violation rate above threshold",
            ),
            (
                stable_regression <= self.config.max_stable_regression,
                "stable-task regression above threshold",
            ),
        ]
        for passed, reason in checks:
            if not passed:
                reasons.append(reason)
        deltas = state_deltas(candidate_before, candidate.state_snapshot())
        if not deltas:
            reasons.append("no native parameter changed")
        if not states_equal(before_accepted, self.model.state_snapshot()):
            raise RuntimeError("Candidate training mutated accepted model through shared storage")
        if not states_equal(self._baseline_state, self.baseline.state_snapshot()):
            raise RuntimeError("Immutable baseline model changed during adaptation")
        candidate_id = f"v-{uuid.uuid4().hex[:12]}"
        version = None
        checkpoint_size = 0
        if not reasons:
            version = ModelVersion(
                version_id=candidate_id,
                business_id=self.business_id,
                parent_version_id=self.current_version.version_id if self.current_version else None,
                created_at=utc_now(),
                checkpoint_path="",
                config_hash=canonical_hash(self.config.to_dict()),
                constraint_hash=canonical_hash([c.to_dict() for c in self.constraints]),
                model_identity=f"tiny_structured:{canonical_hash(candidate.config.to_dict())}",
                seed=self.config.seed,
                accepted=False,
                metadata={
                    "optimizer": "Adam",
                    "optimizer_state_persisted": False,
                    "selected_parameters": [name for name, _ in selected],
                    "baseline_sha256": self.baseline_identity,
                },
            )
            version = self.store.promote(candidate, version)
            self.model = candidate
            self.current_version = version
            checkpoint_size = Path(version.checkpoint_path).stat().st_size
            if self.config.replay_capacity:
                self.replay = (self.replay + batch)[-self.config.replay_capacity :]
        else:
            reasons.insert(0, "candidate rejected; accepted model and optimizer state preserved")
        return AdaptationReport(
            business_id=self.business_id,
            candidate_version_id=candidate_id,
            accepted=not reasons,
            reasons=reasons or ["all configured acceptance thresholds passed"],
            evaluation=evaluation,
            changed_parameters=deltas,
            gradient_steps=self.config.gradient_steps,
            update_latency_seconds=time.perf_counter() - start,
            checkpoint_size_bytes=checkpoint_size,
            replay_examples_used=len(replay_used),
            replay_policy=self.config.replay_sampling,
            version=version,
        )

    def rollback(self, version_id: str) -> ModelVersion:
        version = self.store.rollback(version_id)
        self.model = TinyModelAdapter.load(Path(version.checkpoint_path))
        self.current_version = version
        return version

    def strict_predict(self, query: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.current_version:
            raise RuntimeError("Strict inference requires an accepted checkpoint")
        request = {
            "checkpoint": self.current_version.checkpoint_path,
            "question": dict(query),
        }
        completed = subprocess.run(
            [sys.executable, "-m", "liveboundary.strict_inference"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
            env={
                "PATH": str(Path(sys.executable).parent),
                "PYTHONPATH": str(Path(__file__).parents[1]),
            },
        )
        response = json.loads(completed.stdout)
        response["exact_subprocess_input"] = request
        return response
