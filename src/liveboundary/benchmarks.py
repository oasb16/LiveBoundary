# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded synthetic experiments. All entities and labels are fictional."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from .constraints import CategoryIn, Not
from .contracts import AdaptationConfig, Constraint, VerifiedExample
from .learner import BusinessLearner, EvaluationSuite
from .models import TinyModelAdapter, TinyModelConfig

CATEGORIES = [
    "collect",
    "refund",
    "complete",
    "escalate",
    "no_action",
    "rent_1200",
    "rent_1350",
    "rent_1400",
    "contractor_aria",
    "contractor_ben",
]


def ex(example_id: str, inputs: Dict[str, str], target: str) -> VerifiedExample:
    return VerifiedExample("synthetic-co", example_id, inputs, target, "synthetic benchmark")


def config() -> TinyModelConfig:
    return TinyModelConfig(
        fields=["task", "entity", "status", "approved", "time", "wording"],
        vocabularies={
            "task": ["action", "rent", "contractor", "stable"],
            "entity": ["acct-a", "acct-b", "home-a", "home-b", "math-a", "math-b"],
            "status": ["settled", "open", "unresolved", "resolved", "na"],
            "approved": ["yes", "no", "na"],
            "time": ["past", "current", "na"],
            "wording": ["train", "heldout"],
        },
        categories=CATEGORIES,
        embedding_dim=10,
        hidden_dim=28,
    )


def rules() -> List[Constraint]:
    return [
        Constraint(
            "settled-no-collection",
            "1",
            "A settled account must not receive collection action",
            Not(CategoryIn(["collect"])).to_dict(),
            {"op": "field", "field": "status", "cmp": "eq", "value": "settled"},
        ),
        Constraint(
            "refund-needs-approval",
            "1",
            "A refund requires explicit approval",
            Not(CategoryIn(["refund"])).to_dict(),
            {"op": "field", "field": "approved", "cmp": "eq", "value": "no"},
        ),
        Constraint(
            "unresolved-not-complete",
            "1",
            "An unresolved issue must not be marked complete",
            Not(CategoryIn(["complete"])).to_dict(),
            {"op": "field", "field": "status", "cmp": "eq", "value": "unresolved"},
        ),
    ]


def dataset() -> Dict[str, Sequence[VerifiedExample]]:
    stable = [
        ex(
            "stable-a",
            {
                "task": "stable",
                "entity": "math-a",
                "status": "na",
                "approved": "na",
                "time": "na",
                "wording": "heldout",
            },
            "complete",
        ),
        ex(
            "stable-b",
            {
                "task": "stable",
                "entity": "math-b",
                "status": "na",
                "approved": "na",
                "time": "na",
                "wording": "heldout",
            },
            "no_action",
        ),
    ]
    prior = [
        ex(
            "prior-rent",
            {
                "task": "rent",
                "entity": "home-a",
                "status": "na",
                "approved": "na",
                "time": "past",
                "wording": "heldout",
            },
            "rent_1200",
        ),
        ex(
            "prior-contractor",
            {
                "task": "contractor",
                "entity": "home-b",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "heldout",
            },
            "contractor_ben",
        ),
    ]
    new_train = [
        ex(
            "train-current-rent",
            {
                "task": "rent",
                "entity": "home-a",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "train",
            },
            "rent_1400",
        ),
        ex(
            "train-contractor",
            {
                "task": "contractor",
                "entity": "home-a",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "train",
            },
            "contractor_aria",
        ),
        ex(
            "train-settled",
            {
                "task": "action",
                "entity": "acct-a",
                "status": "settled",
                "approved": "no",
                "time": "na",
                "wording": "train",
            },
            "no_action",
        ),
        ex(
            "train-open",
            {
                "task": "action",
                "entity": "acct-b",
                "status": "open",
                "approved": "no",
                "time": "na",
                "wording": "train",
            },
            "collect",
        ),
        ex(
            "train-approved-refund",
            {
                "task": "action",
                "entity": "acct-a",
                "status": "open",
                "approved": "yes",
                "time": "na",
                "wording": "train",
            },
            "refund",
        ),
        ex(
            "train-unresolved",
            {
                "task": "action",
                "entity": "acct-b",
                "status": "unresolved",
                "approved": "no",
                "time": "na",
                "wording": "train",
            },
            "escalate",
        ),
    ]
    new_validation = [
        ex(
            "held-current-rent",
            {
                "task": "rent",
                "entity": "home-a",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "heldout",
            },
            "rent_1400",
        ),
        ex(
            "held-contractor",
            {
                "task": "contractor",
                "entity": "home-a",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "heldout",
            },
            "contractor_aria",
        ),
        ex(
            "held-settled",
            {
                "task": "action",
                "entity": "acct-a",
                "status": "settled",
                "approved": "no",
                "time": "na",
                "wording": "heldout",
            },
            "no_action",
        ),
        ex(
            "held-open",
            {
                "task": "action",
                "entity": "acct-b",
                "status": "open",
                "approved": "no",
                "time": "na",
                "wording": "heldout",
            },
            "collect",
        ),
        ex(
            "held-approved-refund",
            {
                "task": "action",
                "entity": "acct-a",
                "status": "open",
                "approved": "yes",
                "time": "na",
                "wording": "heldout",
            },
            "refund",
        ),
        ex(
            "held-unresolved",
            {
                "task": "action",
                "entity": "acct-b",
                "status": "unresolved",
                "approved": "no",
                "time": "na",
                "wording": "heldout",
            },
            "escalate",
        ),
    ]
    final_test = [
        ex(
            "final-home-b",
            {
                "task": "rent",
                "entity": "home-b",
                "status": "na",
                "approved": "na",
                "time": "current",
                "wording": "heldout",
            },
            "rent_1350",
        )
    ]
    return {
        "stable": stable,
        "prior": prior,
        "train": new_train,
        "validation": new_validation,
        "final": final_test,
    }


def pretrain(seed: int, data: Dict[str, Sequence[VerifiedExample]]) -> TinyModelAdapter:
    model = TinyModelAdapter.seeded(config(), seed)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.04)
    examples = list(data["stable"]) + list(data["prior"])
    for _ in range(100):
        optimizer.zero_grad()
        loss = model.task_loss(examples)
        loss.backward()
        optimizer.step()
    return model


def run_seed(seed: int, root: Path) -> List[Dict[str, Any]]:
    data = dataset()
    starting = pretrain(seed, data)
    suite = EvaluationSuite(
        new_examples=data["validation"],
        prior_examples=data["prior"],
        correction_examples=[data["validation"][0]],
        historical_examples=[data["prior"][0]],
        stable_examples=data["stable"],
        useful_examples=[data["validation"][3], data["validation"][4], data["validation"][5]],
        final_test_examples=data["final"],
    )
    results = []
    frozen = BusinessLearner(
        "synthetic-co",
        starting,
        starting,
        rules(),
        AdaptationConfig(seed=seed),
        root / f"{seed}-frozen",
    )
    frozen_eval = frozen.evaluate(frozen.model, suite)
    results.append(
        {
            "seed": seed,
            "strategy": "frozen_baseline",
            "accepted": None,
            "metrics_scope": "frozen_live_model",
            **frozen_eval.to_dict(),
            "gradient_steps": 0,
            "update_latency_seconds": 0.0,
            "checkpoint_size_bytes": 0,
        }
    )
    settings = [
        ("online_task_only", [], 0.0, 0.0, 0),
        ("boundary_guided", rules(), 2.0, 0.0, 0),
        ("boundary_preservation_replay", rules(), 2.0, 0.7, 4),
    ]
    for name, constraints, boundary_weight, preservation_weight, replay_capacity in settings:
        learner = BusinessLearner(
            "synthetic-co",
            starting,
            starting,
            constraints,
            AdaptationConfig(
                learning_rate=0.035,
                gradient_steps=80,
                boundary_weight=boundary_weight,
                preservation_weight=preservation_weight,
                replay_capacity=replay_capacity,
                replay_sample_size=min(replay_capacity, len(data["prior"])),
                seed=seed,
                min_new_accuracy=0.3,
                min_prior_accuracy=0.0,
                max_boundary_violation_rate=1.0,
                max_stable_regression=0.0,
            ),
            root / f"{seed}-{name}",
        )
        if replay_capacity:
            learner.replay = list(data["prior"])
        report = learner.adapt(data["train"], suite)
        results.append(
            {
                "seed": seed,
                "strategy": name,
                "accepted": report.accepted,
                "metrics_scope": "candidate_before_acceptance_decision",
                "rejection_reasons": [] if report.accepted else report.reasons,
                **report.evaluation.to_dict(),
                "gradient_steps": report.gradient_steps,
                "update_latency_seconds": report.update_latency_seconds,
                "checkpoint_size_bytes": report.checkpoint_size_bytes,
            }
        )
    return results


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "new_fact_accuracy",
        "prior_fact_retention",
        "correction_accuracy",
        "historical_query_accuracy",
        "boundary_violation_rate",
        "useful_action_accuracy",
        "useful_action_coverage",
        "stable_task_accuracy",
        "update_latency_seconds",
        "checkpoint_size_bytes",
    ]
    result = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        selected = [row for row in rows if row["strategy"] == strategy]
        result[strategy] = {
            metric: {
                "mean": statistics.fmean(float(row[metric]) for row in selected),
                "values": [row[metric] for row in selected],
            }
            for metric in metrics
        }
        result[strategy]["accepted_updates"] = sum(row["accepted"] is True for row in selected)
        result[strategy]["rejected_updates"] = sum(row["accepted"] is False for row in selected)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="3,17,41")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for seed in seeds for row in run_seed(seed, args.work_dir)]
    else:
        with tempfile.TemporaryDirectory(prefix="liveboundary-benchmark-") as temp:
            rows = [row for seed in seeds for row in run_seed(seed, Path(temp))]
    payload = {
        "benchmark": "synthetic_boundary_and_parametric_memory",
        "synthetic_data": True,
        "seeds": seeds,
        "budgets": {"pretraining_steps": 100, "adaptation_steps": 80},
        "individual_runs": rows,
        "aggregate": aggregate(rows),
        "caveat": "Mechanism experiment only; not evidence of production or frontier-model reliability.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
