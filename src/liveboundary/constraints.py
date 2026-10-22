# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite, explicit constraint language and exact satisfaction-mass loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from .contracts import Constraint


class UnsatisfiableConstraints(ValueError):
    """Raised when active rules leave no valid categorical output."""


class Expression:
    def evaluate(self, output: str) -> bool:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class CategoryIn(Expression):
    values: Sequence[str]

    def evaluate(self, output: str) -> bool:
        return output in self.values

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "category_in", "values": list(self.values)}


@dataclass(frozen=True)
class Not(Expression):
    child: Expression

    def evaluate(self, output: str) -> bool:
        return not self.child.evaluate(output)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "not", "child": self.child.to_dict()}


@dataclass(frozen=True)
class AllOf(Expression):
    children: Sequence[Expression]

    def evaluate(self, output: str) -> bool:
        return all(child.evaluate(output) for child in self.children)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "and", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True)
class AnyOf(Expression):
    children: Sequence[Expression]

    def evaluate(self, output: str) -> bool:
        return any(child.evaluate(output) for child in self.children)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": "or", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True)
class Implies(Expression):
    antecedent: Expression
    consequent: Expression

    def evaluate(self, output: str) -> bool:
        return (not self.antecedent.evaluate(output)) or self.consequent.evaluate(output)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": "implies",
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
        }


@dataclass(frozen=True)
class FieldCondition:
    field: str
    op: str
    value: Any

    def evaluate(self, inputs: Mapping[str, Any]) -> bool:
        if self.field not in inputs:
            return False
        actual = inputs[self.field]
        if self.op == "eq":
            return actual == self.value
        if self.op == "ne":
            return actual != self.value
        if self.op == "in":
            return actual in self.value
        if self.op == "not_in":
            return actual not in self.value
        raise ValueError(f"Unsupported field condition operation: {self.op}")

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}


def expression_from_dict(value: Mapping[str, Any]) -> Expression:
    op = value.get("op")
    if op == "category_in":
        return CategoryIn(tuple(value["values"]))
    if op == "not":
        return Not(expression_from_dict(value["child"]))
    if op == "and":
        return AllOf(tuple(expression_from_dict(v) for v in value["children"]))
    if op == "or":
        return AnyOf(tuple(expression_from_dict(v) for v in value["children"]))
    if op == "implies":
        return Implies(
            expression_from_dict(value["antecedent"]),
            expression_from_dict(value["consequent"]),
        )
    raise ValueError(f"Unsupported constraint expression operation: {op!r}")


def condition_matches(condition: Mapping[str, Any] | None, inputs: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    op = condition.get("op", "field")
    if op == "field":
        return FieldCondition(
            condition["field"], condition.get("cmp", "eq"), condition["value"]
        ).evaluate(inputs)
    if op == "and":
        return all(condition_matches(child, inputs) for child in condition["children"])
    if op == "or":
        return any(condition_matches(child, inputs) for child in condition["children"])
    if op == "not":
        return not condition_matches(condition["child"], inputs)
    raise ValueError(f"Unsupported condition operation: {op!r}")


class BoundaryLossEngine:
    """Scores mass on outputs satisfying every active rule.

    The model distribution is categorical over the explicitly supplied finite
    categories. It is not a distribution over arbitrary strings.
    """

    def __init__(
        self,
        categories: Sequence[str],
        constraints: Iterable[Constraint],
        max_enumerated_outputs: int = 256,
    ) -> None:
        self.categories = list(categories)
        self.constraints = list(constraints)
        if len(self.categories) > max_enumerated_outputs:
            raise ValueError(
                f"Output space has {len(self.categories)} values; limit is {max_enumerated_outputs}"
            )
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("Output categories must be unique")
        for constraint in self.constraints:
            expression_from_dict(constraint.rule)

    def valid_mask(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        valid = []
        active_ids = []
        active = [c for c in self.constraints if condition_matches(c.when, inputs)]
        for constraint in active:
            active_ids.append(constraint.constraint_id)
        for category in self.categories:
            valid.append(all(expression_from_dict(c.rule).evaluate(category) for c in active))
        if not any(valid):
            raise UnsatisfiableConstraints(
                "Active constraints leave no valid output for input "
                f"{dict(inputs)!r}; active constraint ids: {active_ids}"
            )
        return torch.tensor(valid, dtype=torch.bool)

    def validate_cases(self, cases: Iterable[Mapping[str, Any]]) -> None:
        for case in cases:
            self.valid_mask(case)

    def loss(self, logits: torch.Tensor, inputs: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(self.categories):
            raise ValueError("Expected [batch, categories] logits")
        if logits.shape[0] != len(inputs):
            raise ValueError("Logit batch and input batch sizes differ")
        losses: List[torch.Tensor] = []
        for row, fields in zip(logits, inputs):
            mask = self.valid_mask(fields).to(row.device)
            log_valid_mass = torch.logsumexp(row[mask], dim=0) - torch.logsumexp(row, dim=0)
            losses.append(-log_valid_mass)
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0

    def violation_rate(
        self, predicted: Sequence[str], inputs: Sequence[Mapping[str, Any]]
    ) -> float:
        if not predicted:
            return 0.0
        violations = 0
        for output, fields in zip(predicted, inputs):
            mask = self.valid_mask(fields)
            index = self.categories.index(output)
            violations += int(not bool(mask[index]))
        return violations / len(predicted)


def assert_finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite {label}")


def safe_float(value: float) -> float:
    return value if math.isfinite(value) else float("nan")
