# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from liveboundary import CategoryIn, Constraint, Not, UnsatisfiableConstraints
from liveboundary.constraints import BoundaryLossEngine


def test_exact_satisfaction_mass_and_gradient():
    rule = Constraint("no-collect", "1", "forbid collect", Not(CategoryIn(["collect"])).to_dict())
    engine = BoundaryLossEngine(["collect", "safe"], [rule])
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = engine.loss(logits, [{}])
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0


def test_contradictory_constraints_are_actionable():
    constraints = [
        Constraint("only-a", "1", "a", CategoryIn(["a"]).to_dict()),
        Constraint("only-b", "1", "b", CategoryIn(["b"]).to_dict()),
    ]
    engine = BoundaryLossEngine(["a", "b"], constraints)
    with pytest.raises(UnsatisfiableConstraints, match="only-a.*only-b"):
        engine.valid_mask({})


def test_output_enumeration_limit():
    with pytest.raises(ValueError, match="limit"):
        BoundaryLossEngine(["a", "b"], [], max_enumerated_outputs=1)
