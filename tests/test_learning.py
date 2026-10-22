# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

from liveboundary import AdaptationConfig, BusinessLearner, EvaluationSuite
from liveboundary.models import states_equal


def test_native_parameter_change_and_immutable_baseline(tmp_path, tiny_model, example_factory):
    original = tiny_model.state_snapshot()
    learner = BusinessLearner(
        "biz",
        tiny_model,
        tiny_model,
        [],
        AdaptationConfig(gradient_steps=20, min_new_accuracy=0.0),
        tmp_path,
    )
    report = learner.adapt(
        [example_factory("train", "collect")],
        EvaluationSuite(new_examples=[example_factory("val", "collect", wording="heldout")]),
    )
    assert report.accepted
    assert report.changed_parameters
    assert states_equal(original, learner.baseline.state_snapshot())
    assert not states_equal(original, learner.model.state_snapshot())


def test_parameter_selection_and_empty_selection(tiny_model):
    selected = tiny_model.select_parameters("patterns", ["classifier.2.*"])
    assert selected
    assert all(name.startswith("classifier.2") for name, _ in selected)
    assert all(parameter.requires_grad for _, parameter in selected)
    assert all(
        (name.startswith("classifier.2")) == parameter.requires_grad
        for name, parameter in tiny_model.network.named_parameters()
    )
    try:
        tiny_model.select_parameters("patterns", ["does.not.exist"])
    except ValueError as error:
        assert "matched nothing" in str(error)
    else:
        raise AssertionError("empty selection did not fail")


def test_rejected_candidate_preserves_accepted_model(tmp_path, tiny_model, example_factory):
    learner = BusinessLearner(
        "biz",
        tiny_model,
        tiny_model,
        [],
        AdaptationConfig(gradient_steps=5, min_new_accuracy=1.1),
        tmp_path,
    )
    before = learner.model.state_snapshot()
    report = learner.adapt(
        [example_factory("train", "collect")],
        EvaluationSuite(new_examples=[example_factory("val", "collect", wording="heldout")]),
    )
    assert not report.accepted
    assert states_equal(before, learner.model.state_snapshot())
    assert learner.current_version is None


def test_business_state_independence(tmp_path, tiny_model, example_factory):
    first = BusinessLearner(
        "one", tiny_model, tiny_model, [], AdaptationConfig(gradient_steps=5), tmp_path
    )
    second = BusinessLearner(
        "two", tiny_model, tiny_model, [], AdaptationConfig(gradient_steps=5), tmp_path
    )
    second_before = second.model.state_snapshot()
    item = example_factory("one-train", "collect", business="one")
    first.adapt(item if isinstance(item, list) else [item], EvaluationSuite())
    assert states_equal(second_before, second.model.state_snapshot())
    for left, right in zip(first.model.network.parameters(), second.model.network.parameters()):
        assert left.data_ptr() != right.data_ptr()


def test_wrong_business_is_rejected(tmp_path, tiny_model, example_factory):
    learner = BusinessLearner("biz", tiny_model, tiny_model, [], AdaptationConfig(), tmp_path)
    try:
        learner.adapt([example_factory("bad", "collect", business="other")], EvaluationSuite())
    except ValueError as error:
        assert "cannot train" in str(error)
    else:
        raise AssertionError("cross-business example was accepted")
