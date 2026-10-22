# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

from liveboundary import AdaptationConfig, BusinessLearner, EvaluationSuite
from liveboundary.models import TinyModelAdapter


def test_save_load_and_fresh_process_equivalence(tmp_path, tiny_model, example_factory):
    learner = BusinessLearner(
        "biz", tiny_model, tiny_model, [], AdaptationConfig(gradient_steps=25), tmp_path
    )
    query = {"entity": "a", "status": "open", "wording": "heldout"}
    report = learner.adapt(
        [example_factory("train", "refund")],
        EvaluationSuite(new_examples=[example_factory("val", "refund", wording="heldout")]),
    )
    assert report.accepted and report.version
    loaded = TinyModelAdapter.load(report.version.checkpoint_path)
    assert loaded.predict([query]) == learner.model.predict([query])
    strict = learner.strict_predict(query)
    assert strict["result"] == learner.predict(query)
    assert strict["inputs_read"] == ["checkpoint", "question"]
    assert strict["external_memory_read"] is False


def test_strict_inference_rejects_external_memory_key(tmp_path, tiny_model):
    checkpoint = tmp_path / "model.pt"
    tiny_model.save(checkpoint)
    request = {"checkpoint": str(checkpoint), "question": {}, "replay": ["secret"]}
    completed = subprocess.run(
        [sys.executable, "-m", "liveboundary.strict_inference"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "forbidden keys" in completed.stderr


def test_restart_recovery_and_rollback(tmp_path, tiny_model, example_factory):
    learner = BusinessLearner(
        "biz", tiny_model, tiny_model, [], AdaptationConfig(gradient_steps=8), tmp_path
    )
    first = learner.adapt([example_factory("a", "collect")], EvaluationSuite())
    second = learner.adapt([example_factory("b", "refund")], EvaluationSuite())
    assert first.version and second.version
    learner.rollback(first.version.version_id)
    expected = learner.predict({"entity": "a", "status": "open", "wording": "train"})
    restarted = BusinessLearner("biz", tiny_model, tiny_model, [], AdaptationConfig(), tmp_path)
    assert restarted.current_version.version_id == first.version.version_id
    assert restarted.predict({"entity": "a", "status": "open", "wording": "train"}) == expected
