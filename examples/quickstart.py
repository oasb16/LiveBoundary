"""Minimal offline constrained-adaptation example."""

from pathlib import Path
from tempfile import TemporaryDirectory

from liveboundary import (
    AdaptationConfig,
    BusinessLearner,
    CategoryIn,
    Constraint,
    EvaluationSuite,
    Not,
    TinyModelAdapter,
    TinyModelConfig,
    VerifiedExample,
)


def example(identifier, status, target, wording):
    return VerifiedExample(
        "demo-business",
        identifier,
        {"account": "a", "status": status, "wording": wording},
        target,
        "verified synthetic demonstration",
    )


model_config = TinyModelConfig(
    fields=["account", "status", "wording"],
    vocabularies={"account": ["a"], "status": ["settled", "open"], "wording": ["train", "heldout"]},
    categories=["collect", "no_action"],
)
baseline = TinyModelAdapter.seeded(model_config, seed=7)
constraint = Constraint(
    "settled-no-collection",
    "1",
    "Settled accounts must not receive collection action",
    Not(CategoryIn(["collect"])).to_dict(),
    {"op": "field", "field": "status", "cmp": "eq", "value": "settled"},
)
train = [
    example("train-settled", "settled", "no_action", "train"),
    example("train-open", "open", "collect", "train"),
]
validation = [
    example("held-settled", "settled", "no_action", "heldout"),
    example("held-open", "open", "collect", "heldout"),
]

with TemporaryDirectory() as directory:
    learner = BusinessLearner(
        business_id="demo-business",
        model=baseline,
        baseline=baseline,
        constraints=[constraint],
        config=AdaptationConfig(
            gradient_steps=100,
            boundary_weight=2.0,
            min_new_accuracy=0.5,
            max_boundary_violation_rate=0.0,
            seed=7,
        ),
        checkpoint_dir=Path(directory),
    )
    report = learner.adapt(
        train, EvaluationSuite(new_examples=validation, useful_examples=[validation[1]])
    )
    print("accepted:", report.accepted)
    print("changed native parameters:", sorted(report.changed_parameters))
    print("prediction:", learner.predict(validation[0].inputs))
    if report.accepted:
        print("fresh process:", learner.strict_predict(validation[0].inputs))
