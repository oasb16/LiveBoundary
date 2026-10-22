import pytest

from liveboundary import TinyModelAdapter, TinyModelConfig, VerifiedExample


@pytest.fixture
def tiny_config():
    return TinyModelConfig(
        fields=["entity", "status", "wording"],
        vocabularies={
            "entity": ["a", "b"],
            "status": ["settled", "open"],
            "wording": ["train", "heldout"],
        },
        categories=["collect", "no_action", "refund"],
        embedding_dim=5,
        hidden_dim=8,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return TinyModelAdapter.seeded(tiny_config, 9)


def make_example(identifier, target, status="open", wording="train", business="biz"):
    return VerifiedExample(
        business,
        identifier,
        {"entity": "a", "status": status, "wording": wording},
        target,
        "test fixture",
    )


@pytest.fixture
def example_factory():
    return make_example
