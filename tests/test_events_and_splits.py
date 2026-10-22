import pytest

from liveboundary import BusinessEvent, EvaluationSuite, EventKind, VerificationStatus
from liveboundary.events import EventLedger


def event(identifier, value, effective, kind=EventKind.FACT, corrects=None, verified=True):
    return BusinessEvent(
        "biz",
        identifier,
        "2026-01-01T00:00:00+00:00",
        "fixture",
        VerificationStatus.VERIFIED if verified else VerificationStatus.UNVERIFIED,
        kind,
        {"subject": "home-a", "field": "rent", "value": value},
        effective_at=effective,
        corrects=corrects,
    )


def test_temporal_change_and_correction_semantics():
    ledger = EventLedger()
    ledger.add(event("initial", 1200, "2025-01-01T00:00:00+00:00"))
    ledger.add(event("change", 1400, "2026-01-01T00:00:00+00:00", EventKind.TEMPORAL_CHANGE))
    ledger.add(
        event("correction", 1350, "2026-01-01T00:00:00+00:00", EventKind.CORRECTION, "change")
    )
    assert ledger.resolve("biz", "home-a", "rent", "2025-06-01T00:00:00+00:00") == 1200
    assert ledger.resolve("biz", "home-a", "rent", "2026-06-01T00:00:00+00:00") == 1350


def test_unverified_policy_is_explicit():
    with pytest.raises(ValueError, match="not verified"):
        EventLedger("reject").add(event("rumor", 999, "2025-01-01T00:00:00+00:00", verified=False))
    ledger = EventLedger("ignore")
    assert ledger.add(event("rumor", 999, "2025-01-01T00:00:00+00:00", verified=False)) is False


def test_benchmark_split_integrity(example_factory):
    same = example_factory("duplicate", "collect")
    suite = EvaluationSuite(new_examples=[same], final_test_examples=[same])
    with pytest.raises(ValueError, match="overlap"):
        suite.validation_examples()
