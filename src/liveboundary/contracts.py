"""Serializable public data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class EventKind(str, Enum):
    FACT = "fact"
    CORRECTION = "correction"
    TEMPORAL_CHANGE = "temporal_change"
    BOUNDARY = "boundary"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BusinessEvent:
    business_id: str
    event_id: str
    observed_at: str
    source: str
    verification_status: VerificationStatus
    kind: EventKind
    payload: Mapping[str, Any]
    effective_at: Optional[str] = None
    supersedes: Optional[str] = None
    corrects: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["verification_status"] = self.verification_status.value
        result["kind"] = self.kind.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BusinessEvent:
        data = dict(value)
        data["verification_status"] = VerificationStatus(data["verification_status"])
        data["kind"] = EventKind(data["kind"])
        return cls(**data)


@dataclass(frozen=True)
class VerifiedExample:
    business_id: str
    example_id: str
    inputs: Mapping[str, Any]
    target: str
    provenance: str
    event_id: Optional[str] = None
    effective_at: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.VERIFIED

    def __post_init__(self) -> None:
        if self.verification_status is not VerificationStatus.VERIFIED:
            raise ValueError("VerifiedExample requires verification_status='verified'")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["verification_status"] = self.verification_status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifiedExample:
        data = dict(value)
        data["verification_status"] = VerificationStatus(data["verification_status"])
        return cls(**data)


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    version: str
    description: str
    rule: Mapping[str, Any]
    when: Optional[Mapping[str, Any]] = None
    source: str = "explicit"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Constraint:
        return cls(**dict(value))


@dataclass
class AdaptationConfig:
    learning_rate: float = 0.03
    gradient_steps: int = 80
    task_weight: float = 1.0
    boundary_weight: float = 1.0
    preservation_weight: float = 0.0
    replay_capacity: int = 0
    replay_sample_size: int = 0
    replay_sampling: str = "uniform"
    trainable: str = "all"
    trainable_patterns: List[str] = field(default_factory=list)
    max_enumerated_outputs: int = 256
    min_new_accuracy: float = 0.0
    min_prior_accuracy: float = 0.0
    min_useful_accuracy: float = 0.0
    max_boundary_violation_rate: float = 1.0
    max_stable_regression: float = 1.0
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptationConfig:
        return cls(**dict(value))


@dataclass
class EvaluationReport:
    loss: float
    accuracy: float
    new_fact_accuracy: float
    prior_fact_retention: float
    correction_accuracy: float
    historical_query_accuracy: float
    boundary_violation_rate: float
    useful_action_accuracy: float
    useful_action_coverage: float
    stable_task_accuracy: float
    finite: bool
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationReport:
        return cls(**dict(value))


@dataclass
class ModelVersion:
    version_id: str
    business_id: str
    parent_version_id: Optional[str]
    created_at: str
    checkpoint_path: str
    config_hash: str
    constraint_hash: str
    model_identity: str
    seed: int
    accepted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelVersion:
        return cls(**dict(value))


@dataclass
class AdaptationReport:
    business_id: str
    candidate_version_id: str
    accepted: bool
    reasons: List[str]
    evaluation: EvaluationReport
    changed_parameters: Dict[str, float]
    gradient_steps: int
    update_latency_seconds: float
    checkpoint_size_bytes: int
    replay_examples_used: int
    replay_policy: str
    version: Optional[ModelVersion] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptationReport:
        data = dict(value)
        data["evaluation"] = EvaluationReport.from_dict(data["evaluation"])
        if data.get("version") is not None:
            data["version"] = ModelVersion.from_dict(data["version"])
        return cls(**data)
