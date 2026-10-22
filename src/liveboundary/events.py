# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

"""Verified-event policy and temporal fact resolution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .contracts import BusinessEvent, EventKind, VerificationStatus


class EventLedger:
    """Resolves verified temporal facts without treating unverified data as truth."""

    def __init__(self, unverified_policy: str = "reject") -> None:
        if unverified_policy not in {"reject", "ignore"}:
            raise ValueError("unverified_policy must be 'reject' or 'ignore'")
        self.unverified_policy = unverified_policy
        self._events: Dict[str, BusinessEvent] = {}

    def add(self, event: BusinessEvent) -> bool:
        if event.verification_status is not VerificationStatus.VERIFIED:
            if self.unverified_policy == "reject":
                raise ValueError(f"Event {event.event_id} is not verified")
            return False
        if event.event_id in self._events:
            raise ValueError(f"Duplicate event_id: {event.event_id}")
        for reference in (event.corrects, event.supersedes):
            if reference and reference not in self._events:
                raise ValueError(f"Event {event.event_id} references unknown event {reference}")
        self._events[event.event_id] = event
        return True

    @staticmethod
    def _time(value: Optional[str], fallback: str) -> datetime:
        return datetime.fromisoformat(value or fallback)

    def resolve(
        self,
        business_id: str,
        subject: str,
        field: str,
        as_of: str,
    ) -> Any:
        cutoff = datetime.fromisoformat(as_of)
        matching = [
            event
            for event in self._events.values()
            if event.business_id == business_id
            and event.kind in {EventKind.FACT, EventKind.CORRECTION, EventKind.TEMPORAL_CHANGE}
            and event.payload.get("subject") == subject
            and event.payload.get("field") == field
            and self._time(event.effective_at, event.observed_at) <= cutoff
        ]
        if not matching:
            raise KeyError(f"No verified value for {subject}.{field} at {as_of}")
        matching.sort(key=lambda e: (self._time(e.effective_at, e.observed_at), e.observed_at))
        return matching[-1].payload["value"]

    def verified_events(self) -> List[BusinessEvent]:
        return list(self._events.values())
