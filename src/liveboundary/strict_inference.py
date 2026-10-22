# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process inference that loads only one checkpoint and one question."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .models import TinyModelAdapter


def main() -> None:
    request = json.load(sys.stdin)
    allowed = {"checkpoint", "question"}
    extra = set(request) - allowed
    if extra:
        raise ValueError(f"Strict inference request includes forbidden keys: {sorted(extra)}")
    checkpoint = Path(request["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = TinyModelAdapter.load(checkpoint)
    result = model.predict([request["question"]])[0]
    json.dump(
        {
            "result": result,
            "inputs_read": ["checkpoint", "question"],
            "external_memory_read": False,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
