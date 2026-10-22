"""Atomic checkpoint lineage storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .contracts import ModelVersion
from .models import TinyModelAdapter, states_equal


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStore:
    def __init__(self, root: Path, business_id: str) -> None:
        self.root = Path(root)
        self.business_id = business_id
        self.business_dir = self.root / business_id
        self.versions_dir = self.business_dir / "versions"
        self.business_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.business_dir / "manifest.json"

    def _atomic_json(self, path: Path, value: Mapping[str, Any]) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def initialize_baseline(self, model: TinyModelAdapter) -> str:
        path = self.root / "baseline.pt"
        identity_path = self.root / "baseline.sha256"
        self.root.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(".tmp")
            model.save(temp, {"immutable_baseline": True})
            os.replace(temp, path)
            identity_path.write_text(file_hash(path), encoding="utf-8")
        stored = TinyModelAdapter.load(path)
        if canonical_hash(stored.config.to_dict()) != canonical_hash(
            model.config.to_dict()
        ) or not states_equal(stored.state_snapshot(), model.state_snapshot()):
            raise RuntimeError("Provided baseline does not match the immutable baseline checkpoint")
        expected = identity_path.read_text(encoding="utf-8").strip()
        actual = file_hash(path)
        if expected != actual:
            raise RuntimeError("Immutable baseline checkpoint hash mismatch")
        return actual

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"business_id": self.business_id, "accepted_version_id": None, "versions": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def accepted_version(self) -> Optional[ModelVersion]:
        manifest = self.load_manifest()
        version_id = manifest.get("accepted_version_id")
        for item in manifest["versions"]:
            if item["version_id"] == version_id:
                return ModelVersion(**item)
        return None

    def version(self, version_id: str) -> ModelVersion:
        for item in self.load_manifest()["versions"]:
            if item["version_id"] == version_id and item["accepted"]:
                return ModelVersion(**item)
        raise KeyError(f"No accepted version {version_id!r} for business {self.business_id}")

    def promote(self, model: TinyModelAdapter, version: ModelVersion) -> ModelVersion:
        final_path = self.versions_dir / f"{version.version_id}.pt"
        temp_path = self.versions_dir / f".{version.version_id}.tmp"
        model.save(temp_path, {"version": version.to_dict()})
        os.replace(temp_path, final_path)
        promoted = ModelVersion(
            **{**version.to_dict(), "checkpoint_path": str(final_path), "accepted": True}
        )
        manifest = self.load_manifest()
        manifest["versions"].append(promoted.to_dict())
        manifest["accepted_version_id"] = promoted.version_id
        self._atomic_json(self.manifest_path, manifest)
        return promoted

    def rollback(self, version_id: str) -> ModelVersion:
        version = self.version(version_id)
        manifest = self.load_manifest()
        manifest["accepted_version_id"] = version_id
        manifest.setdefault("rollbacks", []).append(version_id)
        self._atomic_json(self.manifest_path, manifest)
        return version

    def checkpoint_path(self, version_id: str) -> Path:
        return Path(self.version(version_id).checkpoint_path)
