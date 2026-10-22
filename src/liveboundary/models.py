# Copyright 2026 LiveBoundary contributors
# SPDX-License-Identifier: Apache-2.0

"""Model adapter protocol and the offline PyTorch reference adapter."""

from __future__ import annotations

import copy
import fnmatch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import VerifiedExample


class ModelAdapter(Protocol):
    categories: Sequence[str]

    def predict(self, inputs: Sequence[Mapping[str, Any]]) -> List[str]: ...
    def logits(self, inputs: Sequence[Mapping[str, Any]]) -> torch.Tensor: ...
    def task_loss(self, examples: Sequence[VerifiedExample]) -> torch.Tensor: ...
    def clone(self) -> ModelAdapter: ...
    def select_parameters(
        self, mode: str, patterns: Sequence[str]
    ) -> List[Tuple[str, nn.Parameter]]: ...
    def save(self, path: Path, metadata: Mapping[str, Any] | None = None) -> None: ...


@dataclass(frozen=True)
class TinyModelConfig:
    fields: Sequence[str]
    vocabularies: Mapping[str, Sequence[str]]
    categories: Sequence[str]
    embedding_dim: int = 12
    hidden_dim: int = 24

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["fields"] = list(self.fields)
        value["categories"] = list(self.categories)
        value["vocabularies"] = {k: list(v) for k, v in self.vocabularies.items()}
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TinyModelConfig:
        return cls(**dict(value))


class TinyStructuredNetwork(nn.Module):
    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(config.vocabularies[field]) + 1, config.embedding_dim)
                for field in config.fields
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(len(config.fields) * config.embedding_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, len(config.categories)),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        parts = [
            self.embeddings[field](encoded[:, i]) for i, field in enumerate(self.config.fields)
        ]
        return self.classifier(torch.cat(parts, dim=-1))


class TinyModelAdapter:
    """A small structured classifier whose existing native weights are updated."""

    format_version = 1

    def __init__(
        self, config: TinyModelConfig, network: TinyStructuredNetwork | None = None
    ) -> None:
        self.config = config
        self.categories = list(config.categories)
        self.network = network or TinyStructuredNetwork(config)
        self._indexes = {
            field: {str(value): index + 1 for index, value in enumerate(config.vocabularies[field])}
            for field in config.fields
        }

    @classmethod
    def seeded(cls, config: TinyModelConfig, seed: int) -> TinyModelAdapter:
        torch.manual_seed(seed)
        return cls(config)

    def encode(self, inputs: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        rows = []
        for item in inputs:
            rows.append(
                [
                    self._indexes[field].get(str(item.get(field, "<missing>")), 0)
                    for field in self.config.fields
                ]
            )
        return torch.tensor(rows, dtype=torch.long)

    def logits(self, inputs: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        return self.network(self.encode(inputs))

    def predict(self, inputs: Sequence[Mapping[str, Any]]) -> List[str]:
        self.network.eval()
        with torch.no_grad():
            indices = self.logits(inputs).argmax(dim=-1).tolist()
        return [self.categories[index] for index in indices]

    def task_loss(self, examples: Sequence[VerifiedExample]) -> torch.Tensor:
        if not examples:
            return next(self.network.parameters()).sum() * 0.0
        targets = torch.tensor([self.categories.index(example.target) for example in examples])
        return F.cross_entropy(self.logits([example.inputs for example in examples]), targets)

    def clone(self) -> TinyModelAdapter:
        return TinyModelAdapter(self.config, copy.deepcopy(self.network))

    def state_snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone() for name, value in self.network.state_dict().items()
        }

    def select_parameters(
        self, mode: str = "all", patterns: Sequence[str] = ()
    ) -> List[Tuple[str, nn.Parameter]]:
        named = list(self.network.named_parameters())
        for _, parameter in named:
            parameter.requires_grad_(False)
        if mode == "all":
            selected = named
        elif mode in {"patterns", "modules"}:
            if not patterns:
                raise ValueError("Explicit parameter selection requires at least one pattern")
            selected = [
                (name, parameter)
                for name, parameter in named
                if any(
                    fnmatch.fnmatch(name, pattern) or name.startswith(pattern + ".")
                    for pattern in patterns
                )
            ]
        else:
            raise ValueError(f"Unknown trainable selection mode: {mode!r}")
        if not selected:
            available = ", ".join(name for name, _ in named)
            raise ValueError(
                f"Parameter selection matched nothing. Available parameters: {available}"
            )
        for _, parameter in selected:
            parameter.requires_grad_(True)
        return selected

    def save(self, path: Path, metadata: Mapping[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": self.format_version,
                "adapter": "tiny_structured",
                "config": self.config.to_dict(),
                "state_dict": self.network.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TinyModelAdapter:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 2.0 compatibility
            payload = torch.load(path, map_location="cpu")
        if payload.get("adapter") != "tiny_structured":
            raise ValueError(f"Unsupported checkpoint adapter: {payload.get('adapter')!r}")
        config = TinyModelConfig.from_dict(payload["config"])
        adapter = cls(config)
        adapter.network.load_state_dict(payload["state_dict"])
        return adapter


def state_deltas(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> Dict[str, float]:
    result = {}
    for name, prior in before.items():
        delta = torch.linalg.vector_norm((after[name] - prior).float()).item()
        if delta > 0:
            result[name] = delta
    return result


def states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


class LocalCausalLMAdapter:
    """Optional local-only candidate-sequence adapter.

    Its probabilities are normalized over explicitly declared candidate strings,
    not over the full language distribution. Importing this class does not require
    Transformers; constructing it does.
    """

    def __init__(self, model_path: str, tokenizer_path: str, candidates: Sequence[str]) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install liveboundary[hf] to use LocalCausalLMAdapter") from exc
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.categories = list(candidates)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.network = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    def logits(self, inputs: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        scores = []
        for item in inputs:
            prompt = str(item["prompt"])
            row = []
            for candidate in self.categories:
                prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                joined = self.tokenizer(prompt + candidate, return_tensors="pt").input_ids
                labels = joined.clone()
                labels[:, : prompt_ids.shape[1]] = -100
                predicted = self.network(input_ids=joined).logits[:, :-1]
                target = joined[:, 1:]
                mask = labels[:, 1:] != -100
                token_logp = (
                    F.log_softmax(predicted, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                )
                row.append(token_logp[mask].sum())
            scores.append(torch.stack(row))
        return torch.stack(scores)

    def predict(self, inputs: Sequence[Mapping[str, Any]]) -> List[str]:
        indexes = self.logits(inputs).argmax(dim=-1).tolist()
        return [self.categories[index] for index in indexes]

    def task_loss(self, examples: Sequence[VerifiedExample]) -> torch.Tensor:
        targets = torch.tensor([self.categories.index(example.target) for example in examples])
        return F.cross_entropy(self.logits([example.inputs for example in examples]), targets)

    def clone(self) -> LocalCausalLMAdapter:
        return copy.deepcopy(self)

    def select_parameters(
        self, mode: str, patterns: Sequence[str]
    ) -> List[Tuple[str, nn.Parameter]]:
        named = list(self.network.named_parameters())
        for _, parameter in named:
            parameter.requires_grad_(False)
        selected = (
            named
            if mode == "all"
            else [
                (name, parameter)
                for name, parameter in named
                if any(
                    fnmatch.fnmatch(name, pattern) or name.startswith(pattern + ".")
                    for pattern in patterns
                )
            ]
        )
        if not selected:
            raise ValueError("Causal-LM parameter selection matched nothing")
        for _, parameter in selected:
            parameter.requires_grad_(True)
        return selected
