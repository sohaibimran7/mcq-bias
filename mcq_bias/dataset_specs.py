"""Canonical heterogeneous dataset specifications for ``mcq_bias`` suites."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DATASET_SPEC_FIELDS = frozenset(
    {
        "dataset",
        "dataset_config",
        "split",
        "revision",
        "question_field",
        "choices_field",
        "answer_field",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One source dataset plus its canonical MCQ field mapping."""

    dataset: str
    dataset_config: str | None = None
    split: str | None = None
    question_field: str = "question"
    choices_field: str = "choices"
    answer_field: str = "answer"
    revision: str | None = None

    def __post_init__(self) -> None:
        for field in ("dataset", "question_field", "choices_field", "answer_field"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset spec field {field!r} must be a non-empty string")
        for field in ("dataset_config", "split", "revision"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"dataset spec field {field!r} must be None or a non-empty string")

    def as_dict(self, *, include_defaults: bool = True) -> dict[str, str]:
        values = {
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "question_field": self.question_field,
            "choices_field": self.choices_field,
            "answer_field": self.answer_field,
            "revision": self.revision,
        }
        if include_defaults:
            return {key: value for key, value in values.items() if value is not None}
        defaults = {
            "question_field": "question",
            "choices_field": "choices",
            "answer_field": "answer",
        }
        return {
            key: value
            for key, value in values.items()
            if value is not None and (key == "dataset" or key not in defaults or value != defaults[key])
        }


DatasetInput = str | Mapping[str, Any] | DatasetSpec


def parse_dataset_cli_token(value: str) -> DatasetInput:
    """Decode one JSON object token while preserving ordinary dataset names."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset CLI values must be non-empty strings")
    candidate = value.strip()
    if not candidate.startswith("{"):
        return value
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dataset-spec JSON: {exc.msg}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("dataset-spec JSON must decode to an object")
    return decoded


def normalize_dataset_spec(value: DatasetInput) -> DatasetSpec:
    """Normalize one string, mapping, or already-validated specification."""

    if isinstance(value, DatasetSpec):
        return value
    if isinstance(value, str):
        return DatasetSpec(dataset=value)
    if not isinstance(value, Mapping):
        raise ValueError("dataset specifications must be strings or mappings")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("dataset spec field names must be strings")
    unknown = sorted(set(value) - DATASET_SPEC_FIELDS)
    if unknown:
        raise ValueError(f"unknown dataset spec field(s): {', '.join(unknown)}")
    if "dataset" not in value:
        raise ValueError("dataset spec is missing required field 'dataset'")
    return DatasetSpec(**dict(value))


def normalize_dataset_specs(values: Sequence[DatasetInput]) -> tuple[DatasetSpec, ...]:
    """Normalize a non-empty, duplicate-free sequence."""

    if isinstance(values, (str, bytes, bytearray)) or not values:
        raise ValueError("datasets must be a non-empty sequence")
    specs = tuple(normalize_dataset_spec(value) for value in values)
    canonical = [json.dumps(spec.as_dict(), sort_keys=True, separators=(",", ":")) for spec in specs]
    if len(canonical) != len(set(canonical)):
        raise ValueError("dataset specifications contain duplicates")
    return specs


def parse_dataset_cli_tokens(values: Sequence[str]) -> tuple[DatasetSpec, ...]:
    """Parse CLI tokens and normalize them in one operation."""

    return normalize_dataset_specs([parse_dataset_cli_token(value) for value in values])


__all__ = [
    "DATASET_SPEC_FIELDS",
    "DatasetInput",
    "DatasetSpec",
    "normalize_dataset_spec",
    "normalize_dataset_specs",
    "parse_dataset_cli_token",
    "parse_dataset_cli_tokens",
]
