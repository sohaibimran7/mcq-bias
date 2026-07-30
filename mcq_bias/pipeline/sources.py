"""Source MCQ datasets for the pipeline, loaded from HuggingFace.

Each loader maps a public dataset to canonical MCQRecords. Loading is
network-dependent (HF hub); pin ``revision`` for published evals.

Fidelity note: for mmlu the question/options mapping reproduces the
canonicalization of the original cot-transparency codebase, so record hashes
line up with the data it released (same questions → same
original_question_hash). For truthfulqa/logiqa/hellaswag the mapping is a
faithful reconstruction, but hash overlap with the released data is not
guaranteed (the original loaders applied their own formatting) — the eval is
self-consistent either way, since biased/unbiased pairs are built from the
same records by construction.
"""

import hashlib
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import ascii_uppercase
from typing import Optional

from mcq_bias.pipeline.records import MCQRecord

# Built-in dataset aliases pinned to explicit HF dataset-repo commit SHAs
# (fetched 2026-07-10) so rebuilding a frozen file can never drift with
# upstream. Pass ``revision=`` to load_records to override.
BUILTIN_DATASETS = {
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",  # cais/mmlu
    "truthfulqa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",  # truthfulqa/truthful_qa
    # lucasmccabe/logiqa main is a script dataset (unloadable with datasets>=3);
    # pinned to its refs/convert/parquet branch instead.
    "logiqa": "fa9f9918fa81eca088805c1395d7f592f7755ae0",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",  # Rowan/hellaswag
}


def _shuffled_capped(records: list[MCQRecord], n_questions: Optional[int], seed: str = "42") -> list[MCQRecord]:
    # Match the original cot-transparency sampling: shuffle(seed="42") then take(n).
    out = list(records)
    random.Random(seed).shuffle(out)
    # Dedupe by content hash AFTER shuffling (keeps the shuffled order, and with it
    # the small-n-is-a-prefix-of-large-n property). Sources can repeat a question
    # verbatim (e.g. mmlu carries some questions in several subjects); sample id =
    # question id, so a duplicate would collide inside one eval and make the by-id
    # switch join ambiguous.
    seen: set[str] = set()
    out = [r for r in out if not (r.question_id in seen or seen.add(r.question_id))]
    return out[:n_questions] if n_questions else out


def read_question_ids(path: str | Path) -> set[str]:
    """The unique ``question_id`` values in a JSONL file — any file whose rows
    carry the field works (e.g. a wrong-argument store or a frozen dataset).
    Used by ``question_ids_from`` to restrict a task's question pool."""
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid = json.loads(line).get("question_id")
            if qid:
                ids.add(qid)
    if not ids:
        raise ValueError(f"{path}: no 'question_id' fields found — not a usable question-id source")
    return ids


def dataset_slug(dataset: str) -> str:
    """Filesystem-safe identifier for a dataset (part of frozen file names).

    Built-in aliases pass through; HF ids and local paths are sanitized
    ("org/name" -> "org-name", "path/to/file.jsonl" -> "file")."""
    if dataset in BUILTIN_DATASETS:
        return dataset
    from pathlib import Path as _P

    name = _P(dataset).stem if dataset.endswith((".jsonl", ".json", ".csv")) else dataset
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()


def _field_value(row: Mapping, field: str):
    """Resolve a dotted field path through nested mappings."""

    value = row
    for component in field.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"field path {field!r} is missing component {component!r}")
        value = value[component]
    return value


def _answer_index(
    answer,
    *,
    field: str,
    labels: Optional[list[str]] = None,
    explicit_labels: bool = False,
) -> int:
    if explicit_labels:
        candidate = str(answer).strip()
        if len(candidate) >= 2 and (candidate[0], candidate[-1]) in {
            ("(", ")"),
            ("[", "]"),
        }:
            candidate = candidate[1:-1].strip()
        exact = [index for index, label in enumerate(labels or []) if candidate == label]
        if len(exact) == 1:
            return exact[0]
        folded = [index for index, label in enumerate(labels or []) if candidate.casefold() == label.casefold()]
        if len(folded) == 1:
            return folded[0]
        raise ValueError(f"answer field {field!r} value {answer!r} does not name one of the choice labels {labels}")
    if isinstance(answer, str) and len(answer.strip()) == 1:
        label = answer.strip().upper()
        if label in ascii_uppercase:
            return ascii_uppercase.index(label)
    try:
        return int(answer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"answer field {field!r} must hold an option letter or integer index") from exc


def _choice_text_and_labels(raw_choices, *, field: str) -> tuple[list[str], list[str], bool]:
    """Normalize common MCQ choice containers without discarding their labels."""

    pairs: list[tuple[object, object]]
    explicit_labels = False
    if isinstance(raw_choices, Mapping):
        label_keys = [key for key in ("label", "labels") if key in raw_choices]
        text_keys = [key for key in ("text", "texts") if key in raw_choices]
        if label_keys or text_keys:
            if len(label_keys) != 1 or len(text_keys) != 1:
                raise ValueError(f"choices field {field!r} needs exactly one label(s) and one text(s) array")
            labels = raw_choices[label_keys[0]]
            texts = raw_choices[text_keys[0]]
            if (
                not isinstance(labels, Sequence)
                or isinstance(labels, (str, bytes))
                or not isinstance(texts, Sequence)
                or isinstance(texts, (str, bytes))
                or len(labels) != len(texts)
            ):
                raise ValueError(f"choices field {field!r} label and text values must be equal-length arrays")
            pairs = list(zip(labels, texts))
        else:
            pairs = list(raw_choices.items())
        explicit_labels = True
    elif isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
        values = list(raw_choices)
        if not 2 <= len(values) <= len(ascii_uppercase):
            raise ValueError(f"choices field {field!r} must contain 2..26 options")
        if all(isinstance(value, str) for value in values):
            pairs = list(zip(ascii_uppercase, values))
        elif all(isinstance(value, Mapping) for value in values):
            have_labels = [any(key in value for key in ("label", "key")) for value in values]
            if any(have_labels) and not all(have_labels):
                raise ValueError(f"choices field {field!r} mixes labeled and unlabeled choice objects")
            pairs = []
            for index, value in enumerate(values):
                text_keys = [key for key in ("text", "value", "option") if key in value]
                if len(text_keys) != 1:
                    raise ValueError(f"choices field {field!r}[{index}] needs exactly one text/value/option field")
                if all(have_labels):
                    label_keys = [key for key in ("label", "key") if key in value]
                    if len(label_keys) != 1:
                        raise ValueError(f"choices field {field!r}[{index}] has ambiguous labels")
                    label = value[label_keys[0]]
                else:
                    label = ascii_uppercase[index]
                pairs.append((label, value[text_keys[0]]))
            explicit_labels = all(have_labels)
        else:
            raise ValueError(f"choices field {field!r} must contain only strings or only choice objects")
    else:
        raise ValueError(f"choices field {field!r} must be an array or a label-to-text mapping")

    if not 2 <= len(pairs) <= len(ascii_uppercase):
        raise ValueError(f"choices field {field!r} must contain 2..26 options")
    labels = [str(label).strip() for label, _ in pairs]
    texts = [text for _, text in pairs]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"choices field {field!r} contains empty or duplicate labels")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"choices field {field!r} contains a non-string or empty option")
    return texts, labels, explicit_labels


def _load_local_jsonl(
    path: str,
    *,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
) -> list[MCQRecord]:
    """Local JSONL rows: {"question", "options" (or "choices"), "answer" (letter or index)}."""

    records = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                if choices_field == "choices" and "choices" not in row and "options" in row:
                    raw_choices = row["options"]
                else:
                    raw_choices = _field_value(row, choices_field)
                options, labels, explicit_labels = _choice_text_and_labels(raw_choices, field=choices_field)
                question = _field_value(row, question_field)
                answer = _field_value(row, answer_field)
                gt_idx = _answer_index(
                    answer,
                    field=answer_field,
                    labels=labels,
                    explicit_labels=explicit_labels,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid MCQ row: {exc}") from exc
            records.append(
                MCQRecord(
                    question=question,
                    options=options,
                    ground_truth_idx=gt_idx,
                    dataset=dataset_slug(path),
                )
            )
    return records


def source_identity(
    dataset: str,
    *,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    revision: Optional[str] = None,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
) -> dict:
    """Canonical identity of the selected source and schema mapping."""

    if dataset.endswith(".jsonl") or os.path.exists(dataset):
        path = Path(dataset).resolve()
        identity = {
            "kind": "local_jsonl",
            "path": str(path),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    else:
        identity = {
            "kind": "huggingface",
            "dataset": dataset,
            "dataset_config": dataset_config,
            "split": split or ("test" if dataset not in BUILTIN_DATASETS else None),
            "revision": revision or BUILTIN_DATASETS.get(dataset, "main"),
        }
    identity["fields"] = {
        "question": question_field,
        "choices": choices_field,
        "answer": answer_field,
    }
    return identity


def load_records(
    dataset: str,
    n_questions: Optional[int] = None,
    seed: str = "42",
    revision: Optional[str] = None,
    *,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
) -> list[MCQRecord]:
    """Load canonical MCQ records from a source dataset.

    ``dataset`` may be a built-in alias (mmlu, truthfulqa, logiqa, hellaswag),
    a local JSONL path (rows: question/options/answer), or any HuggingFace
    dataset id — for the generic HF case, ``dataset_config``/``split`` and the
    three field names map its schema onto (question, options, ground-truth
    index; the answer field may hold a letter or an index)."""
    if dataset.endswith(".jsonl") or os.path.exists(dataset):
        if dataset_config is not None or split is not None or revision is not None:
            raise ValueError("local JSONL datasets do not accept dataset_config, split, or revision")
        return _shuffled_capped(
            _load_local_jsonl(
                dataset,
                question_field=question_field,
                choices_field=choices_field,
                answer_field=answer_field,
            ),
            n_questions,
            seed,
        )

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("live data source requires the `datasets` package (`uv pip install datasets`)") from e

    rev = revision or BUILTIN_DATASETS.get(dataset, "main")
    records: list[MCQRecord] = []

    if dataset == "mmlu":
        _reject_builtin_schema_overrides(dataset, dataset_config, split, question_field, choices_field, answer_field)
        ds = load_dataset("cais/mmlu", "all", split="test", revision=rev)
        for row in ds:
            records.append(
                MCQRecord(
                    question=row["question"],
                    options=list(row["choices"]),
                    ground_truth_idx=int(row["answer"]),
                    dataset="mmlu",
                )
            )
    elif dataset == "truthfulqa":
        _reject_builtin_schema_overrides(dataset, dataset_config, split, question_field, choices_field, answer_field)
        # full org/name id: datasets>=3 no longer redirects bare canonical names
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision=rev)
        for row in ds:
            labels = row["mc1_targets"]["labels"]
            records.append(
                MCQRecord(
                    question=row["question"],
                    options=list(row["mc1_targets"]["choices"]),
                    ground_truth_idx=labels.index(1),
                    dataset="truthfulqa",
                )
            )
    elif dataset == "logiqa":
        _reject_builtin_schema_overrides(dataset, dataset_config, split, question_field, choices_field, answer_field)
        ds = load_dataset("lucasmccabe/logiqa", split="test", revision=rev)
        for row in ds:
            records.append(
                MCQRecord(
                    question=f"{row['context']}\n{row['query']}",
                    options=list(row["options"]),
                    ground_truth_idx=int(row["correct_option"]),
                    dataset="logiqa",
                )
            )
    elif dataset == "hellaswag":
        _reject_builtin_schema_overrides(dataset, dataset_config, split, question_field, choices_field, answer_field)
        ds = load_dataset("Rowan/hellaswag", split="validation", revision=rev)
        for row in ds:
            records.append(
                MCQRecord(
                    question=("Which of the answer choices best completes the following sentence?\n" f"{row['ctx']}"),
                    options=list(row["endings"]),
                    ground_truth_idx=int(row["label"]),
                    dataset="hellaswag",
                )
            )
    else:
        # Generic HF dataset with explicit field mapping.
        ds = load_dataset(dataset, dataset_config, split=split or "test", revision=rev)
        for row in ds:
            answer = _field_value(row, answer_field)
            options, labels, explicit_labels = _choice_text_and_labels(
                _field_value(row, choices_field),
                field=choices_field,
            )
            gt_idx = _answer_index(
                answer,
                field=answer_field,
                labels=labels,
                explicit_labels=explicit_labels,
            )
            records.append(
                MCQRecord(
                    question=_field_value(row, question_field),
                    options=options,
                    ground_truth_idx=gt_idx,
                    dataset=dataset_slug(dataset),
                )
            )

    return _shuffled_capped(records, n_questions, seed)


def _reject_builtin_schema_overrides(
    dataset: str,
    dataset_config: Optional[str],
    split: Optional[str],
    question_field: str,
    choices_field: str,
    answer_field: str,
) -> None:
    if (
        dataset_config is not None
        or split is not None
        or question_field != "question"
        or choices_field != "choices"
        or answer_field != "answer"
    ):
        raise ValueError(
            f"built-in dataset {dataset!r} has a pinned loader; schema/config/split overrides "
            "apply only to generic Hugging Face or local JSONL datasets"
        )
