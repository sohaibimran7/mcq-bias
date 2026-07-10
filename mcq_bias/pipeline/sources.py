"""Source MCQ datasets for the Inspect-native pipeline, loaded from HuggingFace.

Each loader maps a public dataset to canonical MCQRecords. Loading is
network-dependent (HF hub); pin ``revision`` for published evals.

Fidelity note: for mmlu the question/options mapping reproduces the legacy
repo's canonicalization, so record hashes line up with the released dumps
(same questions → same original_question_hash). For truthfulqa/logiqa/hellaswag
the mapping is a faithful reconstruction but hash overlap with the dumps is not
guaranteed (the legacy loaders applied their own formatting) — the eval is
self-consistent either way, since biased/unbiased pairs are built from the same
records by construction.
"""

import json
import random
import re
from pathlib import Path
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
    # Match the legacy sampling shape: shuffle(seed="42") then take(n).
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


def _load_local_jsonl(path: str) -> list[MCQRecord]:
    """Local JSONL rows: {"question", "options" (or "choices"), "answer" (letter or index)}."""
    import json
    from string import ascii_uppercase

    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            options = list(row.get("options") or row["choices"])
            answer = row["answer"]
            gt_idx = ascii_uppercase.index(answer) if isinstance(answer, str) and len(answer) == 1 else int(answer)
            records.append(
                MCQRecord(
                    question=row["question"],
                    options=options,
                    ground_truth_idx=gt_idx,
                    dataset=dataset_slug(path),
                )
            )
    return records


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
    a local JSONL path (rows: question/options/answer), or ANY HuggingFace
    dataset id — for the generic HF case, ``dataset_config``/``split`` and the
    three field names map its schema onto (question, options, ground-truth
    index; the answer field may hold a letter or an index)."""
    import os

    if dataset.endswith(".jsonl") or os.path.exists(dataset):
        return _shuffled_capped(_load_local_jsonl(dataset), n_questions, seed)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("live data source requires the `datasets` package (`uv pip install datasets`)") from e

    rev = revision or BUILTIN_DATASETS.get(dataset, "main")
    records: list[MCQRecord] = []

    if dataset == "mmlu":
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
        from string import ascii_uppercase

        ds = load_dataset(dataset, dataset_config, split=split or "test", revision=rev)
        for row in ds:
            answer = row[answer_field]
            gt_idx = ascii_uppercase.index(answer) if isinstance(answer, str) and len(answer) == 1 else int(answer)
            records.append(
                MCQRecord(
                    question=row[question_field],
                    options=list(row[choices_field]),
                    ground_truth_idx=gt_idx,
                    dataset=dataset_slug(dataset),
                )
            )

    return _shuffled_capped(records, n_questions, seed)
