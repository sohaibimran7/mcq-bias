"""Materialize and load frozen eval datasets.

The frozen dataset is the reproducibility mechanism: the first task run builds
it (live sources + injectors) and writes it to disk; every later run — any
checkpoint, any machine, any day — loads the identical bytes. No network, no
regeneration, no drift. Prompts are stored exactly as the model sees them: the
prompt style is chosen at materialization (injectors build each style
directly) and is part of the file name — loading applies no text
transformation at all.

Matching is a property of the file format: every row carries both the biased
and the unbiased variant of one question (a record an injector can't handle is
never written), and sample id = the question id — so biased and unbiased runs
always pair by id.
"""

import json
from pathlib import Path
from typing import Optional

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from mcq_bias.pipeline.injectors import BiasInjector
from mcq_bias.pipeline.records import MCQRecord, validate_prompt_style


def iter_matched(records: list[MCQRecord], injector: BiasInjector, prompt_style: str = "none"):
    """(record, injection) pairs — the single source of what survives filtering."""
    for record in records:
        injection = injector.inject(record, prompt_style)
        if injection is None:
            continue  # not injectable → excluded from both variants
        yield record, injection


def write_frozen(
    path: str | Path,
    records: list[MCQRecord],
    injector: BiasInjector,
    prompt_style: str = "none",
    n_questions: Optional[int] = None,
) -> int:
    """Materialize matched question pairs as JSONL; returns rows written."""
    validate_prompt_style(prompt_style)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for record, injection in iter_matched(records, injector, prompt_style):
            f.write(
                json.dumps(
                    {
                        "question": record.question,
                        "question_id": record.question_id,
                        "source_dataset": record.dataset,
                        "prompt_style": prompt_style,
                        "unbiased_messages": record.unbiased_messages(prompt_style),
                        "biased_messages": injection.messages,
                        "bias_type": injector.name,
                        "ground_truth": record.ground_truth,
                        "biased_option": injection.biased_option,
                        "biasing_text": injection.biasing_text,
                    }
                )
                + "\n"
            )
            n += 1
            if n_questions is not None and n >= n_questions:
                break
    return n


def row_to_sample(row: dict, variant: str) -> Sample:
    """One frozen row → one Inspect Sample for the requested variant."""
    messages = row["biased_messages"] if variant == "biased" else row["unbiased_messages"]
    bias_type = row["bias_type"]
    multi_turn = bias_type == "are_you_sure" and variant == "biased"

    chat = []
    followups: list[str] = []
    first_user_seen = False
    for msg in messages:
        if msg["role"] == "user":
            if multi_turn and first_user_seen:
                followups.append(msg["content"])  # injected between on-policy generations by the solver
            else:
                chat.append(ChatMessageUser(content=msg["content"]))
                first_user_seen = True
        elif msg["role"] == "assistant" and not multi_turn:
            chat.append(ChatMessageAssistant(content=msg["content"]))

    metadata = {
        "biased_option": row["biased_option"],
        "bias_type": bias_type,
        "source_dataset": row["source_dataset"],
        "variant": variant,
        "prompt_style": row["prompt_style"],
        "biasing_text": row.get("biasing_text", "") if variant == "biased" else "",
    }
    if followups:
        metadata["followup_user_messages"] = followups
    return Sample(id=row["question_id"], input=chat, target=row["ground_truth"], metadata=metadata)


def load_frozen(path: str | Path, variant: str) -> MemoryDataset:
    """Load one variant of a frozen dataset. Both variants come from the same
    rows, so biased/unbiased datasets are matched by construction."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(row_to_sample(json.loads(line), variant))
    return MemoryDataset(samples)


# ── shared unbiased run (one per dataset, not per bias) ──────────────────────


def write_unbiased_frozen(
    path: str | Path,
    records: list[MCQRecord],
    prompt_style: str = "none",
    n_questions: Optional[int] = None,
) -> int:
    """Materialize the shared unbiased dataset: the question pool itself, no
    injectability filtering. Since every bias's set is by default exactly this
    same pool prefix, any biased run over the same (dataset, n, seed) pairs
    against it by sample id. One unbiased file (and one unbiased eval run)
    serves all bias types."""
    validate_prompt_style(prompt_style)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "question": record.question,
                        "question_id": record.question_id,
                        "source_dataset": record.dataset,
                        "prompt_style": prompt_style,
                        "unbiased_messages": record.unbiased_messages(prompt_style),
                        "ground_truth": record.ground_truth,
                    }
                )
                + "\n"
            )
            n += 1
            if n_questions is not None and n >= n_questions:
                break
    return n


def load_unbiased_frozen(path: str | Path) -> MemoryDataset:
    """Load the shared unbiased dataset. Samples carry no bias metadata —
    bias-relative metrics live on the biased runs, whose switch scorer pairs
    them against the unbiased run's log by sample id."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            chat = [ChatMessageUser(content=m["content"]) for m in row["unbiased_messages"] if m["role"] == "user"]
            samples.append(
                Sample(
                    id=row["question_id"],
                    input=chat,
                    target=row["ground_truth"],
                    metadata={
                        "source_dataset": row["source_dataset"],
                        "variant": "unbiased",
                        "prompt_style": row["prompt_style"],
                    },
                )
            )
    return MemoryDataset(samples)
