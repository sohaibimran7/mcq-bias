"""Publishable Inspect tasks for the sycophancy-bias MCQ eval.

Stable ``@task`` entry points for inspect_evals external registration. The
package is self-contained: pipeline, solver, scorers, and parsers all live
under mcq_bias/ (see README.md). No data ships with the package.

Reproducibility model: the first run for a given (dataset, bias_type,
n_questions, seed) builds the dataset (public HF sources at pinned revisions +
deterministic bias injection) and freezes it as JSONL in the data directory
(``$MCQ_BIAS_DATA_DIR`` or ``~/.cache/mcq_bias``, override per-task with
``dataset_dir``); every later run — any checkpoint, any machine — loads that
identical file. Share the frozen file to evaluate on the exact question set.
Both variants live in each row, so the biased task and its unbiased
counterpart are always matched, paired by sample id.

Usage:
    inspect eval mcq_bias/tasks.py@mcq_bias \\
        -T bias_type=wrong_few_shot -T dataset=truthfulqa --model openai/gpt-4o-mini

    # paired unbiased run (switch rates pair by sample id):
    inspect eval mcq_bias/tasks.py@mcq_bias_unbiased \\
        -T dataset=truthfulqa --model openai/gpt-4o-mini

``n_questions`` is the only parameter that controls dataset size: it fixes the
question pool (the seed-shuffled prefix of the source dataset, which is also
the unbiased set), and by default every task evaluates exactly that many
matched questions — materialization raises on a shortfall instead of freezing
a short file. For biases whose injection can fail on individual questions
(wrong_argument generation), ``min_n_questions`` sets a lower floor: matched
counts in [min, n] freeze with a warning, still as a subset of the same pool,
so pairing against the shared unbiased run is unaffected. For small runs,
lower n_questions — the smaller pool is a prefix of the larger one, so a small
biased run can even pair against an existing larger unbiased log.
"""

from pathlib import Path
from typing import Optional

from inspect_ai import Task, task

BIAS_TYPES = [
    "suggested_answer",
    "wrong_argument",
    "distractor_fact",
    "wrong_few_shot",
    "post_hoc",
    "are_you_sure",
    "spurious_few_shot_squares",
]
SOURCE_DATASETS = ["mmlu", "truthfulqa", "logiqa", "hellaswag"]


def frozen_path(
    dataset: str,
    bias_type: str,
    prompt_style: str,
    n_questions: int,
    seed: str,
    dataset_dir: Optional[str | Path] = None,
    argument_model: Optional[str] = None,
    ids_slug: Optional[str] = None,
) -> Path:
    """The frozen dataset file for one task parameterization (the cache key).

    ``prompt_style`` is part of the file name because the file stores the
    exact prompts, built directly in that style (loading applies no text
    transformation). For wrong_argument the argument model is part of the name
    too (``..._args-<model-slug>.jsonl``): the same questions with arguments
    written by different models are different datasets. So is a
    ``question_ids_from`` restriction (``..._ids-<hash>``), since a restricted
    pool selects different questions."""
    from mcq_bias.paths import generated_dir
    from mcq_bias.pipeline.sources import dataset_slug

    directory = Path(dataset_dir) if dataset_dir else generated_dir()
    name = f"{dataset_slug(dataset)}_{bias_type}_{prompt_style}_n{n_questions}_seed{seed}"
    if bias_type == "wrong_argument":
        from mcq_bias.pipeline.wrong_arguments import model_slug

        name += f"_args-{model_slug(argument_model)}"
    if ids_slug:
        name += f"_ids-{ids_slug}"
    return directory / f"{name}.jsonl"


def unbiased_frozen_path(
    dataset: str,
    prompt_style: str,
    n_questions: int,
    seed: str,
    dataset_dir: Optional[str | Path] = None,
    ids_slug: Optional[str] = None,
) -> Path:
    """The shared unbiased file: one per (dataset, prompt_style, n, seed), serving all bias types."""
    from mcq_bias.paths import generated_dir
    from mcq_bias.pipeline.sources import dataset_slug

    directory = Path(dataset_dir) if dataset_dir else generated_dir()
    name = f"{dataset_slug(dataset)}_unbiased_{prompt_style}_n{n_questions}_seed{seed}"
    if ids_slug:
        name += f"_ids-{ids_slug}"
    return directory / f"{name}.jsonl"


def _question_id_allowlist(question_ids_from: list[str]) -> tuple[set, str, dict]:
    """Resolve ``question_ids_from`` files → (allowed ids, filename slug, per-file counts).

    The allowed set is the intersection of the files' unique question ids; the
    slug hashes the set itself, so the same restriction always maps to the
    same frozen file."""
    import hashlib

    from mcq_bias.pipeline.sources import read_question_ids

    per_file = {str(p): read_question_ids(p) for p in question_ids_from}
    allowed = set.intersection(*per_file.values())
    counts = {Path(p).name: len(ids) for p, ids in per_file.items()}
    if not allowed:
        raise ValueError(
            "question_ids_from files share no question ids — their intersection is empty "
            f"({'; '.join(f'{name}: {n} unique ids' for name, n in counts.items())})."
        )
    slug = hashlib.sha1("\n".join(sorted(allowed)).encode()).hexdigest()[:10]
    return allowed, slug, counts


def _load_pool(
    dataset: str,
    n_questions: int,
    seed: str,
    source_kwargs: Optional[dict] = None,
    allowlist: Optional[tuple[set, str, dict]] = None,
):
    """The task's question pool: seed-shuffled prefix of the source, optionally
    restricted to an allowed question-id set. Returns exactly n_questions
    records or raises ValueError — the single place pool size is enforced."""
    from mcq_bias.pipeline.sources import load_records

    if allowlist is None:
        records = load_records(dataset, n_questions=n_questions, seed=seed, **(source_kwargs or {}))
        if len(records) < n_questions:
            raise ValueError(
                f"{dataset!r} has only {len(records)} questions — lower n_questions "
                f"(requested {n_questions}; the task evaluates exactly that many, "
                "so the request must be satisfiable in full)."
            )
        return records
    allowed, _, counts = allowlist
    pool = load_records(dataset, n_questions=None, seed=seed, **(source_kwargs or {}))
    in_pool = [r for r in pool if r.question_id in allowed]
    if len(in_pool) < n_questions:
        raise ValueError(
            f"question_ids_from allows only {len(in_pool)} of the requested {n_questions} questions "
            f"in the {len(pool)}-question {dataset!r} pool. Unique question ids per file: "
            f"{'; '.join(f'{name}: {n}' for name, n in counts.items())}; intersection: {len(allowed)}. "
            "Lower n_questions or pass id files with broader coverage of this dataset."
        )
    return in_pool[:n_questions]


def materialize(
    path: Path,
    dataset: str,
    bias_type: str,
    prompt_style: str,
    n_questions: int,
    seed: str,
    argument_model: Optional[str] = None,
    generate_missing_arguments: bool = False,
    unbiased_path: Optional[Path] = None,
    source_kwargs: Optional[dict] = None,
    question_ids_from: Optional[list[str]] = None,
    min_n_questions: Optional[int] = None,
) -> int:
    """Build and freeze the dataset (live sources + injectors); returns rows written.

    ``n_questions`` fixes the question pool (the seed-shuffled prefix of the
    source — also the unbiased set) and, by default, is exact: the frozen file
    contains exactly that many matched question pairs, or this raises (a short
    file is never frozen). ``min_n_questions`` relaxes the floor for biases
    whose injection can fail on individual questions (wrong_argument: some
    questions reliably yield no accepted argument, and shrinking n_questions
    cannot exclude them — they stay inside the smaller prefix): matched pairs
    in [min, n] freeze with a warning; below min still raises. The matched set
    is always a subset of the pool prefix, so pairing against the shared
    unbiased run (the full prefix) is unaffected. Also materializes that
    shared unbiased file (``unbiased_path``) from the same source snapshot if
    it doesn't exist yet. ``question_ids_from`` restricts the pool to question
    ids present in all of the given JSONL files."""
    from mcq_bias.pipeline.build import write_frozen, write_unbiased_frozen
    from mcq_bias.pipeline.injectors import default_injectors

    allowlist = _question_id_allowlist(question_ids_from) if question_ids_from else None
    records = _load_pool(dataset, n_questions, seed, source_kwargs, allowlist)
    if unbiased_path is not None and not unbiased_path.exists():
        n_unbiased = write_unbiased_frozen(unbiased_path, records, prompt_style, n_questions=n_questions)
        print(f"Materialized shared unbiased set ({n_unbiased} questions) → {unbiased_path}")
    argument_store = None
    if bias_type == "wrong_argument":
        from mcq_bias.pipeline.wrong_arguments import (
            WrongArgumentStore,
            arguments_path,
            generate_wrong_arguments_sync,
        )

        argument_store = WrongArgumentStore.for_model(argument_model)  # the single source of truth
        if generate_missing_arguments:
            n_generated = generate_wrong_arguments_sync(
                records, argument_model, argument_store, store_path=arguments_path(argument_model)
            )
            print(
                f"wrong_argument: generated {n_generated} arguments with "
                f"{argument_model} → {arguments_path(argument_model)}"
            )
    injector = default_injectors(records, wrong_arguments=argument_store)[bias_type]

    floor = n_questions if min_n_questions is None else min_n_questions
    n = write_frozen(path, records, injector, prompt_style, n_questions=n_questions)
    if n < floor:
        path.unlink(missing_ok=True)  # never freeze a dataset below the floor
        raise ValueError(
            f"Only {n}/{n_questions} matched questions for {bias_type!r} on {dataset!r} "
            f"(floor: {floor}). Remedies: for wrong_argument pass generate_missing_arguments=True "
            "(fills the argument store using the original recipe), restrict the pool to covered "
            "questions with question_ids_from=[<store file>], or tolerate per-question failures "
            "with min_n_questions."
        )
    if n < n_questions:
        print(
            f"⚠️  {bias_type} on {dataset}: matched {n}/{n_questions} questions "
            f"(≥ min_n_questions={floor}) — {n_questions - n} dropped (no accepted argument). "
            "The matched set is a subset of the pool prefix, so pairing against the shared "
            "unbiased run is unaffected."
        )
    print(
        f"Materialized {n} question pairs → {path}\n"
        "Commit or share this file to evaluate other checkpoints/machines on the identical set."
    )
    return n


def task_from_frozen(
    path: str | Path,
    metadata: Optional[dict] = None,
    unbiased_log: Optional[str] = None,
    grader_model: Optional[str] = None,
    question_ids_from: Optional[list[str]] = None,
) -> Task:
    """Assemble the biased eval Task over a frozen dataset (offline; no generation
    and no text transformation — the file stores the exact prompts).

    ``unbiased_log``: path/dir/glob of the shared unbiased run's .eval log —
    adds the switch_scorer so switch metrics land directly in this run's results.
    ``grader_model``: model for the bias_acknowledged grader (default: the
    scorer's DEFAULT_GRADER_MODEL)."""
    from mcq_bias.pipeline.build import load_frozen
    from mcq_bias.scorers import (
        bias_acknowledged_scorer,
        mcq_bias_scorer,
        options_considered_scorer,
        switch_scorer,
    )
    from mcq_bias.solver import multi_turn_generate

    scorers = [mcq_bias_scorer(), options_considered_scorer(), bias_acknowledged_scorer(grader_model)]
    if unbiased_log:
        # question_ids_from disambiguates the watched dir: a restricted run must
        # pair only with an identically-restricted unbiased log (same id set).
        scorers.append(switch_scorer(unbiased_log, question_ids_from=question_ids_from))
    return Task(
        dataset=load_frozen(path, "biased"),
        solver=[multi_turn_generate()],
        scorer=scorers,
        metadata=metadata,
    )


def unbiased_task_from_frozen(path: str | Path, metadata: Optional[dict] = None) -> Task:
    """Assemble the shared unbiased Task: plain questions, no bias-aware scorers
    (matches_bias needs a bias; the acknowledgement grader has nothing to grade).
    Bias-relative metrics come from joining with a biased run — see switch_rate.py."""
    from mcq_bias.pipeline.build import load_unbiased_frozen
    from mcq_bias.scorers import mcq_bias_scorer, options_considered_scorer
    from mcq_bias.solver import multi_turn_generate

    return Task(
        dataset=load_unbiased_frozen(path),
        solver=[multi_turn_generate()],
        scorer=[mcq_bias_scorer(), options_considered_scorer()],
        metadata=metadata,
    )


def _source_kwargs(dataset_config, split, question_field, choices_field, answer_field) -> dict:
    out = {}
    if dataset_config:
        out["dataset_config"] = dataset_config
    if split:
        out["split"] = split
    if question_field != "question":
        out["question_field"] = question_field
    if choices_field != "choices":
        out["choices_field"] = choices_field
    if answer_field != "answer":
        out["answer_field"] = answer_field
    return out


def _biased_task(
    bias_type: str,
    dataset: str,
    prompt_style: str,
    n_questions: int,
    seed: str,
    argument_model: str,
    generate_missing_arguments: bool,
    dataset_dir: Optional[str],
    source_kwargs: Optional[dict] = None,
    unbiased_log: Optional[str] = None,
    grader_model: Optional[str] = None,
    question_ids_from: Optional[list[str]] = None,
    min_n_questions: Optional[int] = None,
) -> Task:
    from mcq_bias.pipeline.records import validate_prompt_style

    if bias_type not in BIAS_TYPES:
        raise ValueError(f"Unknown bias_type: {bias_type!r}. Known: {BIAS_TYPES}")
    if generate_missing_arguments and bias_type != "wrong_argument":
        raise ValueError("generate_missing_arguments only applies to bias_type='wrong_argument'")
    if min_n_questions is not None and not 1 <= min_n_questions <= n_questions:
        raise ValueError(f"min_n_questions must be in [1, n_questions]; got {min_n_questions} with n={n_questions}")
    validate_prompt_style(prompt_style)
    ids_slug = _question_id_allowlist(question_ids_from)[1] if question_ids_from else None
    path = frozen_path(dataset, bias_type, prompt_style, n_questions, seed, dataset_dir, argument_model, ids_slug)
    unbiased_path = unbiased_frozen_path(dataset, prompt_style, n_questions, seed, dataset_dir, ids_slug)
    if not path.exists():
        materialize(
            path,
            dataset,
            bias_type,
            prompt_style,
            n_questions,
            seed,
            argument_model,
            generate_missing_arguments,
            unbiased_path=unbiased_path,
            source_kwargs=source_kwargs,
            question_ids_from=question_ids_from,
            min_n_questions=min_n_questions,
        )
    else:
        # A tolerant earlier run may have frozen fewer than n_questions matched
        # pairs; a stricter run must not silently evaluate the smaller set.
        floor = n_questions if min_n_questions is None else min_n_questions
        n_frozen = sum(1 for _ in open(path))
        if n_frozen < floor:
            raise ValueError(
                f"{path} holds {n_frozen} matched pairs, below the requested floor of {floor} "
                f"(n_questions={n_questions}, min_n_questions={min_n_questions}). It was frozen by a "
                "more tolerant run. Lower min_n_questions to accept it, or move the file aside "
                "(e.g. to an _archive/) to re-materialize."
            )
    return task_from_frozen(
        path,
        unbiased_log=unbiased_log,
        grader_model=grader_model,
        question_ids_from=question_ids_from,
        metadata={
            "bias_type": bias_type,
            "source_dataset": dataset,
            "dataset_file": str(path),
            "unbiased_dataset_file": str(unbiased_path),
            **({"unbiased_log": unbiased_log} if unbiased_log else {}),
            **({"question_ids_from": question_ids_from} if question_ids_from else {}),
        },
    )


def _unbiased_task(
    dataset: str,
    prompt_style: str,
    n_questions: int,
    seed: str,
    dataset_dir: Optional[str],
    source_kwargs: Optional[dict] = None,
    question_ids_from: Optional[list[str]] = None,
) -> Task:
    from mcq_bias.pipeline.records import validate_prompt_style

    validate_prompt_style(prompt_style)
    allowlist = _question_id_allowlist(question_ids_from) if question_ids_from else None
    path = unbiased_frozen_path(
        dataset, prompt_style, n_questions, seed, dataset_dir, allowlist[1] if allowlist else None
    )
    if not path.exists():
        from mcq_bias.pipeline.build import write_unbiased_frozen

        records = _load_pool(dataset, n_questions, seed, source_kwargs, allowlist)
        n = write_unbiased_frozen(path, records, prompt_style, n_questions=n_questions)
        print(f"Materialized shared unbiased set ({n} questions) → {path}")
    return unbiased_task_from_frozen(
        path,
        metadata={
            "source_dataset": dataset,
            "dataset_file": str(path),
        },
    )


@task
def mcq_bias(
    bias_type: str = "suggested_answer",
    dataset: str = "mmlu",
    prompt_style: str = "none",
    n_questions: int = 250,
    min_n_questions: Optional[int] = None,
    seed: str = "42",
    argument_model: Optional[str] = None,
    generate_missing_arguments: bool = False,
    question_ids_from: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    unbiased_log: Optional[str] = None,
    grader_model: Optional[str] = None,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
) -> Task:
    """MCQ accuracy under an injected bias (biased variant).

    The dataset is materialized once per parameterization into the data
    directory (``$MCQ_BIAS_DATA_DIR`` or ``~/.cache/mcq_bias``, or
    ``dataset_dir``) and loaded from that frozen file on every subsequent run,
    so many checkpoints can be evaluated on the identical question set by
    default. Materializing also freezes the shared unbiased file (one per
    dataset — see ``mcq_bias_unbiased``) from the same source snapshot. The
    biased option is a fixed wrong option chosen pseudo-randomly with the
    question text as seed, so bias-following is separable from correctness.
    ``seed`` is a string on purpose: it seeds ``random.Random`` exactly as the
    original cot-transparency pipeline did (``"42"`` ≠ ``42``).

    ``prompt_style``: ``none`` (default) adds no reasoning instructions —
    prompts carry only the answer-format line, for models that reason in their
    own reasoning channel; ``encourage_cot`` uses the exact step-by-step
    instructions from the original cot-transparency prompts. Each style is
    built directly and frozen as its own file (the style is part of the file
    name).

    ``dataset`` may be a built-in alias (mmlu/truthfulqa/logiqa/hellaswag), a
    local JSONL path (rows: question/options/answer), or any HF dataset id —
    use ``dataset_config``/``split`` and the ``*_field`` names to map its schema.

    For wrong_argument, ``argument_model`` names whose LLM-written wrong
    arguments to use (default: the released Gemma-4 set) and is part of the
    frozen file's name; a question without a stored argument fails the
    n_questions guarantee unless ``generate_missing_arguments=True`` (fills
    them by querying the model once) or the pool is restricted to covered
    questions via ``question_ids_from``.

    ``min_n_questions`` (default: ``n_questions``, i.e. exact-or-error) relaxes
    the floor for biases whose injection can fail on individual questions:
    generation can reliably fail for some questions, and shrinking n_questions
    cannot exclude them (they stay inside the smaller prefix). With a floor
    set, matched counts in [min, n] freeze with a warning; the matched set is
    a subset of the pool prefix, so pairing against the shared unbiased run
    still works (unaffected sample ids simply have no biased counterpart).

    ``question_ids_from`` (optional, off by default): a list of JSONL paths;
    the pool is restricted to question ids present in all of them (any file
    whose rows carry ``question_id`` works — e.g. a wrong-argument store, to
    evaluate exactly the questions it covers with no generation). The
    restriction is part of the frozen file's name (``_ids-<hash>``).

    Headline metrics: accuracy, ``matches_bias`` (answer == the biased option),
    and ``bias_acknowledged`` (model-graded: does the response reference the
    biasing text?). Switch rates: pass ``unbiased_log=<path-or-logs-dir>`` —
    the switch_scorer awaits the completed unbiased run's log (so both evals
    can launch in parallel) and writes switched_to_bias / switched_from_bias /
    net_switch / abs_switch / unbiased_matches_bias into this run's results.
    Or join post-hoc with
    ``python -m mcq_bias.switch_rate <biased.eval> <unbiased.eval>``.
    """
    return _biased_task(
        bias_type,
        dataset,
        prompt_style,
        n_questions,
        seed,
        _argument_model(argument_model),
        generate_missing_arguments,
        dataset_dir,
        _source_kwargs(dataset_config, split, question_field, choices_field, answer_field),
        unbiased_log=unbiased_log,
        grader_model=grader_model,
        question_ids_from=question_ids_from,
        min_n_questions=min_n_questions,
    )


@task
def mcq_bias_unbiased(
    dataset: str = "mmlu",
    prompt_style: str = "none",
    n_questions: int = 250,
    seed: str = "42",
    question_ids_from: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
) -> Task:
    """The shared unbiased run — one per (dataset, prompt_style, n_questions,
    seed), serving all bias types (unbiased prompts are bias-independent, and
    every bias's matched set is drawn from this same pool prefix — identical
    to it by default, a subset of it when a biased run used min_n_questions).
    Pass the same ``question_ids_from`` as the biased runs, if any — it
    selects the pool. There is no min_n_questions here: the unbiased set never
    shrinks (no injection can fail), so it always covers every biased run.

    Samples carry no bias metadata; scorers report accuracy and parse rate only.
    Bias-relative metrics (switch rates) come from joining a biased run against
    this one by sample id: ``python -m mcq_bias.switch_rate``.
    """
    return _unbiased_task(
        dataset,
        prompt_style,
        n_questions,
        seed,
        dataset_dir,
        _source_kwargs(dataset_config, split, question_field, choices_field, answer_field),
        question_ids_from=question_ids_from,
    )


def _argument_model(value: Optional[str]) -> str:
    from mcq_bias.pipeline.wrong_arguments import DEFAULT_ARGUMENT_MODEL

    return value or DEFAULT_ARGUMENT_MODEL


def suite_tasks(
    bias_types: list[str],
    datasets: list[str],
    prompt_style: str = "none",
    n_questions: int = 250,
    min_n_questions: Optional[int] = None,
    seed: str = "42",
    variants: tuple[str, ...] = ("biased", "unbiased"),
    unbiased_log: Optional[str] = None,
    argument_model: Optional[str] = None,
    generate_missing_arguments: bool = False,
    question_ids_from: Optional[list[str]] = None,
    dataset_dir: Optional[str] = None,
    grader_model: Optional[str] = None,
    skip_unbuildable: bool = False,
) -> list[Task]:
    """The full suite: per-bias biased tasks + one shared unbiased task per dataset.

    Unbiased tasks come first so a sequential runner (``inspect_ai.eval``)
    completes them before any biased task's switch scorer starts polling.
    Pass ``unbiased_log`` (typically the run's log dir) to add switch scoring
    to every biased task. Built via the ``@task`` entry points so eval logs
    carry proper task names and task_args (the directory watcher matches on
    them).
    """
    tasks = []
    if "unbiased" in variants:
        for dataset in datasets:
            tasks.append(
                mcq_bias_unbiased(
                    dataset=dataset,
                    prompt_style=prompt_style,
                    n_questions=n_questions,
                    seed=seed,
                    question_ids_from=question_ids_from,
                    dataset_dir=dataset_dir,
                )
            )
    if "biased" in variants:
        for dataset in datasets:
            for bias_type in bias_types:
                try:
                    tasks.append(
                        mcq_bias(
                            bias_type=bias_type,
                            dataset=dataset,
                            prompt_style=prompt_style,
                            n_questions=n_questions,
                            min_n_questions=min_n_questions,
                            seed=seed,
                            argument_model=argument_model,
                            # only meaningful (and only accepted) for wrong_argument
                            generate_missing_arguments=(generate_missing_arguments and bias_type == "wrong_argument"),
                            question_ids_from=question_ids_from,
                            dataset_dir=dataset_dir,
                            unbiased_log=unbiased_log,
                            grader_model=grader_model,
                        )
                    )
                except ValueError as err:
                    if not skip_unbuildable:
                        raise
                    print(f"⚠️  SKIPPING {bias_type} on {dataset}: {err}")
    return tasks
