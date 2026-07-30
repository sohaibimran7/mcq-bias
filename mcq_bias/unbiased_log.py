"""Finding and reading the unbiased baseline log for switch scoring.

Switch metrics compare a biased run against the shared unbiased run, and a
scorer can only read a finished log — so the switch scorer (scorers.py) uses
``wait_for_unbiased_log`` to wait for the unbiased run's log to complete, then
``unbiased_answers`` to read each question's baseline answer out of it.

To add switch scores to a biased log that was evaluated without
``unbiased_log``, re-score it with Inspect's standard command:

    inspect score logs/<biased>.eval --scorer mcq_bias/switch_scorer \\
        -S unbiased_log=logs/ --action append
"""

from pathlib import Path
from typing import Optional

from mcq_bias.scorers import ANSWER_SCORER_NAME


def _log_header(path: str) -> dict:
    """Cheap header read (status/task/model/args) — no samples loaded."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path, header_only=True)
    return {
        "status": log.status,
        "task": log.eval.task or "",
        "model": log.eval.model,
        "task_args": log.eval.task_args or {},
        "metadata": log.eval.metadata or {},
    }


def _candidate_logs(spec: str) -> list[str]:
    import glob as _glob

    p = Path(spec)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*.eval"))
    if any(ch in spec for ch in "*?["):
        return sorted(_glob.glob(spec, recursive=True))
    return [spec] if p.exists() else []


def _matches_unbiased(
    header: dict,
    model: Optional[str],
    dataset: Optional[str],
    prompt_style: Optional[str],
    question_ids_from: Optional[list[str]] = None,
    prompt_family: str = "chua",
    source_identity_digest: Optional[str] = None,
) -> bool:
    if header["status"] != "success":
        return False
    if "unbiased" not in header["task"].replace("-", "_"):
        return False
    # Compare the exact user-facing dataset id/path. Lossy filename slugs can
    # collide (different local paths with the same basename, or HF ids whose
    # punctuation sanitizes identically).
    if dataset and header["task_args"].get("dataset", "mmlu") != dataset:
        return False
    if prompt_style and header["task_args"].get("prompt_style", "none") != prompt_style:
        return False
    if header["task_args"].get("prompt_family", "chua") != prompt_family:
        return False
    if (
        source_identity_digest is not None
        and header.get("metadata", {}).get("source_identity_digest") != source_identity_digest
    ):
        return False
    # Strict in both directions (unlike the filters above): a question_ids_from
    # run holds a different question set, so it must pair only with an
    # identically-restricted unbiased log — and an unrestricted run must never
    # pair with a restricted one.
    if sorted(header["task_args"].get("question_ids_from") or []) != sorted(question_ids_from or []):
        return False
    if model and header["model"] != model:
        return False
    return True


async def wait_for_unbiased_log(
    unbiased_log: str,
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    prompt_style: Optional[str] = None,
    question_ids_from: Optional[list[str]] = None,
    prompt_family: str = "chua",
    source_identity_digest: Optional[str] = None,
    timeout: float = 3600.0,
    poll_interval: float = 10.0,
) -> str:
    """Wait for a completed unbiased log.

    ``unbiased_log`` may be an exact .eval path (waits for the file to exist
    and its status to be "success"; no further matching, since the path was
    explicit), or a directory / glob to watch — then the newest completed log
    whose task is the unbiased task and whose
    model/dataset/prompt_style/question_ids_from match is chosen, so a biased
    run for checkpoint A can never pair with checkpoint B's unbiased run (nor
    an encourage_cot run with a ``none``-style one, nor a restricted pool with
    an unrestricted one).

    Lets biased and unbiased evals launch in parallel: generation proceeds
    immediately; only switch scoring waits here until the unbiased run
    finishes. Raises TimeoutError (failing the scores visibly) if nothing
    appears in time.
    """
    import asyncio
    import time

    exact = not Path(unbiased_log).is_dir() and not any(ch in unbiased_log for ch in "*?[")
    deadline = time.monotonic() + timeout
    while True:
        for path in reversed(_candidate_logs(unbiased_log)):  # newest first (names start with timestamps)
            try:
                header = _log_header(path)
            except Exception:
                continue  # partial file / still being written
            if exact:
                if header["status"] == "success":
                    return path
            elif _matches_unbiased(
                header,
                model,
                dataset,
                prompt_style,
                question_ids_from,
                prompt_family,
                source_identity_digest,
            ):
                return path
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No completed unbiased log found for {unbiased_log!r} (model={model!r}, "
                f"dataset={dataset!r}) within {timeout:.0f}s — was the unbiased eval started? "
                "Run: inspect eval mcq_bias/tasks.py@mcq_bias_unbiased"
            )
        await asyncio.sleep(poll_interval)


def unbiased_answers(path: str) -> dict[str, Optional[str]]:
    """Sample id -> parsed answer from a completed unbiased run's .eval log
    (used by scorers.switch_scorer)."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    answers: dict[str, Optional[str]] = {}
    for sample in log.samples or []:
        score = (sample.scores or {}).get(ANSWER_SCORER_NAME)
        answers[str(sample.id)] = score.answer if score else None
    return answers
