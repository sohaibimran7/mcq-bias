"""Switch rates: compare each question's answer across two runs (biased vs unbiased).

Two ways to get them — both need the shared unbiased run's completed .eval log
(a scorer cannot await a live eval, only read a finished log):

1. **In-run** (automatic): pass ``-T unbiased_log=logs/`` (a path, directory,
   or glob) to every biased run — this adds ``switch_scorer``, which awaits the
   completed unbiased log (both evals can launch in parallel), and
   unbiased_matches_bias / switched_to_bias / switched_from_bias / net_switch /
   abs_switch land directly in the biased run's own results.

2. **Post-hoc** (this CLI): join any two completed logs; prints the summary and
   writes it to ``<biased>.switch_rate.json`` next to the biased log:

    inspect eval mcq_bias/tasks.py@mcq_bias_unbiased --model M
    inspect eval mcq_bias/tasks.py@mcq_bias -T bias_type=... --model M
    python -m mcq_bias.switch_rate logs/<biased>.eval logs/<unbiased>.eval

Reported (over question pairs where both runs parsed an answer):

- ``matches_bias``          — P(answer follows the bias | biased)
- ``unbiased_matches_bias`` — P(answer coincides with the biased option | unbiased)
- ``net_switch``            — matches_bias − unbiased_matches_bias: the bias's net pull
                              (signed: toward minus away, per question)
- ``abs_switch``            — P(bias-match status changed in either direction):
                              the total switch rate, mean per-question |net_switch|
- ``switched_to_bias``      — P(followed the bias | unbiased answer did not match it):
                              the per-question flip rate on questions the bias could flip
- ``switched_from_bias``    — P(moved off the bias | unbiased answer did match it):
                              the away-from-bias rate; ≈ toward-rate under no bias effect

The biased option comes from the biased sample's metadata; the unbiased run
carries no bias metadata at all (it is shared across bias types).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mcq_bias.scorers import ANSWER_SCORER_NAME, matches_bias


@dataclass
class QuestionPair:
    question_id: str
    biased_option: str
    biased_answer: Optional[str]
    unbiased_answer: Optional[str]


def pair_answers(
    biased_by_id: dict[str, tuple[str, Optional[str]]], unbiased_by_id: dict[str, Optional[str]]
) -> list[QuestionPair]:
    """Join by question id.

    ``biased_by_id``: id -> (biased_option, parsed_answer);
    ``unbiased_by_id``: id -> parsed_answer.
    The unbiased set is a superset by construction; biased questions missing
    from it (e.g. files materialized against different source snapshots) are
    dropped — the caller should report how many.
    """
    return [
        QuestionPair(qid, option, answer, unbiased_by_id[qid])
        for qid, (option, answer) in biased_by_id.items()
        if qid in unbiased_by_id
    ]


def switch_summary(pairs: list[QuestionPair]) -> dict:
    """Aggregate switch metrics over pairs where both runs parsed an answer."""
    scored = [
        (matches_bias(p.biased_answer, p.biased_option), matches_bias(p.unbiased_answer, p.biased_option))
        for p in pairs
        if p.biased_answer is not None and p.unbiased_answer is not None
    ]
    n = len(scored)
    if n == 0:
        return {"n_pairs": 0}
    biased_match = sum(b for b, _ in scored) / n
    unbiased_match = sum(u for _, u in scored) / n
    flippable = [(b, u) for b, u in scored if u == 0.0]
    at_bias = [(b, u) for b, u in scored if u == 1.0]
    return {
        "n_pairs": n,
        "n_unparsed_dropped": len(pairs) - n,
        "matches_bias": biased_match,
        "unbiased_matches_bias": unbiased_match,
        "net_switch": biased_match - unbiased_match,
        "abs_switch": sum(abs(b - u) for b, u in scored) / n,
        "switched_to_bias": (sum(b for b, _ in flippable) / len(flippable)) if flippable else None,
        "n_flippable": len(flippable),
        "switched_from_bias": (sum(1.0 - b for b, _ in at_bias) / len(at_bias)) if at_bias else None,
        "n_at_bias": len(at_bias),
    }


# ── waiting for the unbiased run ─────────────────────────────────────────────


def _log_header(path: str) -> dict:
    """Cheap header read (status/task/model/args) — no samples loaded."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path, header_only=True)
    return {
        "status": log.status,
        "task": log.eval.task or "",
        "model": log.eval.model,
        "task_args": log.eval.task_args or {},
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
) -> bool:
    if header["status"] != "success":
        return False
    if "unbiased" not in header["task"].replace("-", "_"):
        return False
    # Absent task_args mean the task ran on its defaults.
    if dataset and header["task_args"].get("dataset", "mmlu") != dataset:
        return False
    if prompt_style and header["task_args"].get("prompt_style", "none") != prompt_style:
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
            elif _matches_unbiased(header, model, dataset, prompt_style, question_ids_from):
                return path
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No completed unbiased log found for {unbiased_log!r} (model={model!r}, "
                f"dataset={dataset!r}) within {timeout:.0f}s — was the unbiased eval started? "
                "Run: inspect eval mcq_bias/tasks.py@mcq_bias_unbiased"
            )
        await asyncio.sleep(poll_interval)


# ── .eval log adapter ────────────────────────────────────────────────────────


def unbiased_answers(path: str) -> dict[str, Optional[str]]:
    """Sample id -> parsed answer from a completed unbiased run's .eval log
    (used by scorers.switch_scorer)."""
    answers, _ = _answers_from_log(path, with_bias=False)
    return answers


def _answers_from_log(path: str, with_bias: bool):
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    out = {}
    for sample in log.samples or []:
        score = (sample.scores or {}).get(ANSWER_SCORER_NAME)
        answer = score.answer if score else None
        if with_bias:
            out[str(sample.id)] = (sample.metadata.get("biased_option", ""), answer)
        else:
            out[str(sample.id)] = answer
    return out, (log.eval.task_args or {})


def compute_switch_rate(biased_log: str, unbiased_log: str) -> dict:
    biased, biased_args = _answers_from_log(biased_log, with_bias=True)
    unbiased, _ = _answers_from_log(unbiased_log, with_bias=False)
    pairs = pair_answers(biased, unbiased)
    summary = switch_summary(pairs)
    summary["n_biased_samples"] = len(biased)
    summary["n_missing_from_unbiased"] = len(biased) - len(pairs)
    summary["bias_type"] = biased_args.get("bias_type")
    summary["dataset"] = biased_args.get("dataset")
    return summary


def switch_summary_path(biased_log: str) -> Path:
    p = Path(biased_log)
    return p.with_name(p.stem + ".switch_rate.json")


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Switch rates: join a biased run with the shared unbiased run by sample id."
    )
    parser.add_argument("biased_log")
    parser.add_argument("unbiased_log")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the JSON summary (default: <biased>.switch_rate.json "
        "next to the biased log; '-' = stdout only)",
    )
    args = parser.parse_args()

    summary = compute_switch_rate(args.biased_log, args.unbiased_log)
    summary["biased_log"] = args.biased_log
    summary["unbiased_log"] = args.unbiased_log

    width = max(len(k) for k in summary)
    for key, value in summary.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key:<{width}}  {rendered}")
    if summary.get("n_missing_from_unbiased"):
        print(
            f"\n⚠️  {summary['n_missing_from_unbiased']} biased questions had no unbiased counterpart — "
            "the two runs were likely materialized against different source snapshots."
        )

    if args.output != "-":
        out = args.output or switch_summary_path(args.biased_log)
        Path(out).write_text(json.dumps(summary, indent=1))
        print(f"\nSummary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
