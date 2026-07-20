"""Run the eval suite from the command line:

    python -m mcq_bias --model openai/gpt-4o-mini
    python -m mcq_bias --model M --bias-types suggested_answer wrong_few_shot \\
        --datasets mmlu truthfulqa --prompt-style encourage_cot --n-questions 100

Builds the per-bias biased tasks plus ONE shared unbiased task per dataset and
runs them with ``inspect_ai.eval``. Unbiased tasks are scheduled first, and the
biased tasks' switch scorer watches the log directory — so switch metrics
(towards_bias_switch / away_from_bias_switch / net_switch / abs_switch /
unbiased_matches_bias) land in each biased run's own log within a single
invocation.

For a single task with full ``-T`` control, use inspect directly:

    inspect eval mcq_bias/tasks.py@mcq_bias -T bias_type=... --model M
"""

import argparse
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    from mcq_bias.pipeline.records import PROMPT_STYLES
    from mcq_bias.tasks import BIAS_TYPES, SOURCE_DATASETS, suite_tasks

    parser = argparse.ArgumentParser(
        prog="python -m mcq_bias",
        description="Run the MCQ bias eval suite: biased tasks per bias type + one shared unbiased run per dataset.",
    )
    parser.add_argument("--model", nargs="+", required=True, help="Inspect model id(s) to evaluate")
    parser.add_argument(
        "--bias-types",
        nargs="+",
        default=BIAS_TYPES,
        choices=BIAS_TYPES,
        metavar="BIAS",
        help=f"bias types to run (default: all — {', '.join(BIAS_TYPES)})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mmlu"],
        metavar="DATASET",
        help=f"built-in alias ({', '.join(SOURCE_DATASETS)}), local JSONL path, or HF dataset id (default: mmlu)",
    )
    parser.add_argument("--prompt-style", default="none", choices=list(PROMPT_STYLES))
    parser.add_argument(
        "--n-questions",
        type=int,
        default=250,
        help="matched questions per task — a hard guarantee by default, and the ONE sizing knob "
        "(biases that cannot reach it are skipped with a warning)",
    )
    parser.add_argument(
        "--min-n-questions",
        type=int,
        default=None,
        help="floor for biases whose injection can fail per-question (wrong_argument generation): "
        "accept [min, n] matched questions with a warning instead of skipping "
        "(default: --n-questions, i.e. exact)",
    )
    parser.add_argument("--seed", default="42", help="question-shuffle seed (a string; part of the frozen file name)")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["biased", "unbiased"],
        choices=["biased", "unbiased"],
        help="run only biased or only unbiased tasks (default: both)",
    )
    parser.add_argument("--argument-model", default=None, help="whose wrong arguments to use for wrong_argument")
    parser.add_argument(
        "--generate-missing-arguments",
        action="store_true",
        help="for wrong_argument: query --argument-model to fill store gaps before freezing "
        "(otherwise the bias is skipped when the store can't cover n_questions)",
    )
    parser.add_argument(
        "--question-ids-from",
        nargs="+",
        default=None,
        metavar="PATH",
        help="restrict the question pool to ids present in ALL of these JSONL files "
        "(any file with question_id fields, e.g. a wrong-argument store — evaluates "
        "exactly the questions it covers, no generation needed)",
    )
    parser.add_argument("--grader-model", default=None, help="model for the bias_acknowledged grader")
    parser.add_argument(
        "--no-bias-acknowledged",
        action="store_true",
        help="omit the model-graded bias_acknowledged scorer when the experiment does not report it",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="where frozen datasets live (default: $MCQ_BIAS_DATA_DIR or ~/.cache/mcq_bias)",
    )
    parser.add_argument("--log-dir", default="logs", help="inspect log directory (default: ./logs)")
    parser.add_argument(
        "--unbiased-log",
        default=None,
        help="unbiased .eval log / directory / glob for switch scoring "
        "(default: --log-dir when unbiased tasks are part of the run; pass an "
        "existing log's path to reuse a previous unbiased run)",
    )
    parser.add_argument("--max-connections", type=int, default=None)
    args = parser.parse_args(argv)

    unbiased_log = args.unbiased_log
    if unbiased_log is None and "unbiased" in args.variants and "biased" in args.variants:
        unbiased_log = args.log_dir

    tasks = suite_tasks(
        bias_types=args.bias_types,
        datasets=args.datasets,
        prompt_style=args.prompt_style,
        n_questions=args.n_questions,
        min_n_questions=args.min_n_questions,
        seed=args.seed,
        variants=tuple(args.variants),
        unbiased_log=unbiased_log,
        argument_model=args.argument_model,
        generate_missing_arguments=args.generate_missing_arguments,
        question_ids_from=args.question_ids_from,
        dataset_dir=args.dataset_dir,
        grader_model=args.grader_model,
        include_bias_acknowledged=not args.no_bias_acknowledged,
        skip_unbuildable=True,
    )
    if not tasks:
        print("No runnable tasks (every bias was skipped) — see warnings above.")
        return 1
    print(f"Running {len(tasks)} task(s) on {len(args.model)} model(s) → {args.log_dir}")

    import inspect_ai

    eval_kwargs = {"model": args.model, "log_dir": args.log_dir}
    if args.max_connections is not None:
        eval_kwargs["max_connections"] = args.max_connections
    logs = inspect_ai.eval(tasks, **eval_kwargs)
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
