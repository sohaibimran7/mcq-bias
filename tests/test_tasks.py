"""Tests for the publishable @task entry points, the suite CLI, and the
n_questions / question_ids_from / min_n_questions machinery."""

import json

import pytest


class TestPublishableTasks:
    """The @task entry points over the materialize-once / load-forever mechanism.
    Offline: pipeline sources are monkeypatched to synthetic records."""

    def _records(self, n=4):
        from mcq_bias.pipeline.records import MCQRecord

        return [
            MCQRecord(
                question=f"Task-level question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit"
            )
            for i in range(n)
        ]

    def _patch_sources(self, monkeypatch, records):
        from mcq_bias.pipeline import sources

        monkeypatch.setattr(
            sources,
            "load_records",
            lambda dataset, n_questions=None, seed="42", revision=None: records[:n_questions],
        )

    def test_materialize_once_then_load_forever(self, tmp_path, monkeypatch):
        from mcq_bias.pipeline import sources
        from mcq_bias.tasks import (
            frozen_path,
            mcq_bias,
            mcq_bias_unbiased,
            unbiased_frozen_path,
        )

        self._patch_sources(monkeypatch, self._records())
        biased = mcq_bias(bias_type="suggested_answer", dataset="unit", n_questions=3, dataset_dir=str(tmp_path))
        path = frozen_path("unit", "suggested_answer", "none", 3, "42", tmp_path)
        unbiased_path = unbiased_frozen_path("unit", "none", 3, "42", tmp_path)
        assert path.exists()
        assert unbiased_path.exists()  # shared unbiased set frozen from the same snapshot
        frozen_bytes = path.read_bytes()

        # "another checkpoint, another day": sources gone entirely — cache hit anyway
        def boom(*a, **k):
            raise AssertionError("sources must not be touched once the frozen file exists")

        monkeypatch.setattr(sources, "load_records", boom)

        unbiased = mcq_bias_unbiased(dataset="unit", n_questions=3, dataset_dir=str(tmp_path))
        again = mcq_bias(bias_type="suggested_answer", dataset="unit", n_questions=3, dataset_dir=str(tmp_path))

        assert path.read_bytes() == frozen_bytes  # the file never changes
        ids = [s.id for s in biased.dataset]
        assert len(ids) == 3
        assert [s.id for s in again.dataset] == ids
        assert [s.id for s in unbiased.dataset] == ids  # unbiased pairs by id
        assert biased.dataset[0].metadata["variant"] == "biased"
        assert unbiased.dataset[0].metadata["variant"] == "unbiased"
        assert "biased_option" not in unbiased.dataset[0].metadata  # no bias metadata

    def test_one_unbiased_run_serves_all_biases(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias, mcq_bias_unbiased

        self._patch_sources(monkeypatch, self._records())
        for bias in ("suggested_answer", "post_hoc"):
            mcq_bias(bias_type=bias, dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
        unbiased = mcq_bias_unbiased(dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
        unbiased_files = list(tmp_path.glob("*_unbiased_*.jsonl"))
        assert len(unbiased_files) == 1  # ONE unbiased file for both biases
        unbiased_ids = {s.id for s in unbiased.dataset}
        for bias in ("suggested_answer", "post_hoc"):
            biased = mcq_bias(bias_type=bias, dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
            assert {s.id for s in biased.dataset} <= unbiased_ids  # superset guarantee

    def test_different_params_are_different_files(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import frozen_path, mcq_bias

        self._patch_sources(monkeypatch, self._records())
        mcq_bias(bias_type="suggested_answer", dataset="unit", n_questions=2, dataset_dir=str(tmp_path))
        mcq_bias(bias_type="post_hoc", dataset="unit", n_questions=2, dataset_dir=str(tmp_path))
        assert frozen_path("unit", "suggested_answer", "none", 2, "42", tmp_path).exists()
        assert frozen_path("unit", "post_hoc", "none", 2, "42", tmp_path).exists()

    def test_unknown_bias_type_rejected_before_network(self):
        from mcq_bias.tasks import mcq_bias

        with pytest.raises(ValueError, match="Unknown bias_type"):
            mcq_bias(bias_type="not_a_bias")  # validated before any HF load

    def test_empty_materialization_errors_and_caches_nothing(self, tmp_path, monkeypatch):
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore
        from mcq_bias.tasks import frozen_path, mcq_bias

        self._patch_sources(monkeypatch, self._records())
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: cls()))
        with pytest.raises(ValueError, match="Only 0/2 matched questions"):
            mcq_bias(bias_type="wrong_argument", dataset="unit", n_questions=2, dataset_dir=str(tmp_path))
        assert not frozen_path("unit", "wrong_argument", "none", 2, "42", tmp_path).exists()

    def test_short_materialization_errors_and_caches_nothing(self, tmp_path, monkeypatch):
        """n_questions is exact-or-error: a partially-covered bias never freezes a short file."""
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore
        from mcq_bias.tasks import frozen_path, mcq_bias

        records = self._records(4)
        self._patch_sources(monkeypatch, records)
        partial = WrongArgumentStore()
        partial.add("only one argument", parsed=records[0].parsed_input())
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: partial))
        with pytest.raises(ValueError, match="Only 1/4 matched questions"):
            mcq_bias(bias_type="wrong_argument", dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
        assert not frozen_path("unit", "wrong_argument", "none", 4, "42", tmp_path).exists()

    def test_pool_smaller_than_n_questions_errors(self, tmp_path, monkeypatch):
        self._patch_sources(monkeypatch, self._records(4))
        from mcq_bias.tasks import mcq_bias_unbiased

        with pytest.raises(ValueError, match="has only 4 questions"):
            mcq_bias_unbiased(dataset="unit", n_questions=9, dataset_dir=str(tmp_path))


class TestQuestionIdsFrom:
    """question_ids_from: restrict the pool to ids present in pre-generated JSONL files."""

    def _records(self, n=4):
        from mcq_bias.pipeline.records import MCQRecord

        return [
            MCQRecord(
                question=f"Task-level question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit"
            )
            for i in range(n)
        ]

    def _patch_sources(self, monkeypatch, records):
        from mcq_bias.pipeline import sources

        monkeypatch.setattr(
            sources,
            "load_records",
            lambda dataset, n_questions=None, seed="42", revision=None: records[:n_questions],
        )

    def _ids_file(self, path, records):
        path.write_text("".join(json.dumps({"question_id": r.question_id}) + "\n" for r in records))
        return str(path)

    def test_restricts_pool_and_brands_filename(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias, mcq_bias_unbiased

        records = self._records(5)
        self._patch_sources(monkeypatch, records)
        allowed = [records[1], records[2], records[4]]
        ids = self._ids_file(tmp_path / "ids.jsonl", allowed)
        biased = mcq_bias(
            bias_type="suggested_answer",
            dataset="unit",
            n_questions=2,
            question_ids_from=[ids],
            dataset_dir=str(tmp_path),
        )
        unbiased = mcq_bias_unbiased(dataset="unit", n_questions=2, question_ids_from=[ids], dataset_dir=str(tmp_path))
        # first two ALLOWED questions in pool order — not the first two of the pool
        expected = [records[1].question_id, records[2].question_id]
        assert [s.id for s in biased.dataset] == expected
        assert [s.id for s in unbiased.dataset] == expected  # shared set matches
        branded = list(tmp_path.glob("*_ids-*.jsonl"))
        assert len(branded) == 2  # biased + unbiased, both carry the restriction hash
        # a restricted run never collides with the unrestricted frozen file
        assert not list(tmp_path.glob("unit_suggested_answer_none_n2_seed42.jsonl"))

    def test_same_restriction_same_file_different_restriction_different_file(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import frozen_path, _question_id_allowlist

        records = self._records(5)
        self._patch_sources(monkeypatch, records)
        ids_a = self._ids_file(tmp_path / "a.jsonl", records[:3])
        ids_b = self._ids_file(tmp_path / "b.jsonl", records[2:])
        slug_a = _question_id_allowlist([ids_a])[1]
        assert slug_a == _question_id_allowlist([ids_a])[1]  # content-derived, stable
        assert slug_a != _question_id_allowlist([ids_b])[1]
        path_a = frozen_path("unit", "suggested_answer", "none", 2, "42", tmp_path, ids_slug=slug_a)
        assert f"_ids-{slug_a}" in path_a.name

    def test_shortfall_reports_per_file_counts(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias

        records = self._records(5)
        self._patch_sources(monkeypatch, records)
        ids = self._ids_file(tmp_path / "sparse.jsonl", [records[3]])
        with pytest.raises(ValueError, match=r"allows only 1 of the requested 2.*sparse\.jsonl: 1"):
            mcq_bias(
                bias_type="suggested_answer",
                dataset="unit",
                n_questions=2,
                question_ids_from=[str(ids)],
                dataset_dir=str(tmp_path),
            )
        assert not list(tmp_path.glob("*_ids-*.jsonl"))  # nothing frozen

    def test_intersection_of_multiple_files(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias

        records = self._records(5)
        self._patch_sources(monkeypatch, records)
        ids_a = self._ids_file(tmp_path / "a.jsonl", records[:4])
        ids_b = self._ids_file(tmp_path / "b.jsonl", records[2:])
        biased = mcq_bias(
            bias_type="suggested_answer",
            dataset="unit",
            n_questions=2,
            question_ids_from=[ids_a, ids_b],
            dataset_dir=str(tmp_path),
        )
        assert [s.id for s in biased.dataset] == [records[2].question_id, records[3].question_id]

    def test_disjoint_files_error_as_empty_intersection(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias

        records = self._records(4)
        self._patch_sources(monkeypatch, records)
        ids_a = self._ids_file(tmp_path / "a.jsonl", records[:2])
        ids_b = self._ids_file(tmp_path / "b.jsonl", records[2:])
        with pytest.raises(ValueError, match="intersection is empty"):
            mcq_bias(
                bias_type="suggested_answer",
                dataset="unit",
                n_questions=1,
                question_ids_from=[ids_a, ids_b],
                dataset_dir=str(tmp_path),
            )

    def test_file_without_question_ids_rejected(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias

        self._patch_sources(monkeypatch, self._records(3))
        bogus = tmp_path / "not_ids.jsonl"
        bogus.write_text(json.dumps({"something": "else"}) + "\n")
        with pytest.raises(ValueError, match="no 'question_id' fields"):
            mcq_bias(
                bias_type="suggested_answer",
                dataset="unit",
                n_questions=1,
                question_ids_from=[str(bogus)],
                dataset_dir=str(tmp_path),
            )

    def test_pool_dedupes_repeated_questions(self):
        """Sources can repeat a question verbatim (same content hash) — the pool keeps
        one copy, so sample ids stay unique and the by-id switch join stays unambiguous."""
        from mcq_bias.pipeline.sources import _shuffled_capped

        records = self._records(4)
        with_dupes = records + [records[1], records[3]]
        pooled = _shuffled_capped(with_dupes, None, seed="42")
        assert len(pooled) == 4
        assert len({r.question_id for r in pooled}) == 4
        # prefix property survives dedupe: capping equals slicing the deduped order
        assert _shuffled_capped(with_dupes, 2, seed="42") == pooled[:2]


class TestMinNQuestions:
    """min_n_questions: a floor below n_questions for biases whose injection can
    fail per-question (n_questions stays exact-or-error by default)."""

    def _records(self, n=4):
        from mcq_bias.pipeline.records import MCQRecord

        return [
            MCQRecord(question=f"Floor question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit")
            for i in range(n)
        ]

    def _patch_sources(self, monkeypatch, records):
        from mcq_bias.pipeline import sources

        monkeypatch.setattr(
            sources,
            "load_records",
            lambda dataset, n_questions=None, seed="42", revision=None: records[:n_questions],
        )

    def _patch_partial_store(self, monkeypatch, records, covered):
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore

        store = WrongArgumentStore()
        for r in covered:
            store.add(f"argument for {r.question_id}", parsed=r.parsed_input())
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: store))

    def test_floor_tolerates_partial_match(self, tmp_path, monkeypatch, capsys):
        from mcq_bias.tasks import mcq_bias, mcq_bias_unbiased

        records = self._records(4)
        self._patch_sources(monkeypatch, records)
        self._patch_partial_store(monkeypatch, records, records[:3])  # 3 of 4 covered
        biased = mcq_bias(
            bias_type="wrong_argument", dataset="unit", n_questions=4, min_n_questions=2, dataset_dir=str(tmp_path)
        )
        assert len(biased.dataset) == 3
        assert "matched 3/4" in capsys.readouterr().out  # loud, not silent
        # biased set ⊆ the shared unbiased set (the full prefix) — pairing intact
        unbiased = mcq_bias_unbiased(dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
        assert {s.id for s in biased.dataset} <= {s.id for s in unbiased.dataset}
        assert len(unbiased.dataset) == 4  # the unbiased set never shrinks

    def test_floor_still_errors_below(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import frozen_path, mcq_bias

        records = self._records(4)
        self._patch_sources(monkeypatch, records)
        self._patch_partial_store(monkeypatch, records, records[:1])  # only 1 covered
        with pytest.raises(ValueError, match=r"Only 1/4 matched questions.*floor: 2"):
            mcq_bias(
                bias_type="wrong_argument", dataset="unit", n_questions=4, min_n_questions=2, dataset_dir=str(tmp_path)
            )
        assert not frozen_path("unit", "wrong_argument", "none", 4, "42", tmp_path).exists()

    def test_stricter_run_rejects_cached_short_file(self, tmp_path, monkeypatch):
        """A file frozen under a floor must not silently serve a stricter request."""
        from mcq_bias.tasks import mcq_bias

        records = self._records(4)
        self._patch_sources(monkeypatch, records)
        self._patch_partial_store(monkeypatch, records, records[:3])
        mcq_bias(
            bias_type="wrong_argument", dataset="unit", n_questions=4, min_n_questions=2, dataset_dir=str(tmp_path)
        )  # freezes 3 rows
        with pytest.raises(ValueError, match="holds 3 matched pairs, below the requested floor of 4"):
            mcq_bias(bias_type="wrong_argument", dataset="unit", n_questions=4, dataset_dir=str(tmp_path))
        # the tolerant parameterization still loads its own file fine
        again = mcq_bias(
            bias_type="wrong_argument", dataset="unit", n_questions=4, min_n_questions=3, dataset_dir=str(tmp_path)
        )
        assert len(again.dataset) == 3

    def test_floor_bounds_validated(self, tmp_path, monkeypatch):
        from mcq_bias.tasks import mcq_bias

        self._patch_sources(monkeypatch, self._records(4))
        for bad in (0, 5):
            with pytest.raises(ValueError, match="min_n_questions must be in"):
                mcq_bias(
                    bias_type="suggested_answer",
                    dataset="unit",
                    n_questions=4,
                    min_n_questions=bad,
                    dataset_dir=str(tmp_path),
                )


class TestSuiteCli:
    """python -m mcq_bias — argparse wrapper over suite_tasks + inspect_ai.eval."""

    def test_cli_builds_suite_and_invokes_eval(self, tmp_path, monkeypatch):
        import inspect_ai

        from mcq_bias.__main__ import main
        from mcq_bias.pipeline import sources
        from mcq_bias.pipeline.records import MCQRecord

        records = [
            MCQRecord(question=f"Cli question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit")
            for i in range(3)
        ]
        monkeypatch.setattr(
            sources, "load_records", lambda dataset, n_questions=None, seed="42", **kw: records[:n_questions]
        )
        captured = {}

        def fake_eval(tasks, **kwargs):
            captured["tasks"] = tasks
            captured.update(kwargs)
            return []

        monkeypatch.setattr(inspect_ai, "eval", fake_eval)
        rc = main(
            [
                "--model", "mockllm/model",
                "--bias-types", "suggested_answer", "post_hoc",
                "--datasets", "unit",
                "--n-questions", "3",
                "--dataset-dir", str(tmp_path),
                "--log-dir", str(tmp_path / "logs"),
            ]
        )  # fmt: skip
        assert rc == 0
        tasks = captured["tasks"]
        assert len(tasks) == 3  # ONE shared unbiased + 2 biased
        assert "unbiased" in tasks[0].name  # unbiased scheduled FIRST (sequential switch resolution)
        # biased tasks watch the log dir: switch scorer added on top of the bias scorers
        assert len(tasks[1].scorer) == 4 and len(tasks[0].scorer) == 2
        assert captured["model"] == ["mockllm/model"]
        assert captured["log_dir"] == str(tmp_path / "logs")

    def test_cli_rejects_unknown_bias(self, capsys):
        from mcq_bias.__main__ import main

        with pytest.raises(SystemExit):
            main(["--model", "m", "--bias-types", "not_a_bias"])

    def test_cli_skips_unbuildable_bias_and_runs_the_rest(self, tmp_path, monkeypatch, capsys):
        """A bias that cannot reach n_questions is skipped loudly; the suite continues."""
        import inspect_ai

        from mcq_bias.__main__ import main
        from mcq_bias.pipeline import sources
        from mcq_bias.pipeline.records import MCQRecord
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore

        records = [
            MCQRecord(question=f"Skip question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit")
            for i in range(3)
        ]
        monkeypatch.setattr(
            sources, "load_records", lambda dataset, n_questions=None, seed="42", **kw: records[:n_questions]
        )
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: cls()))
        captured = {}
        monkeypatch.setattr(inspect_ai, "eval", lambda tasks, **kw: captured.update(tasks=tasks) or [])
        rc = main(
            [
                "--model", "mockllm/model",
                "--bias-types", "suggested_answer", "wrong_argument",
                "--datasets", "unit",
                "--n-questions", "3",
                "--dataset-dir", str(tmp_path),
                "--log-dir", str(tmp_path / "logs"),
            ]
        )  # fmt: skip
        assert rc == 0
        assert len(captured["tasks"]) == 2  # unbiased + suggested_answer; wrong_argument skipped
        assert "SKIPPING wrong_argument" in capsys.readouterr().out

    def test_cli_generate_missing_arguments_unskips_wrong_argument(self, tmp_path, monkeypatch):
        """--generate-missing-arguments fills the store during materialization instead of skipping."""
        import inspect_ai

        from mcq_bias.__main__ import main
        from mcq_bias.pipeline import sources, wrong_arguments
        from mcq_bias.pipeline.records import MCQRecord
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore

        records = [
            MCQRecord(question=f"Gen question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit")
            for i in range(3)
        ]
        monkeypatch.setattr(
            sources, "load_records", lambda dataset, n_questions=None, seed="42", **kw: records[:n_questions]
        )
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: cls()))

        def fake_generate(recs, model_name, store, **kwargs):
            for r in recs:
                store.add(f"generated for {r.question_id}", question_id=r.question_id)
            return len(recs)

        monkeypatch.setattr(wrong_arguments, "generate_wrong_arguments_sync", fake_generate)
        captured = {}
        monkeypatch.setattr(inspect_ai, "eval", lambda tasks, **kw: captured.update(tasks=tasks) or [])
        rc = main(
            [
                "--model", "mockllm/model",
                "--bias-types", "suggested_answer", "wrong_argument",
                "--datasets", "unit",
                "--n-questions", "3",
                "--generate-missing-arguments",
                "--dataset-dir", str(tmp_path),
                "--log-dir", str(tmp_path / "logs"),
            ]
        )  # fmt: skip
        assert rc == 0
        assert len(captured["tasks"]) == 3  # unbiased + suggested_answer + wrong_argument (unskipped)

    def test_cli_min_n_questions_unskips_partially_covered_bias(self, tmp_path, monkeypatch, capsys):
        """--min-n-questions lets a partially-covered bias run (smaller set, loud warning)
        instead of being skipped."""
        import inspect_ai

        from mcq_bias.__main__ import main
        from mcq_bias.pipeline import sources
        from mcq_bias.pipeline.records import MCQRecord
        from mcq_bias.pipeline.wrong_arguments import WrongArgumentStore

        records = [
            MCQRecord(
                question=f"Floor cli question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit"
            )
            for i in range(4)
        ]
        monkeypatch.setattr(
            sources, "load_records", lambda dataset, n_questions=None, seed="42", **kw: records[:n_questions]
        )
        store = WrongArgumentStore()
        for r in records[:3]:  # 3 of 4 covered
            store.add(f"argument for {r.question_id}", parsed=r.parsed_input())
        monkeypatch.setattr(WrongArgumentStore, "for_model", classmethod(lambda cls, model, data_dir=None: store))
        captured = {}
        monkeypatch.setattr(inspect_ai, "eval", lambda tasks, **kw: captured.update(tasks=tasks) or [])
        rc = main(
            [
                "--model", "mockllm/model",
                "--bias-types", "wrong_argument",
                "--datasets", "unit",
                "--n-questions", "4",
                "--min-n-questions", "2",
                "--dataset-dir", str(tmp_path),
                "--log-dir", str(tmp_path / "logs"),
            ]
        )  # fmt: skip
        assert rc == 0
        out = capsys.readouterr().out
        assert "SKIPPING" not in out and "matched 3/4" in out
        assert len(captured["tasks"]) == 2  # unbiased (4 questions) + wrong_argument (3)
        assert len(captured["tasks"][1].dataset) == 3

    def test_cli_question_ids_from_restricts_every_task(self, tmp_path, monkeypatch):
        import inspect_ai

        from mcq_bias.__main__ import main
        from mcq_bias.pipeline import sources
        from mcq_bias.pipeline.records import MCQRecord

        records = [
            MCQRecord(
                question=f"Cli ids question {i}?", options=["a", "b", "c"], ground_truth_idx=i % 3, dataset="unit"
            )
            for i in range(5)
        ]
        monkeypatch.setattr(
            sources, "load_records", lambda dataset, n_questions=None, seed="42", **kw: records[:n_questions]
        )
        ids_path = tmp_path / "ids.jsonl"
        ids_path.write_text("".join(json.dumps({"question_id": r.question_id}) + "\n" for r in records[2:]))
        captured = {}
        monkeypatch.setattr(inspect_ai, "eval", lambda tasks, **kw: captured.update(tasks=tasks) or [])
        rc = main(
            [
                "--model", "mockllm/model",
                "--bias-types", "suggested_answer",
                "--datasets", "unit",
                "--n-questions", "2",
                "--question-ids-from", str(ids_path),
                "--dataset-dir", str(tmp_path),
                "--log-dir", str(tmp_path / "logs"),
            ]
        )  # fmt: skip
        assert rc == 0
        allowed = {r.question_id for r in records[2:]}
        for t in captured["tasks"]:  # unbiased AND biased both restricted to the same set
            assert {s.id for s in t.dataset} <= allowed and len(t.dataset) == 2


class TestDataDirResolution:
    """Nothing ships with the package and nothing is written into it: data
    lives at $MCQ_BIAS_DATA_DIR, else ~/.cache/mcq_bias."""

    def test_env_var_wins(self, monkeypatch, tmp_path):
        from mcq_bias.paths import data_root
        from mcq_bias.tasks import frozen_path

        monkeypatch.setenv("MCQ_BIAS_DATA_DIR", str(tmp_path / "root"))
        assert data_root() == tmp_path / "root"
        assert frozen_path("unit", "suggested_answer", "none", 2, "42").parent == tmp_path / "root" / "generated"

    def test_cache_default_never_inside_the_package(self, monkeypatch):
        from pathlib import Path

        import mcq_bias
        from mcq_bias.paths import generated_dir
        from mcq_bias.pipeline.wrong_arguments import arguments_path

        monkeypatch.delenv("MCQ_BIAS_DATA_DIR", raising=False)
        assert generated_dir() == Path.home() / ".cache" / "mcq_bias" / "generated"
        assert arguments_path("some/model").parent == Path.home() / ".cache" / "mcq_bias" / "wrong_arguments"
        package_dir = str(Path(mcq_bias.__file__).parent)
        assert not str(generated_dir()).startswith(package_dir)
