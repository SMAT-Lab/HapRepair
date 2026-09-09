#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_localization as runner
import audit_localization_run as auditor
import evaluate_homecheck_condition as homecheck_evaluator


class LocalizationRunnerTests(unittest.TestCase):
    def test_frozen_benchmark_contract(self) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_PROTOCOL)
        self.assertEqual(len(benchmark.files), 18)
        self.assertEqual(len(benchmark.rules), 8)
        self.assertEqual(
            sum(len(meta["defects"]) for meta in benchmark.gt.values()),
            38,
        )
        self.assertEqual(
            benchmark.gt_sha256,
            "72dddda7a1bd63341b3b50133365c8c922fa5dca0cc7f85f3632afea8788bbe3",
        )

    def test_prompt_does_not_expose_ground_truth(self) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_PROTOCOL)
        relative = benchmark.files[0]
        code = (benchmark.source_root / relative).read_text(encoding="utf-8")
        prompt = runner.build_prompt(code, relative, benchmark.rules)
        self.assertIn(code, prompt)
        self.assertNotIn("gt_defects", prompt)
        self.assertNotIn("ground truth", prompt.lower())

    def test_bounded_retry_prompt_only_adds_format_constraints(self) -> None:
        prompt = "base prompt\n"
        retried = runner.add_bounded_retry_instruction(prompt, 213)
        self.assertTrue(retried.startswith(prompt))
        self.assertIn("one item per distinct violating construct", retried)
        self.assertIn("between 1 and 213", retried)

    def test_parse_object_array_and_fence(self) -> None:
        rules = {"@performance/rule-a"}
        expected = [{"rule": "@performance/rule-a", "line": 7}]
        for content in (
            '{"defects":[{"rule":"@performance/rule-a","line":7}]}',
            '[{"rule":"@performance/rule-a","line":"7"}]',
            '```json\n{"defects":[{"rule":"@performance/rule-a","line":7}]}\n```',
        ):
            parsed, error = runner.parse_prediction(content, rules)
            self.assertIsNone(error)
            self.assertEqual(parsed, expected)

    def test_parse_rejects_unknown_rule(self) -> None:
        parsed, error = runner.parse_prediction(
            '{"defects":[{"rule":"@performance/not-frozen","line":7}]}',
            {"@performance/rule-a"},
        )
        self.assertEqual(parsed, [])
        self.assertEqual(error, "schema_error:unknown_rule")

    def test_metric_counts(self) -> None:
        metrics = runner.metric_counts(
            {("a", 1), ("b", 2)},
            {("a", 1), ("c", 3)},
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["f1"], 0.5)

    def test_file_rule_evaluation_ignores_lines_and_deduplicates_rules(self) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_PROTOCOL)
        relative = "entry/src/main/ets/MainAbility/pages/List/List07.ets"
        records = [
            {
                "model_requested": "test-model",
                "file": relative,
                "status": "completed",
                "parse_ok": True,
                "predicted_defects": [
                    {
                        "rule": "@performance/hp-arkui-load-on-demand",
                        "line": 999,
                    }
                ],
            }
        ]
        summary = runner.evaluate_file_rule_records(benchmark, records)
        overall = summary["model_summaries"]["test-model"]["strata"]["overall"]
        self.assertEqual(overall["tp"], 1)
        self.assertEqual(overall["fp"], 0)
        self.assertEqual(overall["fn"], 1)

    def test_provider_specific_endpoint_paths(self) -> None:
        self.assertEqual(
            runner.endpoint_url("https://example.test", "/v1/chat/completions"),
            "https://example.test/v1/chat/completions",
        )

    def test_thinking_is_disabled(self) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_PROTOCOL)
        self.assertEqual(
            benchmark.protocol["generation"]["reasoning_effort"],
            "none",
        )
        self.assertEqual(
            runner.endpoint_url(
                "https://example.test/v1", "/chat/completions"
            ),
            "https://example.test/v1/chat/completions",
        )

    def test_new_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            runner.write_json_new(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                runner.write_json_new(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text())["value"], 1)

    def test_completed_formal_run_passes_audit(self) -> None:
        run_dir = runner.DEFAULT_RUN_ROOT / "exp_loc_18_three_model_v1"
        if not (run_dir / "summary.json").is_file():
            self.skipTest("formal run is not available")
        audit = auditor.audit_run(runner.DEFAULT_PROTOCOL, run_dir)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observations"]["response_count"], 54)
        self.assertEqual(audit["observations"]["parse_failure_count"], 0)

    def test_formal_subset_files_must_be_frozen_and_unique(self) -> None:
        benchmark = runner.load_benchmark(runner.DEFAULT_PROTOCOL)
        self.assertIn(
            "entry/src/main/ets/MainAbility/pages/List/ListLevel1.ets",
            benchmark.files,
        )
        self.assertEqual(len(set(benchmark.files)), len(benchmark.files))

    def test_completed_homecheck_condition_passes_audit(self) -> None:
        run_dir = homecheck_evaluator.DEFAULT_HOMECHECK_RUN
        if not (run_dir / "state.json").is_file():
            self.skipTest("HomeCheck run is not available")
        _, combined, audit, predictions = homecheck_evaluator.evaluate(
            runner.DEFAULT_PROTOCOL,
            run_dir,
            homecheck_evaluator.DEFAULT_LLM_RUN,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(len(predictions), 38)
        homecheck = combined["comparison"][0]
        self.assertEqual(homecheck["tp"], 30)
        self.assertEqual(homecheck["fp"], 0)
        self.assertEqual(homecheck["fn"], 0)


if __name__ == "__main__":
    unittest.main()
