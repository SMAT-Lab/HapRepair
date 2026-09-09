#!/usr/bin/env python3
import csv
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from annotation_app.server import (
    ADJUDICATION_COLUMNS,
    AUTHOR_COLUMNS,
    AnnotationHandler,
    AnnotationStore,
    ThreadingHTTPServer,
)


SOURCE_PACKAGE = (
    Path(__file__).resolve().parents[4]
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1"
)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reset_package_annotations(package):
    bundle = package / "judge_bundle"
    for role in ("author_1", "author_2"):
        path = bundle / f"labels_{role}.csv"
        rows = read_csv(path)
        for row in rows:
            row["label"] = ""
            row["rationale"] = ""
        write_csv(path, AUTHOR_COLUMNS, rows)
    write_csv(package / "third_author_adjudication.csv", ADJUDICATION_COLUMNS, [])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        shutil.copytree(SOURCE_PACKAGE, self.package)
        reset_package_annotations(self.package)

    def tearDown(self):
        self.temp.cleanup()

    @property
    def bundle(self):
        return self.package / "judge_bundle"

    def fill_author(self, role, label="Correct"):
        path = self.bundle / f"labels_{role}.csv"
        rows = read_csv(path)
        for row in rows:
            row["label"] = label
            row["rationale"] = "" if label == "Correct" else "Required rationale"
        write_csv(path, AUTHOR_COLUMNS, rows)

    def test_author_persistence_is_atomic_and_role_isolated(self):
        other_rows = read_csv(self.bundle / "labels_author_2.csv")
        other_rows[0]["label"] = "Incorrect"
        other_rows[0]["rationale"] = "private second-author rationale"
        write_csv(self.bundle / "labels_author_2.csv", AUTHOR_COLUMNS, other_rows)

        store = AnnotationStore(self.package, "author_1")
        blind_id = store.order[0]
        bootstrap = store.save(blind_id, "Suspicious", "Needs stronger rule evidence")
        serialized = json.dumps(bootstrap)
        self.assertNotIn("private second-author rationale", serialized)
        self.assertNotIn("author_2_label", serialized)
        saved = read_csv(self.bundle / "labels_author_1.csv")
        self.assertEqual(saved[0]["blind_id"], blind_id)
        self.assertEqual(saved[0]["label"], "Suspicious")
        self.assertEqual(saved[0]["rationale"], "Needs stronger rule evidence")
        self.assertEqual([row["blind_id"] for row in saved], store.order)
        self.assertFalse(list(self.bundle.glob(".labels_author_1.csv.*.tmp")))

    def test_non_correct_label_requires_rationale(self):
        store = AnnotationStore(self.package, "author_1")
        with self.assertRaisesRegex(ValueError, "Rationale is required"):
            store.save(store.order[0], "Incorrect", "")

    def test_adjudicator_is_locked_until_231_labels_per_author(self):
        self.fill_author("author_1")
        second_path = self.bundle / "labels_author_2.csv"
        second = read_csv(second_path)
        for row in second[:-1]:
            row["label"] = "Correct"
        write_csv(second_path, AUTHOR_COLUMNS, second)

        bootstrap = AnnotationStore(self.package, "adjudicator").bootstrap()
        self.assertEqual(bootstrap["mode"], "locked")
        self.assertEqual(bootstrap["author_progress"]["author_1"], 231)
        self.assertEqual(bootstrap["author_progress"]["author_2"], 230)
        self.assertEqual(bootstrap["cases"], [])

    def test_adjudicator_receives_disagreements_only(self):
        self.fill_author("author_1")
        self.fill_author("author_2")
        second_path = self.bundle / "labels_author_2.csv"
        second = read_csv(second_path)
        disagreement_id = second[7]["blind_id"]
        second[7]["label"] = "Incorrect"
        second[7]["rationale"] = "Reported condition is absent"
        write_csv(second_path, AUTHOR_COLUMNS, second)

        store = AnnotationStore(self.package, "adjudicator")
        bootstrap = store.bootstrap()
        self.assertEqual(bootstrap["mode"], "adjudication")
        self.assertEqual([case["blind_id"] for case in bootstrap["cases"]], [disagreement_id])
        self.assertEqual(bootstrap["agreement_count"], 230)
        store.save(disagreement_id, "Incorrect", "Source contradicts the rule condition")
        rows = read_csv(self.package / "third_author_adjudication.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0]), ADJUDICATION_COLUMNS)
        self.assertEqual(rows[0]["blind_id"], disagreement_id)

    def test_author_files_freeze_when_both_are_complete(self):
        self.fill_author("author_1")
        self.fill_author("author_2")
        store = AnnotationStore(self.package, "author_1")
        self.assertTrue(store.bootstrap()["locked"])
        with self.assertRaisesRegex(PermissionError, "both authors completed"):
            store.save(store.order[0], "Incorrect", "Late change")

    def test_case_contains_complete_source_and_rule_evidence(self):
        store = AnnotationStore(self.package, "author_1")
        case = store.get_case(store.order[0])
        content_hash = hashlib.sha256(case["source"]["content"].encode()).hexdigest()
        self.assertEqual(case["source"]["content_sha256"], content_hash)
        self.assertIn(case["finding"]["relative_path"], case["source"]["relative_path"])
        self.assertGreater(len(case["source"]["content"]), 20)
        self.assertGreaterEqual(len(case["rule_evidence"]), 2)
        self.assertTrue(all(item["content"].strip() for item in case["rule_evidence"]))
        self.assertIsNone(case["display_range"])

    def test_case_merges_native_range_sidecar_without_mutating_finding(self):
        plain = AnnotationStore(self.package, "author_1")
        blind_id = plain.order[0]
        finding = plain.get_case(blind_id)["finding"]
        sidecar_path = Path(self.temp.name) / "ranges.json"
        display_range = {
            "start_line": finding["line"],
            "start_column": finding["column"],
            "end_line": finding["line"] + 4,
            "end_column": 3,
            "source": "ir-statement",
        }
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "annotation_display_only",
                    "detector_unchanged": "CodeLinter 6.0.240",
                    "ranges": {blind_id: display_range},
                }
            ),
            encoding="utf-8",
        )

        case = AnnotationStore(
            self.package, "author_1", sidecar_path
        ).get_case(blind_id)

        self.assertEqual(case["display_range"], display_range)
        self.assertEqual(case["finding"], finding)

    def test_sidecar_range_may_begin_after_frozen_container_location(self):
        store = AnnotationStore(self.package, "author_1")
        blind_id = store.order[0]
        finding = store.get_case(blind_id)["finding"]
        sidecar_path = Path(self.temp.name) / "following-ranges.json"
        display_range = {
            "start_line": finding["line"] + 1,
            "start_column": finding["column"],
            "end_line": finding["line"] + 2,
            "end_column": 1,
            "source": "ir-statement",
        }
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "annotation_display_only",
                    "detector_unchanged": "CodeLinter 6.0.240",
                    "ranges": {blind_id: display_range},
                }
            ),
            encoding="utf-8",
        )

        case = AnnotationStore(
            self.package, "author_1", sidecar_path
        ).get_case(blind_id)
        self.assertEqual(case["display_range"], display_range)

    def test_sidecar_range_must_stay_inside_frozen_source(self):
        store = AnnotationStore(self.package, "author_1")
        blind_id = store.order[0]
        sidecar_path = Path(self.temp.name) / "outside-source-ranges.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "annotation_display_only",
                    "detector_unchanged": "CodeLinter 6.0.240",
                    "ranges": {
                        blind_id: {
                            "start_line": 999999,
                            "start_column": 1,
                            "end_line": 1000000,
                            "end_column": 1,
                            "source": "ast-node",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "outside frozen source"):
            AnnotationStore(self.package, "author_1", sidecar_path)

    def test_sidecar_range_may_begin_before_frozen_report_line(self):
        plain = AnnotationStore(self.package, "author_1")
        blind_id = plain.order[0]
        finding = plain.get_case(blind_id)["finding"]
        sidecar_path = Path(self.temp.name) / "leading-ranges.json"
        display_range = {
            "start_line": max(1, finding["line"] - 3),
            "start_column": 1,
            "end_line": finding["line"] + 2,
            "end_column": 1,
            "source": "comment-block",
        }
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "annotation_display_only",
                    "detector_unchanged": "CodeLinter 6.0.240",
                    "ranges": {blind_id: display_range},
                }
            ),
            encoding="utf-8",
        )

        case = AnnotationStore(
            self.package, "author_1", sidecar_path
        ).get_case(blind_id)
        self.assertEqual(case["display_range"], display_range)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        shutil.copytree(SOURCE_PACKAGE, self.package)
        reset_package_annotations(self.package)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), AnnotationHandler)
        self.server.store = AnnotationStore(self.package, "author_1")
        self.server.static_dir = Path(__file__).resolve().parent / "static"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, method="GET", payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            content_type = response.headers.get_content_type()
            if content_type == "application/json":
                return response.status, json.load(response)
            return response.status, response.read().decode("utf-8")

    def test_api_reads_evidence_and_saves_label(self):
        status, bootstrap = self.request("/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(bootstrap["role"], "author_1")
        self.assertNotIn("author_2_label", json.dumps(bootstrap))
        blind_id = bootstrap["cases"][0]["blind_id"]
        status, case = self.request(f"/api/cases/{blind_id}")
        self.assertEqual(status, 200)
        self.assertEqual(case["blind_id"], blind_id)
        self.assertIn("source", case)
        self.assertIn("rule_evidence", case)
        status, updated = self.request(
            f"/api/labels/{blind_id}",
            method="POST",
            payload={"label": "Correct", "rationale": ""},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["completed"], 1)

    def test_compact_save_response_omits_case_bundle(self):
        _, bootstrap = self.request("/api/bootstrap")
        blind_id = bootstrap["cases"][0]["blind_id"]
        status, updated = self.request(
            f"/api/labels/{blind_id}?compact=1",
            method="POST",
            payload={"label": "Correct", "rationale": ""},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["saved"], True)
        self.assertEqual(updated["completed"], 1)
        self.assertNotIn("cases", updated)
        self.assertNotIn("guide", updated)

    def test_fixed_role_ignores_role_query_and_never_exposes_other_author(self):
        other_rows = read_csv(self.package / "judge_bundle" / "labels_author_2.csv")
        other_rows[0]["label"] = "Incorrect"
        other_rows[0]["rationale"] = "Visible only on the author 2 server"
        write_csv(self.package / "judge_bundle" / "labels_author_2.csv", AUTHOR_COLUMNS, other_rows)

        _, payload = self.request("/api/bootstrap?role=author_2")
        self.assertEqual(payload["role"], "author_1")
        self.assertNotIn("Visible only on the author 2 server", json.dumps(payload))

    def test_combined_server_switches_roles_and_keeps_files_separate(self):
        other_rows = read_csv(self.package / "judge_bundle" / "labels_author_2.csv")
        other_rows[0]["label"] = "Incorrect"
        other_rows[0]["rationale"] = "Author 2 decision"
        write_csv(self.package / "judge_bundle" / "labels_author_2.csv", AUTHOR_COLUMNS, other_rows)
        self.server.stores = {
            role: AnnotationStore(self.package, role)
            for role in ("author_1", "author_2", "adjudicator")
        }

        _, first = self.request("/api/bootstrap?role=author_1")
        _, second = self.request("/api/bootstrap?role=author_2")
        self.assertEqual(first["available_roles"], ["author_1", "author_2", "adjudicator"])
        self.assertEqual(first["cases"][0]["label"], "")
        self.assertEqual(second["cases"][0]["label"], "Incorrect")
        self.assertNotIn("Author 2 decision", json.dumps(first))

    def test_api_rejects_incomplete_annotation(self):
        blind_id = self.server.store.order[0]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                f"/api/labels/{blind_id}",
                method="POST",
                payload={"label": "Incorrect", "rationale": ""},
            )
        self.assertEqual(caught.exception.code, 400)

    def test_static_frontend_is_served(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("RQ1 Finding Precision", body)
        self.assertIn("sourceCode", body)
        self.assertIn("evidenceContent", body)


if __name__ == "__main__":
    unittest.main()
