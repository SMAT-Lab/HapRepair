#!/usr/bin/env python3
import csv
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from server import (
    ADJUDICATION_COLUMNS,
    AUTHOR_COLUMNS,
    AnnotationHandler,
    AnnotationStore,
    ThreadingHTTPServer,
)


SOURCE_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "adjudication_packages"
    / "exp_indep_final_static_blind_01"
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
            row["rationale"] = ""
        write_csv(path, AUTHOR_COLUMNS, rows)

    def test_author_persistence_and_role_isolation(self):
        other_rows = read_csv(self.bundle / "labels_author_2.csv")
        other_rows[0]["label"] = "Incorrect"
        other_rows[0]["rationale"] = "private second-author rationale"
        write_csv(self.bundle / "labels_author_2.csv", AUTHOR_COLUMNS, other_rows)

        store = AnnotationStore(self.package, "author_1")
        blind_id = store.order[0]
        bootstrap = store.save(blind_id, "Suspicious", "Needs framework evidence")
        serialized = json.dumps(bootstrap)
        self.assertNotIn("private second-author rationale", serialized)
        self.assertNotIn("author_2_label", serialized)
        saved = read_csv(self.bundle / "labels_author_1.csv")
        self.assertEqual(saved[0]["blind_id"], blind_id)
        self.assertEqual(saved[0]["label"], "Suspicious")
        self.assertEqual(saved[0]["rationale"], "Needs framework evidence")
        self.assertEqual([row["blind_id"] for row in saved], store.order)
        self.assertFalse(list(self.bundle.glob(".labels_author_1.csv.*.tmp")))

    def test_rationale_is_required_for_non_correct_labels(self):
        store = AnnotationStore(self.package, "author_1")
        with self.assertRaisesRegex(ValueError, "rationale is required"):
            store.save(store.order[0], "Incorrect", "")

    def test_adjudicator_is_locked_until_both_authors_complete(self):
        store = AnnotationStore(self.package, "adjudicator")
        bootstrap = store.bootstrap()
        self.assertEqual(bootstrap["mode"], "locked")
        self.assertEqual(bootstrap["cases"], [])
        self.assertIn("all 60 cases", bootstrap["message"])

    def test_adjudicator_receives_only_disagreements(self):
        self.fill_author("author_1", "Correct")
        self.fill_author("author_2", "Correct")
        second_path = self.bundle / "labels_author_2.csv"
        second = read_csv(second_path)
        disagreement_id = second[7]["blind_id"]
        second[7]["label"] = "Incorrect"
        second[7]["rationale"] = "Behavior removed"
        write_csv(second_path, AUTHOR_COLUMNS, second)

        store = AnnotationStore(self.package, "adjudicator")
        bootstrap = store.bootstrap()
        self.assertEqual(bootstrap["mode"], "adjudication")
        self.assertEqual(
            [case["blind_id"] for case in bootstrap["cases"]], [disagreement_id]
        )
        store.save(disagreement_id, "Incorrect", "Reference behavior is absent")
        rows = read_csv(self.package / "third_author_adjudication.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0]), ADJUDICATION_COLUMNS)
        self.assertEqual(rows[0]["adjudicated_label"], "Incorrect")


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
            return response.status, json.load(response)

    def test_api_reads_case_and_saves_label(self):
        status, bootstrap = self.request("/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(bootstrap["role"], "author_1")
        self.assertNotIn("author_2_label", json.dumps(bootstrap))
        blind_id = bootstrap["cases"][0]["blind_id"]
        status, case = self.request(f"/api/cases/{blind_id}")
        self.assertEqual(status, 200)
        self.assertEqual(case["blind_id"], blind_id)
        self.assertNotIn("requested_model", case)
        status, updated = self.request(
            f"/api/labels/{blind_id}",
            method="POST",
            payload={"label": "Correct", "rationale": ""},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["completed"], 1)

    def test_api_rejects_incomplete_annotation(self):
        blind_id = self.server.store.order[0]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                f"/api/labels/{blind_id}",
                method="POST",
                payload={"label": "Incorrect", "rationale": ""},
            )
        self.assertEqual(caught.exception.code, 400)

    def test_combined_api_switches_roles_without_mixing_labels(self):
        author_2_path = self.package / "judge_bundle" / "labels_author_2.csv"
        author_2_rows = read_csv(author_2_path)
        author_2_rows[0]["label"] = "Incorrect"
        author_2_rows[0]["rationale"] = "Visible only to author 2"
        write_csv(author_2_path, AUTHOR_COLUMNS, author_2_rows)
        self.server.stores = {
            role: AnnotationStore(self.package, role)
            for role in ("author_1", "author_2", "adjudicator")
        }

        _, first = self.request("/api/bootstrap?role=author_1")
        _, second = self.request("/api/bootstrap?role=author_2")
        self.assertEqual(
            first["available_roles"], ["author_1", "author_2", "adjudicator"]
        )
        self.assertEqual(first["cases"][0]["label"], "")
        self.assertNotIn("Visible only to author 2", json.dumps(first))
        self.assertEqual(second["cases"][0]["label"], "Incorrect")
        self.assertIn("Visible only to author 2", json.dumps(second))


if __name__ == "__main__":
    unittest.main()
