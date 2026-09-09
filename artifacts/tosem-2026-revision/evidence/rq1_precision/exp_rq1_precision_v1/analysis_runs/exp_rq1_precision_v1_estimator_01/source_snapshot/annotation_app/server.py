#!/usr/bin/env python3
"""Role-isolated local annotation server for EXP-RQ1-PRECISION."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROLES = ("author_1", "author_2", "adjudicator")
SERVER_ROLES = (*ROLES, "all")
LABELS = ("Correct", "Suspicious", "Incorrect")
AUTHOR_COLUMNS = ("order", "blind_id", "label", "rationale")
ADJUDICATION_COLUMNS = (
    "blind_id",
    "author_1_label",
    "author_2_label",
    "adjudicated_label",
    "rationale",
)
DEFAULT_PACKAGE = (
    Path(__file__).resolve().parents[4]
    / "paper/rebuttal/rq1_precision/exp_rq1_precision_v1"
)
RANGE_SOURCES = {
    "ast-node",
    "ir-statement",
    "ir-operand",
    "declaration",
    "comment-block",
}
DETECTOR_IDENTITY = "CodeLinter 6.0.240"


class AnnotationStore:
    def __init__(
        self,
        package_dir: Path,
        role: str,
        range_sidecar: Path | None = None,
    ):
        self.package_dir = package_dir.resolve()
        self.role = role
        self.bundle_dir = self.package_dir / "judge_bundle"
        self.manifest_path = self.bundle_dir / "package_manifest.json"
        self.guide_path = self.bundle_dir / "ANNOTATION_GUIDE.md"
        self.adjudication_path = self.package_dir / "third_author_adjudication.csv"
        self.author_paths = {
            role_name: self.bundle_dir / f"labels_{role_name}.csv"
            for role_name in ("author_1", "author_2")
        }
        self.lock = threading.RLock()
        self._validate_package()
        self.display_ranges = self._load_range_sidecar(range_sidecar)

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError(f"Unexpected columns in {path}")
            return list(reader)

    @staticmethod
    def _atomic_write(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def _safe_path(self, relative: str) -> Path:
        path = (self.bundle_dir / relative).resolve()
        if self.bundle_dir not in path.parents or not path.is_file():
            raise ValueError(f"Invalid package path: {relative}")
        return path

    def _validate_package(self) -> None:
        required = [
            self.manifest_path,
            self.guide_path,
            self.adjudication_path,
            *self.author_paths.values(),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing package files: " + ", ".join(missing))
        manifest = self._read_json(self.manifest_path)
        entries = manifest.get("case_files", [])
        if manifest.get("case_count") != len(entries) or not entries:
            raise ValueError("Invalid case manifest")
        self.entries = entries
        self.order = [entry["blind_id"] for entry in entries]
        if len(self.order) != len(set(self.order)):
            raise ValueError("Duplicate blind IDs")
        self.case_paths = {
            entry["blind_id"]: self._safe_path(entry["path"]) for entry in entries
        }
        for path in self.author_paths.values():
            rows = self._read_csv(path, AUTHOR_COLUMNS)
            if [row["blind_id"] for row in rows] != self.order:
                raise ValueError(f"Annotation order mismatch: {path}")
        self._read_csv(self.adjudication_path, ADJUDICATION_COLUMNS)

    def _load_range_sidecar(self, path: Path | None) -> dict[str, dict]:
        if path is None:
            return {}
        payload = self._read_json(path.resolve())
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported range sidecar schema")
        if payload.get("purpose") != "annotation_display_only":
            raise ValueError("Range sidecar is not annotation-display-only")
        if payload.get("detector_unchanged") != DETECTOR_IDENTITY:
            raise ValueError("Range sidecar changes the frozen detector identity")
        ranges = payload.get("ranges")
        if not isinstance(ranges, dict) or not set(ranges).issubset(self.case_paths):
            raise ValueError("Range sidecar contains invalid blind IDs")
        for blind_id, display_range in ranges.items():
            if not isinstance(display_range, dict):
                raise ValueError(f"Invalid display range: {blind_id}")
            case = self._read_json(self.case_paths[blind_id])
            coordinates = tuple(
                display_range.get(field)
                for field in ("start_line", "start_column", "end_line", "end_column")
            )
            if not all(isinstance(value, int) and value > 0 for value in coordinates):
                raise ValueError(f"Invalid display range coordinates: {blind_id}")
            start_line, start_column, end_line, end_column = coordinates
            if end_line < start_line or (
                end_line == start_line and end_column <= start_column
            ):
                raise ValueError(f"Invalid display range ordering: {blind_id}")
            source_record = self._read_json(self._safe_path(case["source_file"]["path"]))
            source_line_count = len(source_record["content"].splitlines())
            if start_line > source_line_count or end_line > source_line_count:
                raise ValueError(f"Display range is outside frozen source: {blind_id}")
            if display_range.get("source") not in RANGE_SOURCES:
                raise ValueError(f"Display range is not native: {blind_id}")
        return ranges

    def _author_rows(self, role: str) -> list[dict[str, str]]:
        return self._read_csv(self.author_paths[role], AUTHOR_COLUMNS)

    def _author_complete(self, rows: list[dict[str, str]]) -> bool:
        return len(rows) == len(self.order) and all(row["label"] in LABELS for row in rows)

    def _authors_complete(self) -> bool:
        return all(
            self._author_complete(self._author_rows(role))
            for role in ("author_1", "author_2")
        )

    def _adjudication_rows(self) -> list[dict[str, str]]:
        return self._read_csv(self.adjudication_path, ADJUDICATION_COLUMNS)

    def _disagreements(self) -> tuple[list[str], dict[str, dict[str, str]]]:
        first = self._author_rows("author_1")
        second = self._author_rows("author_2")
        if not self._author_complete(first) or not self._author_complete(second):
            raise PermissionError("Adjudication opens after both authors complete all 231 cases.")
        first_by_id = {row["blind_id"]: row for row in first}
        second_by_id = {row["blind_id"]: row for row in second}
        ids = []
        labels = {}
        for blind_id in self.order:
            left = first_by_id[blind_id]["label"]
            right = second_by_id[blind_id]["label"]
            if left != right:
                ids.append(blind_id)
                labels[blind_id] = {"author_1_label": left, "author_2_label": right}
        return ids, labels

    def _case_summary(self, blind_id: str) -> dict:
        case = self._read_json(self.case_paths[blind_id])
        finding = case["finding"]
        return {
            "blind_id": blind_id,
            "category": case["category"],
            "rule": case["rule"],
            "location": f"{finding['relative_path']}:{finding['line']}",
        }

    def bootstrap(self) -> dict:
        with self.lock:
            if self.role in self.author_paths:
                rows = self._author_rows(self.role)
                annotations = {row["blind_id"]: row for row in rows}
                cases = []
                for blind_id in self.order:
                    item = self._case_summary(blind_id)
                    item.update(
                        label=annotations[blind_id]["label"],
                        rationale=annotations[blind_id]["rationale"],
                    )
                    cases.append(item)
                return {
                    "role": self.role,
                    "mode": "annotation",
                    "labels": LABELS,
                    "guide": self.guide_path.read_text(encoding="utf-8"),
                    "total": len(cases),
                    "completed": sum(item["label"] in LABELS for item in cases),
                    "locked": self._authors_complete(),
                    "cases": cases,
                }
            try:
                disagreement_ids, author_labels = self._disagreements()
            except PermissionError as error:
                first = self._author_rows("author_1")
                second = self._author_rows("author_2")
                return {
                    "role": self.role,
                    "mode": "locked",
                    "message": str(error),
                    "guide": self.guide_path.read_text(encoding="utf-8"),
                    "author_progress": {
                        "author_1": sum(row["label"] in LABELS for row in first),
                        "author_2": sum(row["label"] in LABELS for row in second),
                        "total": len(self.order),
                    },
                    "cases": [],
                    "total": 0,
                    "completed": 0,
                }
            existing = {row["blind_id"]: row for row in self._adjudication_rows()}
            cases = []
            for blind_id in disagreement_ids:
                item = self._case_summary(blind_id)
                item.update(author_labels[blind_id])
                saved = existing.get(blind_id, {})
                item["label"] = saved.get("adjudicated_label", "")
                item["rationale"] = saved.get("rationale", "")
                cases.append(item)
            return {
                "role": self.role,
                "mode": "adjudication",
                "labels": LABELS,
                "guide": self.guide_path.read_text(encoding="utf-8"),
                "total": len(cases),
                "completed": sum(item["label"] in LABELS for item in cases),
                "agreement_count": len(self.order) - len(cases),
                "cases": cases,
            }

    def get_case(self, blind_id: str) -> dict:
        with self.lock:
            bootstrap = self.bootstrap()
            allowed = {item["blind_id"] for item in bootstrap.get("cases", [])}
            if blind_id not in allowed:
                raise PermissionError("Case is not available for this role")
            case = self._read_json(self.case_paths[blind_id])
            source_record = self._read_json(self._safe_path(case["source_file"]["path"]))
            evidence = []
            for relative in case["rule_evidence"]["semantic_spec_paths"]:
                evidence.append(
                    {"name": Path(relative).name, "content": self._safe_path(relative).read_text(encoding="utf-8")}
                )
            reference = case["rule_evidence"]["static_reference_path"]
            if reference:
                evidence.append(
                    {"name": "Static rule reference", "content": self._safe_path(reference).read_text(encoding="utf-8")}
                )
            summary = next(item for item in bootstrap["cases"] if item["blind_id"] == blind_id)
            return {
                "blind_id": blind_id,
                "category": case["category"],
                "rule": case["rule"],
                "finding": case["finding"],
                "display_range": self.display_ranges.get(blind_id),
                "source": source_record,
                "rule_evidence": evidence,
                "annotation": {
                    key: summary.get(key, "")
                    for key in ("label", "rationale", "author_1_label", "author_2_label")
                    if key in summary
                },
            }

    def save(self, blind_id: str, label: str, rationale: str) -> dict:
        label = label.strip()
        rationale = rationale.strip()
        if label not in LABELS:
            raise ValueError("Select Correct, Suspicious, or Incorrect")
        if label != "Correct" and not rationale:
            raise ValueError("Rationale is required for Suspicious and Incorrect")
        with self.lock:
            if self.role in self.author_paths:
                if self._authors_complete():
                    raise PermissionError("Author labels are frozen because both authors completed annotation")
                rows = self._author_rows(self.role)
                matching = [row for row in rows if row["blind_id"] == blind_id]
                if len(matching) != 1:
                    raise KeyError(blind_id)
                matching[0]["label"] = label
                matching[0]["rationale"] = rationale
                self._atomic_write(self.author_paths[self.role], AUTHOR_COLUMNS, rows)
                return self.bootstrap()

            disagreement_ids, author_labels = self._disagreements()
            if blind_id not in disagreement_ids:
                raise PermissionError("Adjudicator may label disagreements only")
            existing = {row["blind_id"]: row for row in self._adjudication_rows()}
            existing[blind_id] = {
                "blind_id": blind_id,
                **author_labels[blind_id],
                "adjudicated_label": label,
                "rationale": rationale,
            }
            rows = [existing[item] for item in disagreement_ids if item in existing]
            self._atomic_write(self.adjudication_path, ADJUDICATION_COLUMNS, rows)
            return self.bootstrap()


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "HapRepairRQ1Annotation/1.0"

    def _store(self) -> AnnotationStore:
        stores = getattr(self.server, "stores", None)
        if stores is not None:
            query = parse_qs(urlparse(self.path).query)
            role = query.get("role", [None])[0]
            if role not in stores:
                raise ValueError("Select a valid role")
            return stores[role]
        return self.server.store

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, relative: str) -> None:
        path = (self.server.static_dir / relative).resolve()
        if self.server.static_dir not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html; charset=utf-8"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                payload = self._store().bootstrap()
                if getattr(self.server, "stores", None) is not None:
                    payload["available_roles"] = list(ROLES)
                self._json(HTTPStatus.OK, payload)
            elif parsed.path.startswith("/api/cases/"):
                self._json(HTTPStatus.OK, self._store().get_case(unquote(parsed.path.split("/")[-1])))
            elif parsed.path in ("/", "/index.html"):
                self._static("index.html")
            elif parsed.path == "/app.js":
                self._static("app.js")
            elif parsed.path == "/app.css":
                self._static("app.css")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except PermissionError as error:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(error)})
        except (KeyError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/labels/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = self._store().save(
                unquote(parsed.path.split("/")[-1]),
                str(payload.get("label", "")),
                str(payload.get("rationale", "")),
            )
            compact = parse_qs(parsed.query).get("compact") == ["1"]
            if compact:
                result = {
                    "saved": True,
                    "completed": result.get("completed", 0),
                    "locked": result.get("locked", False),
                }
            if getattr(self.server, "stores", None) is not None:
                result["available_roles"] = list(ROLES)
            self._json(HTTPStatus.OK, result)
        except PermissionError as error:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(error)})
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--role", choices=SERVER_ROLES, default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--range-sidecar", type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    server.static_dir = Path(__file__).resolve().parent / "static"
    if args.role == "all":
        server.stores = {
            role: AnnotationStore(args.package, role, args.range_sidecar)
            for role in ROLES
        }
    else:
        server.store = AnnotationStore(args.package, args.role, args.range_sidecar)
    print(f"http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
