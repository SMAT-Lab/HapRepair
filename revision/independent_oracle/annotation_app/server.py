#!/usr/bin/env python3
"""Role-isolated local annotation server for independent repair evaluation."""

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


class AnnotationStore:
    def __init__(self, package_dir: Path, role: str):
        self.package_dir = package_dir.resolve()
        self.role = role
        self.bundle_dir = self.package_dir / "judge_bundle"
        self.manifest_path = self.bundle_dir / "package_manifest.json"
        self.guide_path = self.bundle_dir / "ANNOTATION_GUIDE.md"
        self.adjudication_path = self.package_dir / "third_author_adjudication.csv"
        self.author_paths = {
            "author_1": self.bundle_dir / "labels_author_1.csv",
            "author_2": self.bundle_dir / "labels_author_2.csv",
        }
        self.lock = threading.RLock()
        self._validate_package()

    def _validate_package(self) -> None:
        required = [
            self.manifest_path,
            self.guide_path,
            self.adjudication_path,
            *self.author_paths.values(),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing annotation package files: " + ", ".join(missing)
            )

        manifest = self._read_json(self.manifest_path)
        entries = manifest.get("case_files", [])
        if manifest.get("case_count") != len(entries) or not entries:
            raise ValueError("Package manifest has an invalid case count")
        self.case_entries = entries
        self.order = [entry["blind_id"] for entry in entries]
        if len(self.order) != len(set(self.order)):
            raise ValueError("Package manifest contains duplicate blind IDs")
        self.case_paths = {
            entry["blind_id"]: (self.bundle_dir / entry["path"]).resolve()
            for entry in entries
        }
        for blind_id, path in self.case_paths.items():
            if self.bundle_dir not in path.parents or not path.is_file():
                raise ValueError(f"Invalid case path for {blind_id}")

        for path in self.author_paths.values():
            rows = self._read_csv(path, AUTHOR_COLUMNS)
            if [row["blind_id"] for row in rows] != self.order:
                raise ValueError(f"Annotation order does not match manifest: {path}")
        self._read_csv(self.adjudication_path, ADJUDICATION_COLUMNS)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_csv(
        path: Path, expected_columns: tuple[str, ...]
    ) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected_columns:
                raise ValueError(f"Unexpected columns in {path}")
            return list(reader)

    @staticmethod
    def _atomic_write_csv(
        path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]
    ) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=columns, extrasaction="ignore"
                )
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

    def _author_rows(self, role: str) -> list[dict[str, str]]:
        return self._read_csv(self.author_paths[role], AUTHOR_COLUMNS)

    def _author_complete(self, rows: list[dict[str, str]]) -> bool:
        return len(rows) == len(self.order) and all(
            row["label"] in LABELS for row in rows
        )

    def _adjudication_rows(self) -> list[dict[str, str]]:
        return self._read_csv(self.adjudication_path, ADJUDICATION_COLUMNS)

    def _disagreements(self) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
        first = self._author_rows("author_1")
        second = self._author_rows("author_2")
        if not self._author_complete(first) or not self._author_complete(second):
            raise PermissionError(
                "Adjudication is locked until both authors complete all "
                f"{len(self.order)} cases."
            )
        first_by_id = {row["blind_id"]: row for row in first}
        second_by_id = {row["blind_id"]: row for row in second}
        disagreements = []
        labels = {}
        for blind_id in self.order:
            left = first_by_id[blind_id]["label"]
            right = second_by_id[blind_id]["label"]
            if left != right:
                disagreements.append(
                    {
                        "blind_id": blind_id,
                        "author_1_label": left,
                        "author_2_label": right,
                    }
                )
                labels[blind_id] = {"author_1_label": left, "author_2_label": right}
        return disagreements, labels

    def _case_summary(self, blind_id: str) -> dict:
        case = self._read_json(self.case_paths[blind_id])
        finding = case.get("target_findings", [{}])[0]
        paths = []
        for group in (
            "defective_files",
            "human_reference_files",
            "candidate_repair_files",
        ):
            for item in case.get(group, []):
                if item.get("path") not in paths:
                    paths.append(item.get("path"))
        return {
            "blind_id": blind_id,
            "category": case.get("category", ""),
            "rule": case.get("rule", ""),
            "location": f"{finding.get('file', '')}:{finding.get('line', '')}",
            "file_count": len(paths),
        }

    def bootstrap(self) -> dict:
        with self.lock:
            if self.role in self.author_paths:
                rows = self._author_rows(self.role)
                labels = {
                    row["blind_id"]: {
                        "label": row["label"],
                        "rationale": row["rationale"],
                    }
                    for row in rows
                }
                cases = []
                for blind_id in self.order:
                    summary = self._case_summary(blind_id)
                    summary.update(labels[blind_id])
                    cases.append(summary)
                completed = sum(row["label"] in LABELS for row in rows)
                return {
                    "role": self.role,
                    "mode": "annotation",
                    "guide": self.guide_path.read_text(encoding="utf-8"),
                    "labels": LABELS,
                    "total": len(cases),
                    "completed": completed,
                    "locked": bool(self._adjudication_rows()),
                    "cases": cases,
                }

            try:
                disagreements, author_labels = self._disagreements()
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
                    "completed": 0,
                    "total": 0,
                }

            existing = {row["blind_id"]: row for row in self._adjudication_rows()}
            cases = []
            for item in disagreements:
                blind_id = item["blind_id"]
                summary = self._case_summary(blind_id)
                summary.update(author_labels[blind_id])
                saved = existing.get(blind_id, {})
                summary["label"] = saved.get("adjudicated_label", "")
                summary["rationale"] = saved.get("rationale", "")
                cases.append(summary)
            completed = sum(case["label"] in LABELS for case in cases)
            return {
                "role": self.role,
                "mode": "adjudication",
                "guide": self.guide_path.read_text(encoding="utf-8"),
                "labels": LABELS,
                "total": len(cases),
                "completed": completed,
                "agreement_count": len(self.order) - len(cases),
                "cases": cases,
            }

    def get_case(self, blind_id: str) -> dict:
        with self.lock:
            if blind_id not in self.case_paths:
                raise KeyError(blind_id)
            result = self.bootstrap()
            allowed = {item["blind_id"] for item in result.get("cases", [])}
            if blind_id not in allowed:
                raise PermissionError("This case is not available for the active role.")
            case = self._read_json(self.case_paths[blind_id])
            allowed_fields = {
                "blind_id",
                "category",
                "rule",
                "rule_description",
                "target_findings",
                "defective_files",
                "human_reference_files",
                "candidate_repair_files",
            }
            case = {key: value for key, value in case.items() if key in allowed_fields}
            summary = next(
                item for item in result["cases"] if item["blind_id"] == blind_id
            )
            case["annotation"] = {
                key: summary.get(key, "")
                for key in ("label", "rationale", "author_1_label", "author_2_label")
                if key in summary
            }
            return case

    def save(self, blind_id: str, label: str, rationale: str) -> dict:
        label = label.strip()
        rationale = rationale.strip()
        if label not in LABELS:
            raise ValueError("Select Correct, Suspicious, or Incorrect.")
        if label != "Correct" and not rationale:
            raise ValueError(
                "A rationale is required for Suspicious and Incorrect labels."
            )

        with self.lock:
            if self.role in self.author_paths:
                if self._adjudication_rows():
                    raise PermissionError(
                        "Author annotations are frozen because adjudication has started."
                    )
                rows = self._author_rows(self.role)
                matching = [row for row in rows if row["blind_id"] == blind_id]
                if not matching:
                    raise KeyError(blind_id)
                matching[0]["label"] = label
                matching[0]["rationale"] = rationale
                self._atomic_write_csv(
                    self.author_paths[self.role], AUTHOR_COLUMNS, rows
                )
            else:
                disagreements, author_labels = self._disagreements()
                allowed = {row["blind_id"] for row in disagreements}
                if blind_id not in allowed:
                    raise PermissionError(
                        "Only author disagreements can be adjudicated."
                    )
                existing = {row["blind_id"]: row for row in self._adjudication_rows()}
                existing[blind_id] = {
                    "blind_id": blind_id,
                    **author_labels[blind_id],
                    "adjudicated_label": label,
                    "rationale": rationale,
                }
                rows = [
                    existing[item["blind_id"]]
                    for item in disagreements
                    if item["blind_id"] in existing
                ]
                self._atomic_write_csv(
                    self.adjudication_path, ADJUDICATION_COLUMNS, rows
                )
            return self.bootstrap()


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "HapRepairAnnotation/1.0"

    @property
    def store(self) -> AnnotationStore:
        stores = getattr(self.server, "stores", None)
        if stores is None:
            return self.server.store  # type: ignore[attr-defined]
        role = parse_qs(urlparse(self.path).query).get("role", ["author_1"])[0]
        if role not in stores:
            raise ValueError("Unknown annotation role.")
        return stores[role]

    @property
    def static_dir(self) -> Path:
        return self.server.static_dir  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                payload = self.store.bootstrap()
                if getattr(self.server, "stores", None) is not None:
                    payload["available_roles"] = ROLES
                self._json(HTTPStatus.OK, payload)
                return
            if parsed.path.startswith("/api/cases/"):
                blind_id = unquote(parsed.path.removeprefix("/api/cases/"))
                self._json(HTTPStatus.OK, self.store.get_case(blind_id))
                return
            if parsed.path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "role": self.store.role})
                return
            self._serve_static(parsed.path)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Case not found.")
        except PermissionError as error:
            self._error(HTTPStatus.FORBIDDEN, str(error))
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/labels/"):
            self._error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64_000:
                raise ValueError("Invalid request body.")
            payload = json.loads(self.rfile.read(length))
            blind_id = unquote(parsed.path.removeprefix("/api/labels/"))
            result = self.store.save(
                blind_id,
                str(payload.get("label", "")),
                str(payload.get("rationale", "")),
            )
            self._json(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            self._error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON.")
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Case not found.")
        except PermissionError as error:
            self._error(HTTPStatus.FORBIDDEN, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def _serve_static(self, request_path: str) -> None:
        relative = (
            "index.html"
            if request_path in ("", "/")
            else unquote(request_path.lstrip("/"))
        )
        path = (self.static_dir / relative).resolve()
        if self.static_dir not in path.parents or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", content_types.get(path.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)


def parse_args() -> argparse.Namespace:
    default_package = (
        Path(__file__).resolve().parents[1]
        / "adjudication_packages"
        / "exp_indep_final_static_blind_01"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=SERVER_ROLES, required=True)
    parser.add_argument("--package", type=Path, default=default_package)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    if args.role == "all":
        server.stores = {role: AnnotationStore(args.package, role) for role in ROLES}  # type: ignore[attr-defined]
    else:
        server.store = AnnotationStore(args.package, args.role)  # type: ignore[attr-defined]
    server.static_dir = Path(__file__).resolve().parent / "static"  # type: ignore[attr-defined]
    print(
        f"HapRepair annotation server: http://{args.host}:{args.port} ({args.role})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
