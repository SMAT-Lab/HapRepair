#!/usr/bin/env python3
"""Install exact OpenHarmony SDK components needed by EXP-AGENT-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATALOG_URL = "https://repo.harmonyos.com/sdkmanager/v5/ohos/getSdkList"
DEFAULT_ROOT = Path(
    "/home/zhihao/hdd/haprepair/baseline_data/openharmony_sdk"
)
DEFAULT_APIS = ("8", "12", "18", "20", "23", "26.0.0")
DEFAULT_COMPONENTS = ("ets", "js", "toolchains")
CATALOG_BY_API = {
    "8": "4.0-ohos-14",
    "12": "26.0-ohos-single-1",
    "18": "26.0-ohos-single-1",
    "20": "26.0-ohos-single-1",
    "23": "26.0-ohos-single-1",
    "26.0.0": "26.0-ohos-single-1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_catalog(support_version: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        CATALOG_URL,
        data=json.dumps(
            {
                "osType": "linux",
                "osArch": "x64",
                "supportVersion": support_version,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected SDK catalog response for {support_version}")
    return payload


def select_entry(
    catalog: list[dict[str, Any]], api: str, component: str
) -> dict[str, Any]:
    matches = [
        item
        for item in catalog
        if str(item.get("apiVersion")) == api
        and item.get("path") == component
        and item.get("archive", {}).get("osArch") in {None, "x64"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one x64 catalog entry for API {api} {component}, got {len(matches)}"
        )
    return matches[0]


def effective_download_url(entry: dict[str, Any]) -> str:
    # Older catalogs still publish the slower .com alias for the same signed
    # CDN object. The .cn alias supports reliable range requests.
    return entry["archive"]["url"].replace(
        "contentcenter-drcn.dbankcdn.com/",
        "contentcenter-drcn.dbankcdn.cn/",
    )


def download(entry: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "aria2c",
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        "--max-tries=5",
        "--retry-wait=2",
        f"--dir={destination.parent}",
        f"--out={destination.name}",
        effective_download_url(entry),
    ]
    subprocess.run(command, check=True)


def inspect_installed(destination: Path, entry: dict[str, Any]) -> bool:
    manifest = destination / "oh-uni-package.json"
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    published_api_versions = {
        str(value)
        for key in ("apiVersion", "fullApiVersion", "platformVersion")
        if (value := payload.get(key)) is not None
    }
    return (
        str(entry["apiVersion"]) in published_api_versions
        and payload.get("path") == entry["path"]
        and payload.get("version") == entry["version"]
    )


def install_archive(
    archive: Path, destination: Path, entry: dict[str, Any]
) -> None:
    if destination.exists():
        if inspect_installed(destination, entry):
            return
        raise RuntimeError(f"Refusing to overwrite incompatible SDK path: {destination}")
    with tempfile.TemporaryDirectory(
        prefix=f".{entry['path']}-", dir=destination.parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temp_dir)
            restore_archive_permissions(handle, temp_dir)
        manifests = list(temp_dir.rglob("oh-uni-package.json"))
        if len(manifests) != 1:
            raise RuntimeError(
                f"Expected one oh-uni-package.json in {archive}, got {len(manifests)}"
            )
        component_dir = manifests[0].parent
        if not inspect_installed(component_dir, entry):
            raise RuntimeError(f"Extracted SDK metadata does not match catalog: {archive}")
        shutil.copytree(component_dir, destination, symlinks=True)


def restore_archive_permissions(
    archive: zipfile.ZipFile, root: Path, strip_prefix: str | None = None
) -> None:
    """Restore Unix modes and links that ZipFile extraction does not preserve."""
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if strip_prefix is not None:
            if not parts or parts[0] != strip_prefix:
                continue
            parts = parts[1:]
        if not parts:
            continue
        target = root.joinpath(*parts)
        archived_mode = member.external_attr >> 16
        if stat.S_ISLNK(archived_mode):
            link_target = archive.read(member).decode("utf-8")
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(link_target)
            continue
        mode = archived_mode & 0o777
        if mode and target.exists():
            target.chmod(mode)


def repair_installed_permissions(
    archive: Path, destination: Path, entry: dict[str, Any]
) -> None:
    with zipfile.ZipFile(archive) as handle:
        restore_archive_permissions(handle, destination, entry["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--api", action="append", dest="apis")
    parser.add_argument("--component", action="append", dest="components")
    parser.add_argument("--manifest-name", default="install_manifest.json")
    args = parser.parse_args()
    root = args.root.resolve()
    apis = tuple(args.apis or DEFAULT_APIS)
    components = tuple(args.components or DEFAULT_COMPONENTS)
    unknown = sorted(set(apis) - set(CATALOG_BY_API))
    if unknown:
        raise SystemExit(f"No frozen catalog mapping for APIs: {unknown}")

    downloads = root / "downloads"
    sdk_root = root / "openharmony"
    catalogs: dict[str, list[dict[str, Any]]] = {}
    installed = []
    for api in apis:
        support_version = CATALOG_BY_API[api]
        catalog = catalogs.setdefault(support_version, fetch_catalog(support_version))
        for component in components:
            entry = select_entry(catalog, api, component)
            checksum = entry["archive"]["checksum"]
            archive = downloads / f"api-{api}-{component}-{checksum[:12]}.zip"
            destination = sdk_root / api / component
            if inspect_installed(destination, entry):
                status = "already_installed"
            else:
                if not archive.is_file() or sha256_file(archive) != checksum:
                    if archive.exists():
                        archive.unlink()
                    download(entry, archive)
                observed = sha256_file(archive)
                if observed != checksum:
                    raise RuntimeError(
                        f"Checksum mismatch for {archive}: {observed} != {checksum}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                install_archive(archive, destination, entry)
                status = "installed"
            if archive.is_file() and sha256_file(archive) == checksum:
                repair_installed_permissions(archive, destination, entry)
            installed.append(
                {
                    "api_version": api,
                    "component": component,
                    "version": entry["version"],
                    "support_version": support_version,
                    "archive_url": entry["archive"]["url"],
                    "effective_download_url": effective_download_url(entry),
                    "archive_size": int(entry["archive"]["size"]),
                    "archive_sha256": checksum,
                    "destination": str(destination),
                    "status": status,
                }
            )
            print(f"[{status}] API {api} {component}", flush=True)

    manifest = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "created_at": utc_now(),
        "catalog_url": CATALOG_URL,
        "host": platform.platform(),
        "machine": platform.machine(),
        "sdk_root": str(sdk_root),
        "requested_apis": list(apis),
        "requested_components": list(components),
        "components": installed,
    }
    manifest_path = root / args.manifest_name
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
