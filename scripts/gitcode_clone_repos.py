#!/usr/bin/env python3
"""Clone GitCode organization repositories in bulk."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://gitcode.com/api/v5"


def fetch_repos(org: str, token: str, per_page: int = 100) -> List[Dict]:
    page = 1
    repos: List[Dict] = []
    while True:
        params = {
            "per_page": per_page,
            "page": page,
            "access_token": token,
        }
        url = f"{API_BASE}/orgs/{org}/repos"
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch repos for {org} (status {resp.status_code}): {resp.text}"
            )
        batch = resp.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1
    return repos


def clone_repo(repo: Dict, org_dir: Path) -> None:
    ns_path = repo["namespace"]["path"]
    repo_path = repo["path"]
    url = f"https://gitcode.com/{ns_path}/{repo_path}.git"
    target = org_dir / repo_path
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {url} -> {target}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"git clone failed for {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone GitCode org repositories")
    parser.add_argument(
        "--org",
        action="append",
        help="Organization name(s). Default: OpenHarmony, OpenHarmony-TPC, OpenHarmony-SIG",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/LLMCodeRepair/repo_new"),
        help="Root directory to place clones (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("GITCODE_TOKEN"),
        help="GitCode personal access token. Defaults to $GITCODE_TOKEN",
    )
    parser.add_argument(
        "--max-per-org",
        type=int,
        default=0,
        help="Maximum number of repositories to clone per org (0 means no limit, default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent clone workers (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.token:
        print("GITCODE_TOKEN is required", file=sys.stderr)
        return 1

    orgs = args.org or ["OpenHarmony", "OpenHarmony-TPC", "OpenHarmony-SIG"]
    root = args.root

    for org in orgs:
        print(f"=== Processing {org} ===")
        repos = fetch_repos(org, args.token)
        print(f"Fetched {len(repos)} repositories for {org}")
        org_dir = root / org
        subset = repos if args.max_per_org <= 0 else repos[: args.max_per_org]
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for repo in subset:
                futures.append(executor.submit(clone_repo, repo, org_dir))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"[warn] clone task failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
