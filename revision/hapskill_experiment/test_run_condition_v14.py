#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_condition_v14 as condition


class OfflinePnpmTests(unittest.TestCase):
    def make_pnpm_source(self, root: Path) -> Path:
        source = root / "source-pnpm"
        (source / "bin").mkdir(parents=True)
        (source / "package.json").write_text(
            json.dumps(
                {
                    "name": "pnpm",
                    "version": "10.28.1",
                    "main": "bin/pnpm.cjs",
                }
            ),
            encoding="utf-8",
        )
        (source / "bin" / "pnpm.cjs").write_text("", encoding="utf-8")
        return source

    def test_stage_matches_hvigor_wrapper_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_pnpm_source(root)

            state_dir = root / "state"
            with patch.object(condition, "PNPM_SOURCE", source):
                staged = condition.stage_offline_pnpm(state_dir)

            tools = state_dir / "build-home" / ".hvigor" / "wrapper" / "tools"
            package = tools / "node_modules" / "pnpm"
            shim = tools / "node_modules" / ".bin" / "pnpm"
            self.assertTrue(package.is_dir())
            self.assertEqual(shim.readlink(), Path("../pnpm/bin/pnpm.cjs"))
            self.assertEqual(staged["container_home"], "/run-state/build-home")
            self.assertFalse(
                (state_dir / "build-home" / "wrapper" / "tools").exists()
            )

    def test_stage_copies_frozen_workspace_for_project_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_pnpm_source(root)
            host_hvigor_home = root / "preflight" / ".hvigor"
            host_workspace = (
                host_hvigor_home / "project_caches" / "cache-id" / "workspace"
            )
            host_node_modules = host_workspace / "node_modules"
            host_node_modules.mkdir(parents=True)
            (host_workspace / "package.json").write_text(
                '{"dependencies":{"@ohos/hvigor":"4.0.5"}}', encoding="utf-8"
            )
            (host_node_modules / "marker").write_text("frozen", encoding="utf-8")
            build_setup = {
                "build_command": ["bash", "hvigorw", "assembleHap"],
                "environment": {
                    "HVIGOR_USER_HOME": str(host_hvigor_home),
                    "NODE_PATH": str(host_node_modules),
                },
            }

            state_dir = root / "state"
            with patch.object(condition, "PNPM_SOURCE", source):
                staged = condition.stage_offline_pnpm(state_dir, build_setup)

            container_workspace = (
                state_dir
                / "build-home"
                / ".hvigor"
                / "project_caches"
                / "cache-id"
                / "workspace"
            )
            self.assertEqual(
                (container_workspace / "node_modules" / "marker").read_text(),
                "frozen",
            )
            self.assertEqual(
                staged["hvigor_workspace"],
                "/run-state/build-home/.hvigor/project_caches/cache-id/workspace",
            )


if __name__ == "__main__":
    unittest.main()
