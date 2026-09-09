from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_adjudication_v14_static_skill.py")
SPEC = importlib.util.spec_from_file_location(
    "prepare_adjudication_v14_static_skill", SCRIPT
)
assert SPEC and SPEC.loader
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class FinalV14AdjudicationPackageTest(unittest.TestCase):
    def test_real_run_builds_deterministic_blind_60_plus_3_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first" / "package"
            second = Path(temporary) / "second" / "package"
            first_result = package.build_package(
                run_id=package.EXPECTED_RUN_ID, package_dir=first
            )
            second_result = package.build_package(
                run_id=package.EXPECTED_RUN_ID, package_dir=second
            )
            self.assertTrue(first_result["all_passed"])
            self.assertTrue(second_result["all_passed"])
            self.assertEqual(first_result["human_candidate_count"], 60)
            self.assertEqual(first_result["automatic_generation_failure_count"], 3)
            first_coordinator = package.read_json(first / "coordinator_manifest.json")
            second_coordinator = package.read_json(second / "coordinator_manifest.json")
            self.assertEqual(
                first_coordinator["judge_bundle_immutable_tree_sha256"],
                second_coordinator["judge_bundle_immutable_tree_sha256"],
            )

    def test_existing_package_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                package.build_package(
                    run_id=package.EXPECTED_RUN_ID, package_dir=destination
                )


if __name__ == "__main__":
    unittest.main()
