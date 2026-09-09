#!/usr/bin/env python3

from __future__ import annotations

import unittest

from freeze_sample import allocate_with_minimum, build_samples, read_jsonl, POPULATION_PATH


class AllocationTests(unittest.TestCase):
    def test_hamilton_allocation_preserves_minimum_and_target(self) -> None:
        observed = allocate_with_minimum({"a": 10, "b": 5, "c": 1}, 9, 2)
        self.assertEqual(sum(observed.values()), 9)
        self.assertEqual(observed["c"], 1)
        self.assertGreaterEqual(observed["a"], 2)
        self.assertGreaterEqual(observed["b"], 2)

    def test_allocation_is_deterministic(self) -> None:
        populations = {"z": 100, "a": 100, "m": 100}
        self.assertEqual(
            allocate_with_minimum(populations, 20, 3),
            allocate_with_minimum(populations, 20, 3),
        )

    def test_invalid_minimum_fails(self) -> None:
        with self.assertRaises(ValueError):
            allocate_with_minimum({"a": 5, "b": 5}, 3, 2)


class FrozenPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.population = read_jsonl(POPULATION_PATH)
        cls.sample, cls.allocation = build_samples(cls.population)

    def test_frozen_sample_sizes_and_rule_coverage(self) -> None:
        performance = [item for item in self.sample if item["category"] == "performance"]
        security = [item for item in self.sample if item["category"] == "security"]
        self.assertEqual(len(performance), 170)
        self.assertEqual(len(security), 61)
        self.assertEqual(len({item["rule"] for item in performance}), 38)
        self.assertEqual(len({item["rule"] for item in security}), 3)

    def test_sample_is_unique_and_has_estimable_strata(self) -> None:
        self.assertEqual(len(self.sample), len({item["alert_id"] for item in self.sample}))
        for row in self.allocation["strata"]:
            self.assertTrue(row["sample"] > 1 or row["sample"] == row["population"])

    def test_sampling_is_reproducible(self) -> None:
        repeated, repeated_allocation = build_samples(self.population)
        self.assertEqual(
            [item["alert_id"] for item in self.sample],
            [item["alert_id"] for item in repeated],
        )
        self.assertEqual(self.allocation, repeated_allocation)


if __name__ == "__main__":
    unittest.main()
