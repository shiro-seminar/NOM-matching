"""Tests for the responsive preference sampler."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from itertools import combinations

import pytest

from sampler import sample_responsive_preference, verify_responsiveness


def all_bundles(items, k):
    return [frozenset(c) for c in combinations(items, k)]


# ── Case 1: strict total order  0 ≻ 1 ≻ 2 ≻ 3 ───────────────────────────────

class TestStrictTotalOrder:
    """marginal: {0:0, 1:1, 2:2, 3:3}, k=2"""

    MARGINAL = {0: 0, 1: 1, 2: 2, 3: 3}
    ITEMS = [0, 1, 2, 3]
    K = 2

    def test_sample_count(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=42)
        assert len(results) == 10

    def test_all_bundles_present(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        expected = set(frozenset(c) for c in combinations(self.ITEMS, self.K))
        for ordering in results:
            assert set(ordering) == expected

    def test_no_duplicates(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        for ordering in results:
            assert len(set(ordering)) == len(ordering)

    def test_top_bundle_fixed(self):
        """{0,1} is uniquely best — always first."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 20, seed=1)
        for ordering in results:
            assert ordering[0] == frozenset({0, 1})

    def test_bottom_bundle_fixed(self):
        """{2,3} is uniquely worst — always last."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 20, seed=1)
        for ordering in results:
            assert ordering[-1] == frozenset({2, 3})

    def test_responsiveness(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 30, seed=2)
        for ordering in results:
            verify_responsiveness(ordering, self.MARGINAL)

    def test_reproducibility(self):
        r1 = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=99)
        r2 = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=99)
        assert r1 == r2

    def test_diversity_in_middle(self):
        """Third position can be {0,3} or {1,2} — both valid after fixed {0,1},{0,2}."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 50, seed=3)
        third_positions = set(tuple(sorted(r[2])) for r in results)
        # After {0,1} and {0,2}, both {0,3} and {1,2} have in-degree 0
        assert len(third_positions) >= 2


# ── Case 2: all items indifferent ────────────────────────────────────────────

class TestAllEqual:
    """marginal: {0:0, 1:0, 2:0, 3:0}, k=2"""

    MARGINAL = {0: 0, 1: 0, 2: 0, 3: 0}
    ITEMS = [0, 1, 2, 3]
    K = 2

    def test_all_bundles_present(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        expected = set(frozenset(c) for c in combinations(self.ITEMS, self.K))
        for ordering in results:
            assert set(ordering) == expected

    def test_responsiveness(self):
        """All swaps are between equal-rank items: condition trivially satisfied."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 30, seed=4)
        for ordering in results:
            verify_responsiveness(ordering, self.MARGINAL)

    def test_ordering_diversity(self):
        """With 6 mutually indifferent bundles, first position should vary."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 60, seed=5)
        first_items = set(tuple(sorted(r[0])) for r in results)
        assert len(first_items) >= 3  # at least 3 of the 6 bundles appear first


# ── Case 3: mixed with ties  o0 ~ o1 ≻ o2 ≻ o3 ──────────────────────────────

class TestMixedWithTies:
    """marginal: {0:0, 1:0, 2:1, 3:2}, k=2"""

    MARGINAL = {0: 0, 1: 0, 2: 1, 3: 2}
    ITEMS = [0, 1, 2, 3]
    K = 2

    def test_sample_count(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=0)
        assert len(results) == 10

    def test_all_bundles_present(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        expected = set(frozenset(c) for c in combinations(self.ITEMS, self.K))
        for ordering in results:
            assert set(ordering) == expected

    def test_top_class_is_first(self):
        """{0,1} is uniquely best (rank-multiset (0,0)): must be first."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 20, seed=6)
        for ordering in results:
            assert ordering[0] == frozenset({0, 1})

    def test_bottom_bundle_is_last(self):
        """{2,3} is uniquely worst (rank-multiset (1,2)): must be last."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 20, seed=6)
        for ordering in results:
            assert ordering[-1] == frozenset({2, 3})

    def test_middle_varies(self):
        """Middle positions ({0,2},{1,2},{0,3},{1,3}) can appear in varied orders."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 50, seed=7)
        second_positions = set(tuple(sorted(r[1])) for r in results)
        assert len(second_positions) >= 2

    def test_responsiveness(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 30, seed=8)
        for ordering in results:
            verify_responsiveness(ordering, self.MARGINAL)

    def test_reproducibility(self):
        r1 = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=77)
        r2 = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=77)
        assert r1 == r2


# ── Case 4: k=1, singletons mirror marginal directly ─────────────────────────

class TestK1:
    """marginal: {0:0, 1:1, 2:2}, k=1"""

    MARGINAL = {0: 0, 1: 1, 2: 2}
    K = 1

    def test_unique_ordering(self):
        """With strict total order and k=1, ordering is fully determined."""
        results = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=0)
        expected = [frozenset({0}), frozenset({1}), frozenset({2})]
        for ordering in results:
            assert ordering == expected

    def test_responsiveness(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 10, seed=0)
        for ordering in results:
            verify_responsiveness(ordering, self.MARGINAL)


# ── Case 5: k=m, single bundle (full set) ────────────────────────────────────

class TestKEqualsM:
    """marginal: {0:0, 1:1, 2:2}, k=3"""

    MARGINAL = {0: 0, 1: 1, 2: 2}
    K = 3

    def test_single_bundle(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        for ordering in results:
            assert len(ordering) == 1
            assert ordering[0] == frozenset({0, 1, 2})

    def test_responsiveness(self):
        results = sample_responsive_preference(self.MARGINAL, self.K, 5, seed=0)
        for ordering in results:
            verify_responsiveness(ordering, self.MARGINAL)
