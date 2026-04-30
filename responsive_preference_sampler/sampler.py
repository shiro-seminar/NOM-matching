"""
Responsive preference sampler.

Samples bundle orderings (weak orders on k-subsets of O) that are
consistent with a given marginal preference over individual items.

Theory
------
A bundle preference R on X = {X in 2^O : |X| = k} is *responsive* to a
marginal preference >= on O if, for any o, p in O and any Q in X with o in Q:

    o >= p  iff  Q R (Q minus {o}) union {p}

Algorithm
---------
1. Enumerate all C(m, k) bundles.
2. Build DAG: edge Q -> Q' for each single "improving swap"
   (replace o in Q with p not-in Q where rank(o) < rank(p)).
3. Group bundles into indifference classes by rank-multiset.
   (Two bundles with the same rank-multiset are indifferent under ALL
   responsive preferences.)
4. Build quotient DAG over indifference classes (guaranteed acyclic).
5. Random topological sort on the quotient DAG; shuffle bundles within
   each class uniformly.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional

import numpy as np


def sample_responsive_preference(
    marginal_pref: Dict,
    k: int,
    num_samples: int,
    seed: Optional[int] = None,
) -> List[List[FrozenSet]]:
    """
    Sample bundle orderings responsive to marginal_pref.

    Args:
        marginal_pref: item -> rank (int), smaller = more preferred.
                       Equal ranks mean indifference.
                       Example: {0: 0, 1: 0, 2: 1, 3: 2}
        k:            bundle size.
        num_samples:  number of orderings to draw.
        seed:         numpy RNG seed for reproducibility.

    Returns:
        List of orderings; each ordering is a list of frozensets from
        most preferred to least preferred.  Bundles in the same
        indifference class appear in consecutive positions.
    """
    rng = np.random.default_rng(seed)
    items = sorted(marginal_pref.keys())

    # ── 1. Enumerate k-subsets ───────────────────────────────────────────
    bundles: List[FrozenSet] = [frozenset(c) for c in combinations(items, k)]
    n = len(bundles)
    b2i: Dict[FrozenSet, int] = {b: i for i, b in enumerate(bundles)}

    # ── 2. Build strict-preference DAG ───────────────────────────────────
    # Edge i → j: replace item o ∈ bundle_i with item p ∉ bundle_i,
    # where rank(o) < rank(p)  (o is strictly better, bundle gets worse).
    children: List[List[int]] = [[] for _ in range(n)]
    seen_edges: set = set()

    for i, Q in enumerate(bundles):
        for o in sorted(Q):
            for p in items:
                if p in Q:
                    continue
                if marginal_pref[o] < marginal_pref[p]:
                    Q2 = (Q - {o}) | frozenset([p])
                    j = b2i.get(Q2)
                    if j is not None and (i, j) not in seen_edges:
                        children[i].append(j)
                        seen_edges.add((i, j))

    # ── 3. Indifference classes (same rank-multiset) ─────────────────────
    def rank_ms(bundle: FrozenSet) -> tuple:
        return tuple(sorted(marginal_pref[x] for x in bundle))

    ms_to_cls: Dict[tuple, int] = {}
    class_of: List[int] = [0] * n
    classes: List[List[int]] = []

    for i, b in enumerate(bundles):
        ms = rank_ms(b)
        if ms not in ms_to_cls:
            ms_to_cls[ms] = len(classes)
            classes.append([])
        cid = ms_to_cls[ms]
        class_of[i] = cid
        classes[cid].append(i)

    nc = len(classes)

    # ── 4. Quotient DAG over indifference classes ─────────────────────────
    # Guaranteed acyclic: rank-sum strictly increases along edges.
    cls_children_set: List[set] = [set() for _ in range(nc)]
    for i in range(n):
        ci = class_of[i]
        for j in children[i]:
            cj = class_of[j]
            if ci != cj:
                cls_children_set[ci].add(cj)

    cls_children: List[List[int]] = [sorted(s) for s in cls_children_set]
    cls_in_deg: List[int] = [0] * nc
    for ci in range(nc):
        for cj in cls_children[ci]:
            cls_in_deg[cj] += 1

    # ── 5. Random topological sort (repeated num_samples times) ──────────
    results: List[List[FrozenSet]] = []

    for _ in range(num_samples):
        cur_in_deg = list(cls_in_deg)
        available = [ci for ci in range(nc) if cur_in_deg[ci] == 0]
        ordering: List[FrozenSet] = []

        while available:
            # Uniform random choice from available (in-degree-0) classes
            idx = int(rng.integers(len(available)))
            chosen = available[idx]
            available[idx] = available[-1]   # swap-and-pop O(1)
            available.pop()

            # Randomly order bundles within the chosen indifference class
            class_bundle_indices = list(classes[chosen])
            rng.shuffle(class_bundle_indices)
            ordering.extend(bundles[bi] for bi in class_bundle_indices)

            for next_c in cls_children[chosen]:
                cur_in_deg[next_c] -= 1
                if cur_in_deg[next_c] == 0:
                    available.append(next_c)

        results.append(ordering)

    return results


def verify_responsiveness(
    ordering: List[FrozenSet],
    marginal_pref: Dict,
) -> None:
    """
    Assert that ordering is consistent with responsiveness.

    For each adjacent pair (Q at position i, Q' at position i+1),
    if they differ by exactly one item — Q' = (Q minus {o}) union {p} —
    then responsiveness requires o >= p, i.e. rank(o) <= rank(p).

    Raises AssertionError on the first violation found.
    """
    for i in range(len(ordering) - 1):
        Q = ordering[i]          # more preferred
        Q_next = ordering[i + 1]
        removed = Q - Q_next    # item in Q but not Q_next
        added = Q_next - Q      # item in Q_next but not Q

        if len(removed) == 1 and len(added) == 1:
            o = next(iter(removed))
            p = next(iter(added))
            assert marginal_pref[o] <= marginal_pref[p], (
                f"Responsiveness violation at position {i}: "
                f"Q={set(Q)} preferred to Q'={set(Q_next)}, "
                f"but rank({o})={marginal_pref[o]} > rank({p})={marginal_pref[p]}"
            )
