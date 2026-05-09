"""Domain specifications for ordinal preference experiments.

Each domain restricts how agents can rank items relative to their endowment.
  epsilon(k) = 1: owned items may appear in indifference class k
  nu(k)      = 1: unowned items may appear in indifference class k

Class indices are 1-based (k=1 is the top/most-preferred class).
Rank values in code are 0-based: rank r corresponds to class k = r+1.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainSpec:
    name: str
    num_ranks: int          # total number of indifference classes R
    owned_ranks: tuple[int, ...]    # rank values (0-based) allowed for owned items
    unowned_ranks: tuple[int, ...]  # rank values (0-based) allowed for unowned items
    strict: bool = False    # True => no ties; sampling draws a permutation of 0..m-1


# ---------------------------------------------------------------------------
# Domain catalogue
# ---------------------------------------------------------------------------

DOMAINS: dict[str, DomainSpec] = {
    # Manjunath-Westkamp (2025) trichotomous domain
    # epsilon(1)=epsilon(2)=1, epsilon(3)=0
    # nu(1)=nu(2)=nu(3)=1
    # Owned items: top or middle class only (rank 0 or 1)
    # Unowned items: any of 3 classes (rank 0, 1, or 2)
    "trichotomous": DomainSpec(
        name="trichotomous",
        num_ranks=3,
        owned_ranks=(0, 1),
        unowned_ranks=(0, 1, 2),
    ),

    # Extended: owned items may also appear in class 3
    # epsilon(1)=epsilon(2)=epsilon(3)=1
    # nu(1)=nu(2)=nu(3)=nu(4)=1
    # 4 indifference classes; owned: ranks 0-2; unowned: ranks 0-3
    "trichotomous_extended_e3": DomainSpec(
        name="trichotomous_extended_e3",
        num_ranks=4,
        owned_ranks=(0, 1, 2),
        unowned_ranks=(0, 1, 2, 3),
    ),

    # Four indifference classes, no class restrictions
    # epsilon(1)=..=epsilon(4)=1; nu(1)=..=nu(4)=1
    "four_chotomous_e4": DomainSpec(
        name="four_chotomous_e4",
        num_ranks=4,
        owned_ranks=(0, 1, 2, 3),
        unowned_ranks=(0, 1, 2, 3),
    ),

    # Strict total order: every item in a distinct class, no ties
    # Sampling: a random permutation of 0..m-1 per agent
    "strict": DomainSpec(
        name="strict",
        num_ranks=4,   # m=4 items => ranks 0,1,2,3
        owned_ranks=(0, 1, 2, 3),
        unowned_ranks=(0, 1, 2, 3),
        strict=True,
    ),
}
