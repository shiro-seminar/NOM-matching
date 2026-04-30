# Responsive Preference Sampler

Samples bundle orderings (weak orders on k-element subsets of an item set O)
that are **responsive** to a given marginal preference over individual items.

## Theory

Let O = {o₁, …, oₘ} be a set of items.
A **marginal preference** ≿ is a weak order on O.
A **bundle preference** R on X = {X ⊆ O : |X| = k} is **responsive** to ≿ if:

> For any o, p ∈ O and any Q ∈ X with o ∈ Q:  
> o ≿ p  ⟺  Q R (Q \ {o}) ∪ {p}

That is, replacing an item with a weakly better item makes the bundle weakly better.

### Indifference classes

Two bundles are indifferent under **all** responsive preferences if and only if
they have the same multiset of item ranks.  Swapping two equally-ranked items
never changes bundle preference, so such bundles are indistinguishable
regardless of the particular responsive preference chosen.

## Algorithm

1. Enumerate all C(m, k) bundles.
2. Build a DAG: add edge Q → Q' when Q' = (Q \ {o}) ∪ {p} with rank(o) < rank(p)
   — one "improving swap" makes Q strictly better than Q'.
3. Group bundles into **indifference classes** by rank-multiset.
4. Build the **quotient DAG** over indifference classes (guaranteed acyclic:
   rank-sum strictly increases along every edge).
5. **Random topological sort**: at each step pick uniformly from the set of
   available (in-degree-0) classes; shuffle bundles within each class uniformly.
6. Repeat `num_samples` times using `numpy.random.default_rng` for reproducibility.

The output orderings are exactly the linear extensions of the partial order
induced by responsiveness, with bundles in the same indifference class kept
adjacent.

## Usage

```python
from sampler import sample_responsive_preference, verify_responsiveness

# o0 ~ o1 ≻ o2 ≻ o3  (ranks: smaller = more preferred)
marginal = {0: 0, 1: 0, 2: 1, 3: 2}
k = 2

orderings = sample_responsive_preference(marginal, k, num_samples=5, seed=42)

for ordering in orderings:
    print([set(b) for b in ordering])
    verify_responsiveness(ordering, marginal)  # raises if violated
```

### Example output

```
[{0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}]
[{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]
...
```

`{0,1}` is always first (unique best class), `{2,3}` is always last (unique worst class).
The middle four bundles vary across samples.

## Files

| File | Description |
|---|---|
| `sampler.py` | Main implementation: `sample_responsive_preference`, `verify_responsiveness` |
| `test_sampler.py` | pytest test suite (5 test classes, 3+ distinct marginal preferences) |
| `requirements.txt` | Dependencies |

## Running Tests

```bash
pip install -r requirements.txt
pytest test_sampler.py -v
```
