"""Learn the CPTs of a Bayesian network from data (maximum likelihood with Laplace smoothing)."""
from itertools import product


def learn_cpts(bn, data, alpha=1.0):
    """Fill in every node's CPT by counting rows of `data`.

    data  : list of dicts, each giving a value for every node, e.g.
            {"Burglary": False, "Earthquake": False, "Alarm": False, ...}
    alpha : Laplace smoothing. Each outcome starts with a count of alpha, so
            combinations never seen in the data still get a small probability.
    """
    if alpha < 0:
        raise ValueError("alpha cannot be negative")
    if not data:
        raise ValueError("data is empty, there is nothing to learn from")
    for i, row in enumerate(data):
        for name, node in bn.nodes.items():
            if name not in row:
                raise ValueError(f"data row {i} has no value for '{name}'")
            if row[name] not in node.values:
                raise ValueError(f"data row {i}: '{row[name]}' is not a possible value of '{name}'")

    for name, node in bn.nodes.items():
        counts = {
            combo: {v: alpha for v in node.values}
            for combo in product(*[bn.nodes[p].values for p in node.parents])
        }
        for row in data:
            combo = tuple(row[p] for p in node.parents)
            counts[combo][row[name]] += 1
        table = {}
        for combo, row_counts in counts.items():
            total = sum(row_counts.values())
            if total == 0:
                raise ValueError(
                    f"no data for '{name}' given parents {combo}; use alpha > 0 to smooth"
                )
            table[combo] = {v: c / total for v, c in row_counts.items()}
        node.cpt = table
    return bn


def sample(bn, n, rng=None):
    """Draw n random complete examples from the network (parents are sampled before children)."""
    import random

    rng = rng or random.Random()
    rows = []
    for _ in range(n):
        row = {}
        for name, node in bn.nodes.items():
            probs = node.cpt[tuple(row[p] for p in node.parents)]
            values = list(probs)
            row[name] = rng.choices(values, weights=[probs[v] for v in values])[0]
        rows.append(row)
    return rows
