"""Exact inference in a Bayesian network."""


# ---------------------------------------------------------------- helpers
def _normalize(dist):
    total = sum(dist.values())
    if total == 0:
        raise ValueError("the evidence has probability zero, so nothing can be concluded from it")
    return {value: p / total for value, p in dist.items()}


def _check_query(bn, query, evidence):
    if query not in bn.nodes:
        raise ValueError(f"unknown query variable '{query}'")
    if query in evidence:
        raise ValueError(f"'{query}' cannot be both the query and part of the evidence")
    for var, value in evidence.items():
        if var not in bn.nodes:
            raise ValueError(f"unknown evidence variable '{var}'")
        if value not in bn.nodes[var].values:
            raise ValueError(f"'{value}' is not a possible value of '{var}'")


# ---------------------------------------------------------------- Part C: enumeration
def enumeration_ask(bn, query, evidence=None):
    """P(query | evidence) by brute-force enumeration (Russell & Norvig, ENUMERATION-ASK).

    For each value of the query variable, add up the joint probability over every
    possible value of the hidden variables, then normalise so the answers sum to 1."""
    evidence = dict(evidence or {})
    _check_query(bn, query, evidence)
    distribution = {}
    for value in bn.nodes[query].values:
        distribution[value] = _enumerate_all(bn, list(bn.nodes), {**evidence, query: value})
    return _normalize(distribution)


def _enumerate_all(bn, variables, assignment):
    """Sum of the joint probability over all hidden variables in `variables`."""
    if not variables:
        return 1.0
    first, rest = variables[0], variables[1:]
    if first in assignment:  # query or evidence variable: its value is fixed
        p = bn.conditional_probability(first, assignment[first], assignment)
        return p * _enumerate_all(bn, rest, assignment)
    total = 0.0  # hidden variable: add up over each of its values
    for value in bn.nodes[first].values:
        p = bn.conditional_probability(first, value, assignment)
        total += p * _enumerate_all(bn, rest, {**assignment, first: value})
    return total
