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


# ---------------------------------------------------------------- Part D: variable elimination
# A factor is a pair (variables, table). `variables` is a tuple of names and
# `table` maps a tuple of values (one per variable, same order) to a number.
from itertools import product


def _cpt_to_factor(bn, name):
    node = bn.nodes[name]
    variables = tuple(node.parents) + (name,)
    table = {}
    for parent_values, row in node.cpt.items():
        for value, p in row.items():
            table[tuple(parent_values) + (value,)] = p
    return variables, table


def _restrict(factor, evidence):
    """Keep only the rows that agree with the evidence, and drop those variables."""
    variables, table = factor
    keep = [i for i, v in enumerate(variables) if v not in evidence]
    new_variables = tuple(variables[i] for i in keep)
    new_table = {}
    for assignment, p in table.items():
        if all(assignment[i] == evidence[v] for i, v in enumerate(variables) if v in evidence):
            new_table[tuple(assignment[i] for i in keep)] = p
    return new_variables, new_table


def _multiply(bn, f1, f2):
    vars1, table1 = f1
    vars2, table2 = f2
    variables = vars1 + tuple(v for v in vars2 if v not in vars1)
    table = {}
    for combo in product(*[bn.nodes[v].values for v in variables]):
        a = dict(zip(variables, combo))
        table[combo] = table1[tuple(a[v] for v in vars1)] * table2[tuple(a[v] for v in vars2)]
    return variables, table


def _sum_out(var, factor):
    variables, table = factor
    i = variables.index(var)
    new_variables = variables[:i] + variables[i + 1:]
    new_table = {}
    for assignment, p in table.items():
        key = assignment[:i] + assignment[i + 1:]
        new_table[key] = new_table.get(key, 0.0) + p
    return new_variables, new_table


def variable_elimination_ask(bn, query, evidence=None):
    """P(query | evidence) by variable elimination. Same answers as enumeration_ask."""
    evidence = dict(evidence or {})
    _check_query(bn, query, evidence)

    factors = [_restrict(_cpt_to_factor(bn, name), evidence) for name in bn.nodes]

    hidden = [n for n in bn.nodes if n != query and n not in evidence]
    for var in reversed(hidden):  # children first
        related = [f for f in factors if var in f[0]]
        others = [f for f in factors if var not in f[0]]
        product_factor = related[0]
        for f in related[1:]:
            product_factor = _multiply(bn, product_factor, f)
        factors = others + [_sum_out(var, product_factor)]

    result = factors[0]
    for f in factors[1:]:
        result = _multiply(bn, result, f)
    return _normalize({key[0]: p for key, p in result[1].items()})
