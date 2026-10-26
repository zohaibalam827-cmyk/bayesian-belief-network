"""The data structures of a Bayesian Belief Network: nodes, CPTs and the network itself."""
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Tuple


@dataclass
class Node:
    name: str
    values: List[Any]                 # the values this variable can take, e.g. [True, False]
    parents: List[str]                # names of the parent nodes (empty for a root node)
    cpt: Dict[Tuple, Dict[Any, float]] = field(default_factory=dict)
    # cpt[(parent values...)][value] = P(node = value | parents = those values)


class BayesianNetwork:
    """A directed acyclic graph where every node holds a conditional probability table.

    Rule: a node can only be added AFTER its parents. That makes cycles impossible and
    means the order in which nodes were added is already a valid topological order."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    # ---------------------------------------------------------------- building
    def add_node(self, name, values, parents=()):
        if name in self.nodes:
            raise ValueError(f"node '{name}' already exists")
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"parent '{p}' of '{name}' must be added first")
        self.nodes[name] = Node(name, list(values), list(parents))
        return self

    def set_cpt(self, name, table):
        """table maps a tuple of parent values to {value: probability}.
        A root node uses the empty tuple () as its only key."""
        node = self.nodes[name]
        node.cpt = {tuple(k): dict(v) for k, v in table.items()}
        return self

    def set_boolean_cpt(self, name, p_true):
        """Shortcut for True/False nodes: p_true maps parent values to P(node = True)."""
        table = {tuple(k): {True: p, False: 1 - p} for k, p in p_true.items()}
        return self.set_cpt(name, table)

    def validate(self):
        """Check every CPT has a row for every parent combination and each row sums to 1."""
        for node in self.nodes.values():
            combos = product(*[self.nodes[p].values for p in node.parents])
            for combo in combos:
                if combo not in node.cpt:
                    raise ValueError(f"'{node.name}' has no CPT row for parents {combo}")
                row = node.cpt[combo]
                if set(row) != set(node.values):
                    raise ValueError(f"'{node.name}' row {combo} must cover values {node.values}")
                if abs(sum(row.values()) - 1.0) > 1e-9:
                    raise ValueError(f"'{node.name}' row {combo} sums to {sum(row.values())}, not 1")
        return True

    # ---------------------------------------------------------------- probabilities
    def conditional_probability(self, name, value, assignment):
        """P(name = value | the parents' values taken from assignment)."""
        node = self.nodes[name]
        parent_values = tuple(assignment[p] for p in node.parents)
        return node.cpt[parent_values][value]

    def joint_probability(self, assignment):
        """P(full assignment) = product over all nodes of P(node | its parents)."""
        missing = set(self.nodes) - set(assignment)
        if missing:
            raise ValueError(f"assignment must give a value for every node, missing {sorted(missing)}")
        p = 1.0
        for name in self.nodes:
            p *= self.conditional_probability(name, assignment[name], assignment)
        return p

    # ---------------------------------------------------------------- display
    def describe(self):
        lines = []
        for node in self.nodes.values():
            parents = ", ".join(node.parents) if node.parents else "none"
            lines.append(f"{node.name}  (values: {node.values}, parents: {parents})")
            for combo, row in node.cpt.items():
                given = ", ".join(f"{p}={v}" for p, v in zip(node.parents, combo))
                probs = "  ".join(f"P({node.name}={v})={pr:.4g}" for v, pr in row.items())
                lines.append(f"    {('given ' + given) if given else 'prior'}: {probs}")
        return "\n".join(lines)
