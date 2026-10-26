import random
import pytest

from bbn.networks import alarm_network, sprinkler_network
from bbn.inference import enumeration_ask, variable_elimination_ask
from bbn.learning import learn_cpts, sample

BURGLARY_EVIDENCE = {"JohnCalls": True, "MaryCalls": True}


def close(dist, expected, tol=1e-4):
    return all(abs(dist[k] - expected[k]) < tol for k in expected)


# ------------------------------------------------------------ networks
def test_networks_are_valid():
    assert alarm_network().validate()
    assert sprinkler_network().validate()


# ------------------------------------------------------------ inference
def test_enumeration_textbook_answer():
    d = enumeration_ask(alarm_network(), "Burglary", BURGLARY_EVIDENCE)
    assert close(d, {True: 0.2842, False: 0.7158})


def test_explaining_away():
    bn = alarm_network()
    alarm_only = enumeration_ask(bn, "Burglary", {"Alarm": True})[True]
    with_quake = enumeration_ask(bn, "Burglary", {"Alarm": True, "Earthquake": True})[True]
    assert with_quake < alarm_only


def test_sprinkler_answers():
    s = sprinkler_network()
    assert close(enumeration_ask(s, "Rain", {"GrassWet": True}), {True: 0.3577, False: 0.6423})
    assert close(enumeration_ask(s, "Sprinkler", {"GrassWet": True}), {True: 0.6467, False: 0.3533})


def test_answers_sum_to_one():
    d = variable_elimination_ask(alarm_network(), "Burglary", BURGLARY_EVIDENCE)
    assert abs(sum(d.values()) - 1) < 1e-9


@pytest.mark.parametrize("make", [alarm_network, sprinkler_network])
def test_variable_elimination_matches_enumeration(make):
    bn = make()
    for q in bn.nodes:
        cases = [{}] + [{e: v} for e in bn.nodes if e != q for v in bn.nodes[e].values]
        for ev in cases:
            a = enumeration_ask(bn, q, ev)
            b = variable_elimination_ask(bn, q, ev)
            assert all(abs(a[k] - b[k]) < 1e-9 for k in a)


@pytest.mark.parametrize("ask", [enumeration_ask, variable_elimination_ask])
def test_bad_queries_raise(ask):
    bn = alarm_network()
    with pytest.raises(ValueError):
        ask(bn, "Burglary", {"Burglary": True})
    with pytest.raises(ValueError):
        ask(bn, "Nope", {})
    with pytest.raises(ValueError):
        ask(bn, "Burglary", {"Alarm": "maybe"})


# ------------------------------------------------------------ learning
def test_sample_has_every_node():
    bn = alarm_network()
    rows = sample(bn, 10, random.Random(1))
    assert len(rows) == 10
    assert all(set(r) == set(bn.nodes) for r in rows)


def test_learning_recovers_tables():
    true_bn = sprinkler_network()
    data = sample(true_bn, 50000, random.Random(0))
    learned = learn_cpts(sprinkler_network(), data)
    assert learned.validate()
    for name, node in true_bn.nodes.items():
        for combo, row in node.cpt.items():
            for value, p in row.items():
                assert abs(learned.nodes[name].cpt[combo][value] - p) < 0.03


def test_smoothing_keeps_unseen_rows_valid():
    data = sample(alarm_network(), 5, random.Random(0))
    assert learn_cpts(alarm_network(), data).validate()


def test_learning_rejects_bad_data():
    with pytest.raises(ValueError):
        learn_cpts(alarm_network(), [])
    with pytest.raises(ValueError):
        learn_cpts(alarm_network(), [{"Burglary": True}])
    with pytest.raises(ValueError):
        learn_cpts(alarm_network(), [{"Burglary": True}], alpha=-1)
