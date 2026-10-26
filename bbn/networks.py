"""Two classic example networks with published numbers.

alarm_network     : the burglary/earthquake example from Russell & Norvig,
                    'Artificial Intelligence: A Modern Approach' (Bayesian networks chapter).
sprinkler_network : the rain/sprinkler/wet-grass example used in the Wikipedia
                    article 'Bayesian network'.
"""
from .network import BayesianNetwork

T, F = True, False


def alarm_network():
    bn = BayesianNetwork()
    bn.add_node("Burglary", [T, F])
    bn.add_node("Earthquake", [T, F])
    bn.add_node("Alarm", [T, F], parents=["Burglary", "Earthquake"])
    bn.add_node("JohnCalls", [T, F], parents=["Alarm"])
    bn.add_node("MaryCalls", [T, F], parents=["Alarm"])

    bn.set_boolean_cpt("Burglary", {(): 0.001})
    bn.set_boolean_cpt("Earthquake", {(): 0.002})
    bn.set_boolean_cpt("Alarm", {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
    bn.set_boolean_cpt("JohnCalls", {(T,): 0.90, (F,): 0.05})
    bn.set_boolean_cpt("MaryCalls", {(T,): 0.70, (F,): 0.01})
    bn.validate()
    return bn


def sprinkler_network():
    bn = BayesianNetwork()
    bn.add_node("Rain", [T, F])
    bn.add_node("Sprinkler", [T, F], parents=["Rain"])
    bn.add_node("GrassWet", [T, F], parents=["Sprinkler", "Rain"])

    bn.set_boolean_cpt("Rain", {(): 0.2})
    bn.set_boolean_cpt("Sprinkler", {(T,): 0.01, (F,): 0.4})
    bn.set_boolean_cpt("GrassWet", {(T, T): 0.99, (T, F): 0.90, (F, T): 0.80, (F, F): 0.0})
    bn.validate()
    return bn
