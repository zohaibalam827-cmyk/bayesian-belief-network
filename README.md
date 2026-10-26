# Bayesian Belief Network

A small Python library for Bayesian belief networks (BBNs) with:

- network structure with validated conditional probability tables (CPTs)
- exact inference by **enumeration** and by **variable elimination**
- **learning** CPTs from data (maximum likelihood with Laplace smoothing)
- two example networks: the textbook alarm network and the sprinkler network

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from bbn.networks import alarm_network
from bbn import enumeration_ask, variable_elimination_ask

bn = alarm_network()
evidence = {"JohnCalls": True, "MaryCalls": True}

print(enumeration_ask(bn, "Burglary", evidence))
# {True: 0.2842..., False: 0.7158...}
print(variable_elimination_ask(bn, "Burglary", evidence))   # same answer, faster
```

Learning tables from data:

```python
import random
from bbn import learn_cpts, sample
from bbn.networks import sprinkler_network

data = sample(sprinkler_network(), 50000, random.Random(0))  # list of dicts
learned = learn_cpts(sprinkler_network(), data)              # fills in every CPT
print(learned.describe())
```

To build your own network, see how the examples are defined in `bbn/networks.py`.

## Project layout

| File | Purpose |
|------|---------|
| `bbn/network.py` | `BayesianNetwork` and `Node`, CPT validation, joint probability |
| `bbn/inference.py` | `enumeration_ask`, `variable_elimination_ask` |
| `bbn/learning.py` | `learn_cpts`, `sample` |
| `bbn/networks.py` | alarm and sprinkler example networks |
| `tests/` | pytest tests |

## Tests

```bash
python -m pytest -v
```

## Notes

- Burglary is rare (0.1%), so tables learned from data are noisy for rare events unless you have a lot of data. The learning test uses the sprinkler network, where every case is common.
- Laplace smoothing (`alpha`, default 1) gives unseen combinations a small non-zero probability. Use `alpha=0` for plain maximum likelihood.
