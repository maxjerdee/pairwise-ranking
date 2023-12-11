# pairwise-ranking
<a href='https://pairwise-ranking.readthedocs.io/en/latest/?badge=latest'> <img src='https://readthedocs.org/projects/pairwise-ranking/badge/?version=latest' alt='Documentation status' /></a> <a href="https://badge.fury.io/py/pytrx"><img src="https://badge.fury.io/py/pairwise-ranking.svg" alt="PyPI status" height="18"></a> <br>

### Models for ranking participants given pairwise comparisons between them

### Maximilian Jerdee, Mark Newman

Paired comparisons may arise from records of sports matches, social interactions, any other preferences between pairs of objects. From these data sets we can model the "strengths" of each participant. <br>

The models implemented implemented in this package are described in this paper:

M. Jerdee, M. E. J. Newman, Luck, skill, and depth of competition in games and social hierarchies, Preprint <a href="https://arxiv.org/abs/2312.04711">arxiv:2312.04711</a>
(2023).

As shown in the paper, these models have strong predictive performance and may also be used to infer the "depth-of-competition" and "luck" present in the hierarchies considered.

## Installation
pairwise-ranking may be installed through pip:

```bash
pip install pairwise-ranking
```

## Typical usage
Once installed, the package can be imported as
```python
import ranking
```
Note that this is not `import pairwise-ranking`.

Files can be in a .gml network format, or read as a .txt format of either a list of matches or of an adjacency matrix of head-to-head records. See the `/data` directory for examples.

After loading a match_list from a data set, the package may be used to rank participants, make predictions, and compute depth and luck. 
```python
# Load data set
match_list = ranking.read_match_list("data/gml_files/dogs.gml")

# Find scores (requires sampling, which can take a while)
scores = ranking.scores(match_list)
print(scores)

# Get ranks (uses cached samples when possible)
ranks = ranking.ranks(match_list)
print(ranks)

# Find probability (and error) of an outcome betwen a pair
player1 = 'MER'
player2 = 'STE'
probability, probability_error = ranking.probability(match_list,player1,player2)
print(f"Probability {probability:.4f}+/-{probability_error:.4f} that {player1} beats {player2}")

# Infer the depth and luck of the hierarchy
params = ranking.depth_and_luck(match_list)
print(f"Depth: {params['depth']:.3f}, Luck: {params['luck']:.3f}")
```

See the file `example.py` for examples of more advanced usage options and our <a href="https://pytrx.readthedocs.io/en/latest/Installation.html">readthedocs page for further documentation and data set citations. 