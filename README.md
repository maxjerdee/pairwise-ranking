# pairwise-ranking
<a href='https://pairwise-ranking.readthedocs.io/en/latest/?badge=latest'> <img src='https://readthedocs.org/projects/pairwise-ranking/badge/?version=latest' alt='Documentation status' /></a> <a href="https://badge.fury.io/py/pytrx"><img src="https://badge.fury.io/py/pairwise-ranking.svg" alt="PyPI status" height="18"></a> <br>

### Models for ranking participants given of comparisons between them

### Maximilian Jerdee, Mark Newman

Paired comparisons may arise from records of sports matches, social interactions, any other preferences between pairs of objects. From these data sets can model the "strengths" of each participant. <br>

The models implemented are described in the paper

M. Jerdee, M. E. J. Newman, Luck, skill, and depth of competition in games and social hierarchies, Preprint <a href="https://arxiv.org/abs/2312.04711">arxiv:2312.04711</a>
(2023).

These models may also be used to infer the "depth-of-competition" and "luck" present in the hierarchies considered.

## Installation
pairwise-ranking may be installed through pip:

```python
import ranking
```

Note that this is not `import pairwise-ranking`.

## Typical usage
Filenames can be in .gml network format, or as a .txt format of a list of matches or adjacency matrix

```
# Read file as a list of matches
match_list = ranking.read_match_list(filename)
# Get ranking of participants
ranks = ranking.ranks(match_list)
# Get scores of participants in the model (higher is better!)
scores = ranking.scores(match_list)
# Get predicted probability that "Winner" will beat "Loser" in a contest
probability = ranking.get_probability(match_list,"Winner","Loser")
# Luck and depth parameters
depth, luck = ranking.get_depth_and_luck(match_list)
```

See the file `example.py` for examples of more advanced usage options and our <a href="https://pytrx.readthedocs.io/en/latest/Installation.html">readthedocs page for further documentation. 