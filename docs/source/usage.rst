Usage
=====

.. _installation:

Installation
------------

To use pairwise-ranking, first install it using pip:

.. code-block:: console

   $ pip install pairwise-ranking

.. _parsing:

Loading data
------------

Match data may be imported from a variety of formats: .gml files, adjacency matrices, and lists of matches. The function ``ranking.read_match_list()`` attempts to import the data in these formats. 

.. autofunction:: ranking.read_match_list()

For a specific file format, the more specific functions can be used:
   
.. autofunction:: ranking.read_match_list_from_match_list()
   
.. autofunction:: ranking.read_match_list_from_gml()
   
.. autofunction:: ranking.read_match_list_from_adj_matrix()

.. _inference:

Inference
----------------

For the models implemented in this package, described in :ref:`models`, point estimates of the strength scores can be found
 with the function ``ranking.scores()``. 

.. autofunction:: ranking.scores()

Listed in decreasing order of the score estimates, the rankings from a ``match_list`` may be found: 

.. autofunction:: ranking.ranks()

We can also infer the probability that an outcome between two players might occur:

.. autofunction:: ranking.probability()

.. _sampling:

Sampling
----------------
We implement a wrapper for Hamiltonian Monte Carlo (HMC) sampling via pystan for the models considered in this package:

.. autofunction:: ranking.sampling()

These samples may also be used to visualize the posterior distribution of the depth and luck in the full model using ``matplotlib.pyplot``:

.. autofunction:: ranking.draw_depth_and_luck_posterior()

