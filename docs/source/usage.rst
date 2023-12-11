Usage
=====

.. _installation:

Installation
------------

To use pairwise-ranking, first install it using pip:

.. code-block:: console

   $ pip install pairwise-ranking

.. _parsing:

Match data may be imported from a variety of formats: .gml files, adjacency matrices, and lists of matches.
 The function ``ranking.read_match_list()`` attempts to import the data in these formats. 

.. autofunction:: read_match_list()

For a specific file format, the more specific functions can be used:
   
.. autofunction:: read_match_list_from_match_list()
   
.. autofunction:: read_match_list_from_gml()
   
.. autofunction:: read_match_list_from_adj_matrix()

.. _scores:

Bayesian inference
----------------

.. autofunction:: ranking.scores()


Sampling
----------------
We implement a wrapper 


For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

