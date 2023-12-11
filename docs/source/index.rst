Welcome to pairwise-ranking's documentation!
===================================

**pairwise-ranking** is a Python library for producing ranking information from pairwise data.
Particularly, it implements Bayesian inference on extensions of the popular Bradley-Terry model
popularly known as the basis for Elo scores. 

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

.. _models:

Models
----------------

This package implements a variety of models for pairwise ranking:

- **Depth and Luck** ('depth_and_luck')
   Full model allowing for both the depth (beta) and luck (alpha) to vary.

- **Depth only** ('depth_only')
   Bradley-Terry model with a hyperprior on the depth (beta). 
   The luck (alpha) is set to 0 so that a very strong player will always beat a very weak one. 

- **Luck only** ('luck_only')
   Minimum violation ranking approximated by setting the depth beta = 100. The luck alpha, the probability
   that the weaker player wins is allowed to vary.

- **Logistic prior** ('logistic_prior')
   Bradley-Terry model with a fixed logistic prior on the scores. This model is approximately described
   by the general model for fixed depth beta = 2.56 and luck alpha = 0. The maximum a posteriori (MAP)
   estimates of the scores in this model can be very efficiently found, unlike in the other models. 
