Models
=====

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

These models and their relative performances are described in more detail in the paper

- M. Jerdee, M. E. J. Newman, Luck, skill, and depth of competition in games and social hierarchies, Preprint `arXiv:2312.04711 <https://arxiv.org/abs/2312.04711>`_ (2023).

Generally the takeaways are that the full 'depth_and_luck' model broadly performs the best, but MAP estimation
of the 'logistic_prior' model is not far behind. 