import ranking
import matplotlib.pyplot as plt

#################################################################
# Example 1: dogs (n = 27 dogs that interact m = 1143 times)
# Load data set
match_list = ranking.read_match_list("data/gml_files/dogs.gml")

# Find scores (requires sampling, which can take a while, can reduce num_samples)
scores = ranking.scores(match_list)
print(scores)

# Get ranks (uses cached samples when possible)
ranks = ranking.ranks(match_list)
print(ranks)

# Find probability (and error) of an outcome betwen a pair
player1 = "MER"
player2 = "STE"
probability, probability_error = ranking.probability(match_list, player1, player2)
print(
    f"Probability {probability:.4f}+/-{probability_error:.4f} that {player1} beats {player2}"
)

# Infer the depth and luck of the hierarchy
params = ranking.depth_and_luck(match_list)
print(f"Depth: {params['depth']:.3f}, Luck: {params['luck']:.3f}")

# Draw posterior distribution over depth and luck
# (compared to the point estimates this is the more informative picture)
ranking.draw_depth_and_luck_posterior(match_list)
plt.savefig("data/parameter_posteriors/dogs.png")


##########################################################################
# Example 2: business_depts (n = 112 business departments that hire m = 7856 faculty between them)
# Load data set
match_list = ranking.read_match_list("data/match_lists/business_depts.txt")

# Find scores (requires sampling, which can take a while)
# This time, use the depth_only model (assume no luck component)
# Also decrease the number of samples taken (may want to do this for larger data sets, default is 5000)
scores = ranking.scores(match_list, model_name="depth_only", num_samples=2000)
print(scores)

# Find probability (and error) of an outcome betwen a pair
player1 = "Cornell_University"
player2 = "MIT"
# Note that for the sample caching to work, need to also restrict the samples required by the probability function call
probability, probability_error = ranking.probability(
    match_list, player1, player2, model_name="depth_only", num_samples=2000
)
print(
    f"Probability {probability:.4f}+/-{probability_error:.4f} that {player1} beats {player2}"
)

# Infer the depth and luck of the hierarchy
params = ranking.depth_and_luck(match_list, model_name="depth_only", num_samples=2000)
# In this case the luck is set to 0 by the model definition
print(f"Depth: {params['depth']:.3f}, Luck: {params['luck']:.3f}")


##########################################################################
# Example 3: tennis (n = 1272 tennis players who played m = 29397 matches)
# Load data set
match_list = ranking.read_match_list("data/match_lists/tennis.txt")

# Maximum a posteriori (MAP) estimates of the scores in the logistic_prior model
# can be found much more quickly than through MCMC sampling.
# Although our analysis suggests that the depth_and_luck model is typically best,
# for large data sets it may make sense to use these estimates in this model,
# since it still performs respectably.
scores = ranking.scores(match_list, model_name="logistic_prior", force_mode="MAP")
print(scores)

# Find probability (and error) of an outcome betwen a pair
player1 = "Novak_Djokovic"
player2 = "Juan_Martin_del_Potro"
# Note that for the sample caching to work, need to also restrict the samples required by the probability function call
probability, probability_error = ranking.probability(
    match_list, player1, player2, model_name="logistic_prior", force_mode="MAP"
)
print(
    f"Probability {probability:.4f}+/-{probability_error:.4f} that {player1} beats {player2}"
)
