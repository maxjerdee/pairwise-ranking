import ranking
import matplotlib.pyplot as plt

# Example 1: dogs (n = 27 dogs that interact m = 1143 times)
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

# Draw posterior distribution over depth and luck 
# (compared to the point estimates this is the more informative picture)
ranking.draw_depth_and_luck_posterior(match_list)
plt.savefig('data/parameter_posteriors/dogs.png')
