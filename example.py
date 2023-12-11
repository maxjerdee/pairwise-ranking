import ranking
import matplotlib.pyplot as plt

match_list = ranking.read_match_list_from_gml("data/gml_files/dogs.gml")
# match_list = ranking.read_match_list_from_match_list("data/match_lists/college_football_D1_2023.txt")
# match_list = ranking.read_match_list_from_match_list("data/match_lists/tennis.txt")

scores = ranking.scores(match_list, model_name="depth_and_luck", force_mode="average")

print(scores)

# ranking.draw_depth_and_luck_posterior(match_list)

# plt.savefig('data/parameter_posteriors/tennis.png')
