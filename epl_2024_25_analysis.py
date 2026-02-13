"""
Analyze the 2024-25 English Premier League season using pairwise Bayesian ranking.

Uses the depth_and_luck model to infer:
  - Team strength scores and rankings
  - Depth (how much skill determines outcomes) and luck parameters
  - Head-to-head win probabilities for selected matchups
"""
import ranking
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load data
print("=" * 60)
print("English Premier League 2024-25 - Bayesian Pairwise Ranking")
print("=" * 60)

match_list = ranking.read_match_list("data/match_lists/epl_2024_25.txt")
print(f"\nLoaded {len(match_list)} decisive matches (draws excluded)\n")

# --- 1. Scores and Rankings ---
print("Fitting depth_and_luck model (MCMC sampling)...")
print("This may take a few minutes on first run (Stan compilation)...\n")

scores_df = ranking.scores(match_list, model_name="depth_and_luck", num_samples=2000, num_chains=4)
print("=" * 60)
print("TEAM RANKINGS (by Bayesian inferred strength)")
print("=" * 60)
print(scores_df.to_string())
print()

# --- 2. Depth and Luck ---
params = ranking.depth_and_luck(match_list, model_name="depth_and_luck", num_samples=2000, num_chains=4)
print("=" * 60)
print("HIERARCHY PARAMETERS")
print("=" * 60)
print(f"  Depth (beta):  {params['depth']:.3f}")
print(f"  Luck (alpha):  {params['luck']:.3f}")
print()
print("Interpretation:")
print("  - Higher depth = skill matters more (outcomes are more predictable)")
print("  - Higher luck  = more randomness (upsets are more common)")
print()

# --- 3. Selected matchup probabilities ---
matchups = [
    ("Liverpool", "Manchester_City"),
    ("Arsenal", "Chelsea"),
    ("Manchester_City", "Manchester_United"),
    ("Nottingham_Forest", "Tottenham"),
    ("Liverpool", "Arsenal"),
]

print("=" * 60)
print("HEAD-TO-HEAD WIN PROBABILITIES")
print("=" * 60)
for team1, team2 in matchups:
    prob, err = ranking.probability(
        match_list, team1, team2,
        model_name="depth_and_luck", num_samples=2000, num_chains=4
    )
    t1 = team1.replace("_", " ")
    t2 = team2.replace("_", " ")
    print(f"  P({t1} beats {t2}) = {prob:.3f} +/- {err:.3f}")
print()

# --- 4. Draw posterior distribution ---
print("Drawing depth & luck posterior distribution...")
ranking.draw_depth_and_luck_posterior(match_list, num_samples=2000, num_chains=4)
plt.title("EPL 2024-25: Depth & Luck Posterior")
plt.savefig("data/epl_2024_25_posterior.png", dpi=150, bbox_inches="tight")
print("Saved posterior plot to data/epl_2024_25_posterior.png")
print()
print("Done!")
