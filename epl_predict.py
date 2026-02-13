"""
EPL 2024-25: Train on first half (MD 1-19), predict second half (MD 20-38).
Skips draws. Estimates overall prediction accuracy.
"""
import re
import ranking
import numpy as np

# ── Team name mapping ──────────────────────────────────────────────
TEAM_MAP = {
    "Manchester United FC": "Manchester_United",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich_Town",
    "Liverpool FC": "Liverpool",
    "Arsenal FC": "Arsenal",
    "Wolverhampton Wanderers FC": "Wolverhampton",
    "Everton FC": "Everton",
    "Brighton & Hove Albion FC": "Brighton",
    "Newcastle United FC": "Newcastle_United",
    "Southampton FC": "Southampton",
    "Nottingham Forest FC": "Nottingham_Forest",
    "AFC Bournemouth": "Bournemouth",
    "West Ham United FC": "West_Ham",
    "Aston Villa FC": "Aston_Villa",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal_Palace",
    "Chelsea FC": "Chelsea",
    "Manchester City FC": "Manchester_City",
    "Leicester City FC": "Leicester_City",
    "Tottenham Hotspur FC": "Tottenham",
}

MATCH_RE = re.compile(
    r"^\s+(?:\d{2}\.\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)\s*(?:\(.*\))?\s*$"
)
MATCHDAY_RE = re.compile(r"^» Matchday (\d+)")

# ── Parse raw data with matchday tracking ──────────────────────────
first_half = []   # list of (winner, loser) for MD 1-19
second_half = []  # list of (winner, loser) for MD 20-38
current_md = 0

with open("data/epl_2024_25_raw.txt") as f:
    for line in f:
        md_match = MATCHDAY_RE.match(line)
        if md_match:
            current_md = int(md_match.group(1))
            continue

        m = MATCH_RE.match(line)
        if m:
            home = TEAM_MAP.get(m.group(1).strip())
            away = TEAM_MAP.get(m.group(2).strip())
            hg, ag = int(m.group(3)), int(m.group(4))

            if not home or not away:
                continue
            if hg == ag:  # skip draws
                continue

            winner = home if hg > ag else away
            loser = away if hg > ag else home

            if current_md <= 19:
                first_half.append((winner, loser))
            else:
                second_half.append((winner, loser))

print(f"First half (MD 1-19):  {len(first_half)} decisive matches")
print(f"Second half (MD 20-38): {len(second_half)} decisive matches")
print()

# ── Build training match_list ──────────────────────────────────────
train_match_list = [{"winner": w, "loser": l} for w, l in first_half]

# ── Fit model on first half ────────────────────────────────────────
print("Fitting depth_and_luck model on first-half data...")
scores_df = ranking.scores(
    train_match_list, model_name="depth_and_luck",
    num_samples=2000, num_chains=4
)
print("\nFirst-half rankings:")
print(scores_df.to_string())
print()

params = ranking.depth_and_luck(
    train_match_list, model_name="depth_and_luck",
    num_samples=2000, num_chains=4
)
print(f"First-half depth (beta): {params['depth']:.3f}")
print(f"First-half luck (alpha): {params['luck']:.3f}")
print()

# ── Predict second-half matches ───────────────────────────────────
print("=" * 65)
print("PREDICTING SECOND-HALF MATCHES")
print("=" * 65)

correct = 0
total = 0
results = []

for actual_winner, actual_loser in second_half:
    # Model gives P(team1 beats team2)
    prob_w, err = ranking.probability(
        train_match_list, actual_winner, actual_loser,
        model_name="depth_and_luck", num_samples=2000, num_chains=4
    )
    # prob_w = P(actual_winner beats actual_loser)
    # If prob_w > 0.5, the model predicts the actual winner → correct
    predicted_correct = prob_w > 0.5
    if predicted_correct:
        correct += 1
    total += 1

    w_display = actual_winner.replace("_", " ")
    l_display = actual_loser.replace("_", " ")
    mark = "OK" if predicted_correct else "X"
    results.append((w_display, l_display, prob_w, mark))

# ── Print results ─────────────────────────────────────────────────
print(f"\n{'Match':<45} {'P(winner)':<12} {'Result'}")
print("-" * 65)
for w, l, p, mark in results:
    label = f"{w} beat {l}"
    print(f"{label:<45} {p:<12.3f} {mark}")

print()
print("=" * 65)
accuracy = correct / total
print(f"OVERALL ACCURACY: {correct}/{total} = {accuracy:.1%}")
print("=" * 65)

# ── Calibration by confidence bucket ──────────────────────────────
print("\nCalibration breakdown (by model confidence):")
buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in buckets:
    # For each match, the model's confidence in its prediction is max(p, 1-p)
    bucket_results = []
    for _, _, p, mark in results:
        confidence = p if p > 0.5 else 1 - p
        if lo <= confidence < hi:
            bucket_results.append(mark == "OK")
    if bucket_results:
        acc = sum(bucket_results) / len(bucket_results)
        print(f"  Confidence [{lo:.0%}-{hi:.0%}): {sum(bucket_results)}/{len(bucket_results)} = {acc:.1%}")
    else:
        print(f"  Confidence [{lo:.0%}-{hi:.0%}): no matches")

# Also count "upsets" — where the model's predicted loser actually won
wrong = [(w, l, p) for w, l, p, m in results if m == "X"]
if wrong:
    print(f"\nBiggest upsets (model was most confident in the wrong team):")
    wrong.sort(key=lambda x: x[2])  # lowest p = model was most wrong
    for w, l, p in wrong[:10]:
        print(f"  {w} beat {l} (model gave only {p:.1%} chance)")
