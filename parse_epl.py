"""Parse the openfootball 2024-25 Premier League data into match list format."""
import re

# Team name mapping: full name -> short name (no spaces)
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

# Pattern to match a result line like:
#   "    15.00  Arsenal FC              v Wolverhampton Wanderers FC  2-0 (1-0)"
# or without time:
#   "           Everton FC              v Brighton & Hove Albion FC  0-3 (0-1)"
MATCH_RE = re.compile(
    r"^\s+(?:\d{2}\.\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)\s*(?:\(.*\))?\s*$"
)

matches = []

with open("data/epl_2024_25_raw.txt") as f:
    for line in f:
        m = MATCH_RE.match(line)
        if m:
            home_team = m.group(1).strip()
            away_team = m.group(2).strip()
            home_goals = int(m.group(3))
            away_goals = int(m.group(4))

            home_short = TEAM_MAP.get(home_team)
            away_short = TEAM_MAP.get(away_team)
            if not home_short:
                print(f"WARNING: Unknown team '{home_team}'")
                continue
            if not away_short:
                print(f"WARNING: Unknown team '{away_team}'")
                continue

            if home_goals > away_goals:
                matches.append(f"{home_short} {away_short}")
            elif away_goals > home_goals:
                matches.append(f"{away_short} {home_short}")
            # Draws are skipped (no winner)

print(f"Total matches with a winner: {len(matches)} (out of 380)")
print(f"Draws skipped: {380 - len(matches)}")

with open("data/match_lists/epl_2024_25.txt", "w") as f:
    f.write("\n".join(matches) + "\n")

print("Written to data/match_lists/epl_2024_25.txt")
