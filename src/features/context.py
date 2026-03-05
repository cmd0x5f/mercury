"""Game context features — rest days, schedule position, etc."""

import pandas as pd


def compute_context_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add contextual features to games.

    Columns added:
        home_rest_days, away_rest_days: days since team's last game
        home_game_num, away_game_num: team's Nth game of the season
        is_back_to_back_home, is_back_to_back_away: rest_days <= 1
    """
    games = games.copy().sort_values("date").reset_index(drop=True)

    last_game: dict[str, str] = {}  # team -> last game date
    game_count: dict[str, int] = {}  # team -> games played this season
    current_season: dict[str, str] = {}  # team -> season tracker

    h_rest, a_rest = [], []
    h_gnum, a_gnum = [], []

    for _, g in games.iterrows():
        h, a, date = g["home_team"], g["away_team"], g["date"]

        # Rest days
        if h in last_game:
            delta = (pd.Timestamp(date) - pd.Timestamp(last_game[h])).days
            h_rest.append(delta)
        else:
            h_rest.append(7)  # season opener default

        if a in last_game:
            delta = (pd.Timestamp(date) - pd.Timestamp(last_game[a])).days
            a_rest.append(delta)
        else:
            a_rest.append(7)

        # Game number (resets each ~Oct for new season)
        month = pd.Timestamp(date).month
        season_key = f"{pd.Timestamp(date).year}-{month >= 10}"

        for team, nums in [(h, h_gnum), (a, a_gnum)]:
            team_season = current_season.get(team)
            if team_season != season_key:
                game_count[team] = 0
                current_season[team] = season_key
            game_count[team] = game_count.get(team, 0) + 1
            nums.append(game_count[team])

        last_game[h] = date
        last_game[a] = date

    games["home_rest_days"] = h_rest
    games["away_rest_days"] = a_rest
    games["home_game_num"] = h_gnum
    games["away_game_num"] = a_gnum
    games["is_b2b_home"] = (games["home_rest_days"] <= 1).astype(int)
    games["is_b2b_away"] = (games["away_rest_days"] <= 1).astype(int)
    return games
