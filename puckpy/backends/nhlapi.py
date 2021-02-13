"""This module defines the backends for NHL api.
"""

import os
from typing import List, Optional

import pandas as pd
from pandas.core.algorithms import value_counts
import requests
from fuzzywuzzy import process
from requests import HTTPError


# ----------------------------------------------------------------------
# NHL API URLs.
# ----------------------------------------------------------------------
# Big thanks to https://github.com/dword4/nhlapi to describing the API.
# ----------------------------------------------------------------------


BASE_URL = "https://statsapi.web.nhl.com/api/v1/"

# Team urls.
get_team_url = lambda: os.path.join(BASE_URL, "teams")
get_roster_url = lambda team_id: os.path.join(
    get_team_url(), f"{team_id}?expand=team.roster"
)

# Player urls
get_player_url = lambda player_id: os.path.join(
    BASE_URL, f"people/{player_id}"
)
get_player_games_stats_url = lambda player_id, season: os.path.join(
    get_player_url(player_id), f"stats?stats=gameLog&season={season}"
)

# Schedule urls
get_schedule_url = lambda: os.path.join(BASE_URL, "schedule")
get_schedule_for_team_url = lambda team_id, season: os.path.join(
    get_schedule_url(),
    f"?teamId={team_id}&startDate={season[:4]}-09-01&endDate={season[4:]}-09-01",
)


# ----------------------------------------------------------------------
# Utils. TODO: should be moved.
# ----------------------------------------------------------------------


def unroll_json(
    content: dict,
    blacklist: Optional[List[str]] = None,
    prefix: Optional[str] = None,
) -> dict:
    """Unroll a json dict in records with a blacklist.

    Parameters
    ----------
    content: dict
        The content of the json dict.
    blacklist: Optional[List[str]]
        A list of json key to throw away (the default value is None,
        which means no blacklist).
    prefix: Optional[str]
        A prefix to add before keys (the default value is None, which
        means no prefix).

    Returns
    -------
    dict
        A unroll json record.
    """
    blacklist = [] if blacklist is None else blacklist
    output = {}
    for k, v in content.items():
        if k in blacklist:
            continue
        key = f"{prefix}_{k}" if prefix is not None else k
        if isinstance(v, dict):
            down_output = unroll_json(v, blacklist, prefix=key)
            output.update(down_output)
        else:
            output[key] = v
    return output


# ----------------------------------------------------------------------
# API raw calls.
# ----------------------------------------------------------------------


def get(url: str) -> dict:
    """Execute a get on the NHL API.

    Parameters
    ----------
    url: str
        The url to get.

    Returns
    -------
    dict
        The json part of the api answer.
    """
    res = requests.get(url)
    if res.status_code != 200:
        raise HTTPError(f"API issue, status code {res.status_code}.")
    return res.json()


# ----------------------------------------------------------------------
# Search functions.
# ----------------------------------------------------------------------


def find_team_id(team_name) -> pd.Series:
    """Find the team id from a team name. Returns the 3 teams that
    match the most "team_name".

    Parameters
    ----------
    team_name: str
        The team name to find.

    Returns
    -------
    pd.Series
        A series of the 3 teams that match the most.

    """
    url = get_team_url()
    teams = get(url)["teams"]
    teams = {t["name"]: t["id"] for t in teams}
    team_choices = process.extract(team_name, teams.keys(), limit=3)
    return pd.Series(
        data=[teams[res[0]] for res in team_choices],
        index=[res[0] for res in team_choices],
    )


def find_player_id(
    player_name: str,
    team_id: Optional[int] = None,
    team_name: Optional[str] = None,
) -> pd.Series:
    """Find the player id from a player name and either a team id or a
    team name. Returns the 3 players that match the most "player_name".

    Parameters
    ----------
    player_name: str
        The name of the player.
    team_id: Optional[int]
        The team id of the player (the default value is None which
        means that "team_name" must be given).
    team_name: Optional[int]
        The team name of the player (the default value is None which
        means that "team_id" must be given).

    Returns
    -------
    pd.Series
        A series of the 3 players that match the most.

    """
    if team_id is not None and team_name is not None:
        raise AttributeError("You need to give either team_id or team_name.")
    if team_id is None and team_name is None:
        raise AttributeError("You need to give either team_id or team_name.")
    if team_id is None:
        team_id = find_team_id(team_name).iloc[0]
    url = get_roster_url(team_id)
    roster = get(url)["teams"][0]["roster"]["roster"]
    players = {p["person"]["fullName"]: p["person"]["id"] for p in roster}
    player_choices = process.extract(player_name, players.keys(), limit=3)
    return pd.Series(
        data=[players[res[0]] for res in player_choices],
        index=[res[0] for res in player_choices],
    )


# ----------------------------------------------------------------------
# Data fetching functions.
# ----------------------------------------------------------------------


def get_player_games_stats(player_id: int, season: str) -> pd.DataFrame:
    """Returns a pd.DataFrame of game stats for a player_id and a
    season.

    Parameters
    ----------
    player_id: int
        The id of the player.
    season: str
        The season to fetch. The season must be formated as "aaaabbbb"
        where "aaaa" is the first year of the season and "bbbb" the
        second year of the season, e.g. "20182019".

    Returns
    -------
    pd.DataFrame
        A dataframe that contains games stats for a player and a season.

    """
    url = get_player_games_stats_url(player_id, season)
    games_stats = get(url)["stats"][0]["splits"]
    game_record_list = []
    for game_stats in games_stats:
        game_record = {}
        for k1, v1 in game_stats.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    key = f"{k1}_{k2}" if k1 != "stat" else k2
                    game_record[key] = v2
            else:
                game_record[k1] = v1
        game_record_list.append(game_record)
    data = pd.DataFrame(game_record_list)
    data.set_index("date", inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    return data


def get_team_games_schedule(team_id: int, season: str):
    """Returns a pd.DataFrame with games scheduled in a season.

    Parameters
    ----------
    team_id: int
        The id of the team.
    season: str
        The season to fetch. The season must be formated as "aaaabbbb"
        where "aaaa" is the first year of the season and "bbbb" the
        second year of the season, e.g. "20182019".

    Returns
    -------
    pd.DataFrame
        A dataframe that contains games schedule for a team and a season.

    Notes
    -----
    It does not seem to work with 20202021 COVID mixup season...

    """
    url = get_schedule_for_team_url(team_id, season)
    schedules = get(url)["dates"]
    schedule_record_list = []
    for schedule in schedules:
        for games in schedule["games"]:
            schedule_record = {"date": schedule["date"]}
            schedule_record.update(unroll_json(games))
            schedule_record_list.append(schedule_record)
    data = pd.DataFrame(schedule_record_list)
    data.set_index("date", inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    return data
