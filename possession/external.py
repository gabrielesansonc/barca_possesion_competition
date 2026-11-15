"""External training matches (PLAN §8, approach B): Metrica's open sample data as Match objects.

Source: github.com/metrica-sports/sample-data (3 anonymised matches, tracking + events; no licence
file, the README allows community use with acknowledgement). Downloaded to
data/external/metrica_sample/ (gitignored). BAR-ATH is never used for training (decision 10-22);
these matches are the only training data for the learned model.

Each match comes back as a possession.data.Match, so possession.features builds identical features for it,
plus its event list. Arrays are indexed by frame - 1. Fields that only BAR-ATH has (homography,
Smart Tagging, the ATD speed field) are None or empty. The event
labels come from possession.ext_labels.

    from possession.external import load_external, EXTERNAL_GAMES
    m, events = load_external(1)
"""

import json
import os

import numpy as np
import pandas as pd

from possession import DATA_DIR
from possession.data import Match, HOME, AWAY
import possession_metrics as pm  # noqa: E402

# Params
EXTERNAL_DIR = os.path.join(DATA_DIR, "external", "metrica_sample", "data")
EXTERNAL_GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)   # README: both axes 0-1 over 105 x 68 m

EVENT_COLUMNS = ["team", "type", "subtype", "period", "start_frame", "end_frame",
                 "start_x", "start_y", "end_x", "end_y"]


def speed_from_positions(players):
    """(n, k) float32 m/s from positions, in place of the ATD's speed field. NaN across slot changes."""
    speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * 25
    speed[speed > 12] = np.nan   # a slot reused by another player, not running
    return speed


def _csv_side(game, side):
    """(periods, players (n, k, 2) metres, ball (n, 2) metres) from one Metrica CSV."""
    path = os.path.join(EXTERNAL_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(game, side))
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    xs = [c for c in cols if c.startswith("Player")]
    players = np.stack([raw[[c, cols[cols.index(c) + 1]]].values for c in xs], axis=1)
    ball = raw[[cols[-2], cols[-1]]].values
    if not np.array_equal(raw["Frame"].values, np.arange(1, len(raw) + 1)):
        raise ValueError("{}: frames are not 1..n".format(path))
    return raw["Period"].values, players * PITCH, ball * PITCH


def _csv_events(game):
    path = os.path.join(EXTERNAL_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_RawEventsData.csv".format(game))
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "team": np.where(raw["Team"] == "Home", "H", "A"),
        "type": raw["Type"].str.strip().str.upper(),
        "subtype": raw["Subtype"].fillna("").str.strip().str.upper(),
        "period": raw["Period"],
        "start_frame": raw["Start Frame"] - 1,
        "end_frame": raw["End Frame"] - 1,
        "start_x": raw["Start X"] * PITCH[0], "start_y": raw["Start Y"] * PITCH[1],
        "end_x": raw["End X"] * PITCH[0], "end_y": raw["End Y"] * PITCH[1],
    })


def _epts(game):
    """(periods, players, team, ball, events) for the EPTS + JSON match, via kloppy."""
    from kloppy import metrica
    base = os.path.join(EXTERNAL_DIR, "Sample_Game_{0}".format(game), "Sample_Game_{0}".format(game))
    ds = metrica.load_tracking_epts(meta_data=base + "_metadata.xml", raw_data=base + "_tracking.txt")
    df = ds.to_df()
    if not np.array_equal(df["frame_id"].values, np.arange(1, len(df) + 1)):
        raise ValueError("game {}: frames are not 1..n".format(game))
    home_team, away_team = ds.metadata.teams  # kloppy lists home first
    cols, team = [], []
    for code, t in ((HOME, home_team), (AWAY, away_team)):
        for p in t.players:
            if p.player_id + "_x" in df:
                cols.append(p.player_id)
                team.append(code)
    players = np.stack([df[[c + "_x", c + "_y"]].values for c in cols], axis=1) * PITCH
    ball = df[["ball_x", "ball_y"]].values * PITCH

    with open(base + "_events.json") as handle:
        data = json.load(handle)["data"]
    home_id = home_team.team_id
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        start, end = e["start"], e["end"] or e["start"]
        rows.append({
            "team": "H" if e["team"]["id"] == home_id else "A",
            "type": e["type"]["name"].strip().upper(), "subtype": (sub or "").strip().upper(),
            "period": e["period"],
            "start_frame": start["frame"] - 1, "end_frame": end["frame"] - 1,
            "start_x": (start["x"] if start["x"] is not None else np.nan) * PITCH[0],
            "start_y": (start["y"] if start["y"] is not None else np.nan) * PITCH[1],
            "end_x": (end["x"] if end["x"] is not None else np.nan) * PITCH[0],
            "end_y": (end["y"] if end["y"] is not None else np.nan) * PITCH[1],
        })
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df["period_id"].values, players, np.array(team, dtype=np.int8), ball, events


def load_external(game):
    """(Match, events DataFrame) for one external match. Coordinates in metres on 105 x 68."""
    if game == 3:
        period, players, team, ball, events = _epts(game)
    else:
        period, home, ball = _csv_side(game, "Home")
        _, away, _ = _csv_side(game, "Away")
        players = np.concatenate([home, away], axis=1)
        team = np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8)
        events = _csv_events(game)
    n = len(ball)
    m = Match(
        players=players.astype(np.float32), player_speed=speed_from_positions(players), team=team,
        ball=ball.astype(np.float32), ball_speed=None, half=period.astype(np.int8),
        homography=None, reproj_error=None,
        tagging=pd.DataFrame(), pitch=PITCH,
    )
    return m, events.sort_values("start_frame", kind="stable", ignore_index=True)
