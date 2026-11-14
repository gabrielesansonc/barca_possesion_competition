"""Read the match data once, save it as numpy arrays (data/cache/), then load it in about a second.

    from possession.data import load_match
    m = load_match()
    m.players[f]       # (46, 2) positions in metres of the 46 track slots at video frame f (NaN if absent)
    m.ball[f]          # (2,) ball position in metres (NaN if the tracking has no ball)

Every array is indexed by video frame (0 .. 199318), which is also the row of the submission.
The tracking's frame_id is video frame + 1; that shift is applied here and nowhere else.

Coordinates are metres on a 105.3 x 68.0 pitch, origin at a corner. Positions outside the pitch
are real: the tracking follows the ball and players off it.
"""

import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from possession import COMPETITION, DATA_DIR
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CACHE_VERSION = 1  # bump when the cache layout changes, to force a rebuild
PUBLIC_GT = os.path.join(COMPETITION, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
N_SLOTS = 46
HOME, AWAY = 0, 1

# The 2nd half is graded from 110808 but kicks off at 113742: those frames are known to be D.
SECOND_HALF_KICKOFF = 113742


@dataclass
class Match:
    """All per-frame data for one match. Arrays of length n_frames unless stated."""
    players: np.ndarray       # (n, 46, 2) float32, metres
    player_speed: np.ndarray  # (n, 46) float32, m/s as delivered by the ATD
    team: np.ndarray          # (46,) int8, HOME / AWAY per track slot
    ball: np.ndarray          # (n, 2) float32, metres
    ball_speed: np.ndarray    # (n,) float32, as delivered (mostly NaN; recompute from positions)
    half: np.ndarray          # (n,) int8, 1 / 2 on graded frames, 0 elsewhere
    homography: np.ndarray    # (n, 3, 3) float32, normalised pitch -> pixels
    reproj_error: np.ndarray  # (n,) float32
    gt_public: np.ndarray     # (n,) int8 label codes (pm.H, pm.A, pm.D, pm.X)
    tagging: pd.DataFrame     # one row per Smart Tagging instance, frames half-open
    pitch: tuple              # (length, width) in metres
    touch_ball: np.ndarray = None  # (n, 2) optional second ball track (training copies only)

    @property
    def n_frames(self):
        return len(self.ball)

    @property
    def graded(self):
        return self.half > 0

    @property
    def ball_seen(self):
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_seen(self):
        return ~np.isnan(self.players[:, :, 0])


def _source_stamp(paths):
    """Sizes and mtimes of the source files, so a changed download invalidates the cache."""
    return np.array([[os.path.getsize(p), int(os.path.getmtime(p))] for p in paths], dtype=np.int64)


def parse_atd(raw_path, n_frames, pitch):
    """(players, player_speed, ball, ball_speed) from the raw ATD text, indexed by video frame.

    The file is "frame_id:x,y,s;x,y,s;...:x,y,s" per line. Mapping both separators to commas turns
    it into a plain CSV that pandas' C parser reads in seconds, instead of a minute in pure Python.
    """
    with open(raw_path) as handle:
        text = handle.read().translate(str.maketrans(":;", ",,"))
    values = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).values
    del text

    n_cols = 1 + 3 * N_SLOTS + 3
    if values.shape[1] != n_cols:
        raise ValueError("expected {} columns, got {}".format(n_cols, values.shape[1]))

    frame_ids = values[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame ids are not 1..{} in order".format(n_frames))

    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    slots = values[:, 1:1 + 3 * N_SLOTS].reshape(n_frames, N_SLOTS, 3)
    players = np.ascontiguousarray(slots[:, :, :2] * scale)
    player_speed = np.ascontiguousarray(slots[:, :, 2])
    ball = np.ascontiguousarray(values[:, -3:-1] * scale)
    ball_speed = np.ascontiguousarray(values[:, -1])
    return players, player_speed, ball, ball_speed


def parse_tagging(game=GAME):
    """Smart Tagging as a DataFrame with start/end video frames and normalised tag columns.

    team is "H", "A" or "" (the file's "N/A", which covers every throw-in).
    """
    rows = []
    home_names = {"barcelona"}
    for event in load_smart_tagging.load(game):
        tags = {key.lower(): (value or "") for key, value in event["tags"].items()}
        team = tags.get("team", "").strip().lower()
        rows.append({
            "code": event["code"].strip().upper(),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "team": "H" if team in home_names else ("" if team in ("", "n/a") else "A"),
            "half": tags.get("half", ""),
            "type": tags.get("type", ""),
            "side": tags.get("side", ""),
            "entry": tags.get("direction of ball entry", ""),
            "max_in_box": tags.get("max players in the box", ""),
        })
    return pd.DataFrame(rows).sort_values("start_frame", ignore_index=True)


def half_array(n_frames, game=GAME):
    """(n,) int8: 1 or 2 on graded frames of that half, 0 on ungraded frames."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def build_cache(game=GAME, cache_dir=CACHE_DIR):
    """Parse everything and write cache_dir/{game}.npz and {game}_tagging.csv."""
    meta_path, raw_path, homography_path, tagging_path = require(
        ["atd_meta", "atd_raw", "homography", "smart_tagging"], game)
    n_frames = GAMES[game]["n_video_frames"]
    pitch = GAMES[game]["pitch"]

    players, player_speed, ball, ball_speed = parse_atd(raw_path, n_frames, pitch)
    team = team_of_each_track(meta_path)
    if len(team) != N_SLOTS:
        raise ValueError("expected {} slots in the metadata, got {}".format(N_SLOTS, len(team)))

    matrices, reproj_error, _ = load_homography.load(game)
    if len(matrices) != n_frames:
        raise ValueError("homography has {} rows, expected {}".format(len(matrices), n_frames))

    gt_public = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    tagging = parse_tagging(game)

    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, game)
    np.savez(
        base + ".npz",
        version=CACHE_VERSION,
        stamp=_source_stamp([meta_path, raw_path, homography_path, tagging_path]),
        players=players, player_speed=player_speed, team=team,
        ball=ball, ball_speed=ball_speed,
        half=half_array(n_frames, game),
        homography=matrices.astype(np.float32), reproj_error=reproj_error.astype(np.float32),
        gt_public=gt_public, pitch=np.array(pitch),
    )
    tagging.to_csv(base + "_tagging.csv", index=False)
    return base + ".npz"


def _cache_is_fresh(npz, game):
    if int(npz["version"]) != CACHE_VERSION:
        return False
    paths = require(["atd_meta", "atd_raw", "homography", "smart_tagging"], game)
    return np.array_equal(npz["stamp"], _source_stamp(paths))


def load_match(game=GAME, cache_dir=CACHE_DIR, rebuild=False):
    """The cached Match, building it first if it is missing, stale or rebuild=True."""
    path = os.path.join(cache_dir, game + ".npz")
    if not rebuild and os.path.exists(path):
        npz = np.load(path)
        if not _cache_is_fresh(npz, game):
            rebuild = True
    else:
        rebuild = True
    if rebuild:
        build_cache(game, cache_dir)
        npz = np.load(path)

    tagging = pd.read_csv(os.path.join(cache_dir, game + "_tagging.csv"), keep_default_na=False)
    return Match(
        players=npz["players"], player_speed=npz["player_speed"], team=npz["team"],
        ball=npz["ball"], ball_speed=npz["ball_speed"], half=npz["half"],
        homography=npz["homography"], reproj_error=npz["reproj_error"],
        gt_public=npz["gt_public"], tagging=tagging, pitch=tuple(npz["pitch"].tolist()),
    )
