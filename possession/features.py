"""Per-frame features from player and ball positions (the input of the per-frame models).

The same function is used for this match and for the training matches, so the models see the
same kind of numbers. Only positions are used: no ATD speed field, no camera geometry, no labels.

Positions are turned so that +x always points at the away team's goal, in both halves. Then
"moving forward" means the same thing everywhere.

Groups of columns:
  BALL_COLS   anything computed from the ball: distance from each team to the ball, ball speed,
              whether the ball was last seen outside the pitch, what we knew at the edges of a gap
              in the ball tracking, ...
  TAG_COLS    Smart Tagging (only this match has it; the per-frame models do not use it).
  the rest    team shape and movement, which exist with or without a ball.

    X = build_features(m)          # DataFrame, one row per frame
"""

import warnings

import numpy as np
import pandas as pd

from possession.data import HOME, AWAY

FPS = 25

# Params
CONTROL_RADIUS = 1.0      # m: a player this close to the ball, and nearer than every opponent, "controls"
                          # it. Chosen on held-out training matches (1.0 beat 1.5 and 2.0), not on this match
NEAR_RADIUS = 5.0         # m, players counted around the ball
MAX_PLAYER_SPEED = 12.0   # m/s; faster steps are slot swaps, not running
SLOW_SPEED = 1.0          # m/s, "standing" for frac_slow
DIST_CAP = 30.0           # m, distances beyond this carry no extra information
TIME_CAP = 60.0           # s, for time-since / time-until features
WINDOWS_S = (1, 3, 10)    # s; centred windows of ±w seconds (column suffix _c{w})
SIDE_WINDOW_S = 3         # past-only and future-only windows
OFFENSIVE_CODES = {"ATTACKING TRANSITION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "BUILD UP",
                   "CREATING CHANCES", "PROGRESSION", "LONG BALL"}
DEFENSIVE_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD",
                   "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION"}

# base columns that also get rolling-window versions
WINDOWED = ["ctrl", "near_team", "d_margin", "ball_seen", "ball_speed", "ball_out", "bx",
            "speed_all", "frac_slow", "h_vfwd", "a_vfwd", "cx_gap", "h_spread", "a_spread"]
SIDED = ["ctrl", "near_team", "speed_all", "ball_seen"]
_BALL_BASE = ["ball_seen", "bx", "by_abs", "ball_speed", "ball_out", "d_h", "d_a", "d_margin",
              "ctrl", "near_team", "n_h_near", "n_a_near",
              "t_since_seen", "t_until_seen", "gap_len", "last_out",
              "last_ctrl", "t_since_ctrl", "next_ctrl", "t_until_ctrl", "last_bx", "next_bx"]


# -- helpers -----------------------------------------------------------------------------------

def _segments(m):
    """Contiguous runs of equal m.half, so rolling windows never cross a half or halftime."""
    change = np.flatnonzero(np.diff(m.half)) + 1
    return list(zip(np.r_[0, change], np.r_[change, m.n_frames]))


def _roll(values, m, seconds, side="centre"):
    """Rolling mean ignoring NaN, per half. side: centre, past (trailing) or future (leading)."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    frames = int(seconds * FPS)
    for start, stop in _segments(m):
        s = pd.Series(values[start:stop])
        if side == "centre":
            r = s.rolling(2 * frames + 1, center=True, min_periods=1).mean()
        elif side == "past":
            r = s.rolling(frames, min_periods=1).mean()
        else:
            r = s[::-1].rolling(frames, min_periods=1).mean()[::-1]
        out[start:stop] = r.values
    return out


def _nanstat(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: NaN is the right answer
        return fn(a, axis=axis)


def _last_value(values, valid):
    """Per frame: the most recent value where valid, NaN before the first one."""
    idx = np.where(valid, np.arange(len(values)), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
    return out, idx


def _next_value(values, valid):
    """Per frame: the next value where valid (this frame included), NaN after the last one."""
    rev, rev_idx = _last_value(values[::-1], valid[::-1])
    n = len(values)
    idx = np.where(rev_idx[::-1] >= 0, n - 1 - rev_idx[::-1], -1)
    return rev[::-1], idx


def attack_sign(m):
    """(n,) +1 where home attacks towards +x, -1 where towards -x; per half, from team centroids."""
    length = m.pitch[0]
    sign = np.ones(m.n_frames, dtype=np.float32)
    for start, stop in _segments(m):
        xs = m.players[start:stop, :, 0]
        home = _nanstat(np.nanmean, xs[:, m.team == HOME])
        away = _nanstat(np.nanmean, xs[:, m.team == AWAY])
        # home defends the side its centroid sits on, so it attacks the other way
        sign[start:stop] = 1.0 if np.nanmean(home) < np.nanmean(away) else -1.0
    return sign, length / 2


# -- feature groups ----------------------------------------------------------------------------

def shape_features(m, sign, mid):
    """Team shape and collective motion; no ball needed."""
    x = (m.players[:, :, 0] - mid) * sign[:, None]   # home-attack coordinates
    y = m.players[:, :, 1]
    vx = np.full_like(x, np.nan)
    vx[1:] = np.diff(x, axis=0) * FPS
    vy = np.full_like(y, np.nan)
    vy[1:] = np.diff(y, axis=0) * FPS
    speed = np.hypot(vx, vy)
    bad = speed > MAX_PLAYER_SPEED                    # slot swaps teleport players
    vx[bad] = np.nan
    speed[bad] = np.nan

    f = {}
    for prefix, mask in (("h", m.team == HOME), ("a", m.team == AWAY)):
        xs, ys = x[:, mask], y[:, mask]
        f[prefix + "_n"] = np.sum(~np.isnan(xs), axis=1).astype(np.float32)
        f[prefix + "_cx"] = _nanstat(np.nanmean, xs)
        f[prefix + "_spread"] = np.sqrt(_nanstat(np.nanvar, xs) + _nanstat(np.nanvar, ys))
        f[prefix + "_spread_x"] = _nanstat(np.nanstd, xs)
        f[prefix + "_spread_y"] = _nanstat(np.nanstd, ys)
        f[prefix + "_vfwd"] = _nanstat(np.nanmean, vx[:, mask])
        f[prefix + "_speed"] = _nanstat(np.nanmean, speed[:, mask])
    # defensive line: home's second-deepest player (lowest x'), away's second-deepest (highest x')
    hx = np.sort(np.where(np.isnan(x[:, m.team == HOME]), np.inf, x[:, m.team == HOME]), axis=1)
    ax = np.sort(np.where(np.isnan(x[:, m.team == AWAY]), -np.inf, x[:, m.team == AWAY]), axis=1)
    f["h_line"] = np.where(np.isfinite(hx[:, 1]), hx[:, 1], np.nan)
    f["a_line"] = np.where(np.isfinite(ax[:, -2]), ax[:, -2], np.nan)
    f["cx_gap"] = f["h_cx"] - f["a_cx"]
    f["speed_all"] = _nanstat(np.nanmean, speed)
    f["frac_slow"] = _nanstat(np.nanmean, np.where(np.isnan(speed), np.nan, speed < SLOW_SPEED))
    f["speed_change_1s"] = f["speed_all"] - _roll(f["speed_all"], m, 1, "past")
    return f


def ball_features(m, sign, mid):
    """Ball state and ball <-> players, plus what we knew at the edges of the current ball gap."""
    length, width = m.pitch
    seen = ~np.isnan(m.ball[:, 0])
    bx = (m.ball[:, 0] - mid) * sign
    by = m.ball[:, 1]

    f = {"ball_seen": seen.astype(np.float32), "bx": bx, "by_abs": np.abs(by - width / 2)}
    step = np.full(m.n_frames, np.nan)
    step[1:] = np.hypot(np.diff(m.ball[:, 0]), np.diff(m.ball[:, 1])) * FPS
    f["ball_speed"] = pd.Series(step).rolling(5, center=True, min_periods=3).mean().values
    x, yy = m.ball[:, 0], m.ball[:, 1]
    out = np.maximum.reduce([-x, x - length, -yy, yy - width, np.zeros_like(x)])
    f["ball_out"] = np.where(seen, out, np.nan)

    dist = np.linalg.norm(m.players - m.ball[:, None, :], axis=2)
    d_h = _nanstat(np.nanmin, dist[:, m.team == HOME])
    d_a = _nanstat(np.nanmin, dist[:, m.team == AWAY])
    f["d_h"] = np.minimum(d_h, DIST_CAP)
    f["d_a"] = np.minimum(d_a, DIST_CAP)
    f["d_margin"] = f["d_a"] - f["d_h"]               # > 0: home closer
    home_ctrl = (d_h <= CONTROL_RADIUS) & (d_h < d_a)
    away_ctrl = (d_a <= CONTROL_RADIUS) & (d_a < d_h)
    f["ctrl"] = np.where(seen, home_ctrl.astype(np.float32) - away_ctrl, np.nan)
    f["near_team"] = np.where(seen, np.sign(f["d_margin"]), np.nan)
    with np.errstate(invalid="ignore"):
        f["n_h_near"] = np.where(seen, np.sum(dist[:, m.team == HOME] <= NEAR_RADIUS, axis=1), np.nan)
        f["n_a_near"] = np.where(seen, np.sum(dist[:, m.team == AWAY] <= NEAR_RADIUS, axis=1), np.nan)

    # gap bridge: time since / until the ball was seen, and what it was doing at either edge
    frames = np.arange(m.n_frames)
    _, last_seen = _last_value(frames.astype(float), seen)
    _, next_seen = _next_value(frames.astype(float), seen)
    t_since = np.where(last_seen >= 0, (frames - last_seen) / FPS, TIME_CAP)
    t_until = np.where(next_seen >= 0, (next_seen - frames) / FPS, TIME_CAP)
    f["t_since_seen"] = np.minimum(t_since, TIME_CAP)
    f["t_until_seen"] = np.minimum(t_until, TIME_CAP)
    f["gap_len"] = np.where(seen, 0.0, np.minimum(t_since + t_until, 2 * TIME_CAP))
    f["last_out"], _ = _last_value(np.nan_to_num(f["ball_out"]), seen)
    f["last_bx"], _ = _last_value(bx, seen)
    f["next_bx"], _ = _next_value(bx, seen)
    controlled = f["ctrl"] != 0
    controlled &= seen
    f["last_ctrl"], last_c = _last_value(np.nan_to_num(f["ctrl"]), controlled)
    f["next_ctrl"], next_c = _next_value(np.nan_to_num(f["ctrl"]), controlled)
    f["t_since_ctrl"] = np.minimum(np.where(last_c >= 0, (frames - last_c) / FPS, TIME_CAP), TIME_CAP)
    f["t_until_ctrl"] = np.minimum(np.where(next_c >= 0, (next_c - frames) / FPS, TIME_CAP), TIME_CAP)
    return f


def tagging_features(m):
    """Smart Tagging as per-frame evidence. All NaN when the match has no tagging."""
    n = m.n_frames
    if m.tagging is None or len(m.tagging) == 0:
        return {c: np.full(n, np.nan, dtype=np.float32) for c in TAG_COLS}
    team = np.zeros(n, dtype=np.float32)
    since_sp = np.full(n, np.inf)
    phase = np.zeros(n, dtype=bool)
    for row in m.tagging.itertuples():
        sl = slice(row.start_frame, row.end_frame)
        if row.code != "SET PIECES":
            phase[sl] = True
        if row.team in ("H", "A") and (row.code in OFFENSIVE_CODES or row.code in DEFENSIVE_CODES):
            vote = 1.0 if row.team == "H" else -1.0
            team[sl] += -vote if row.code in DEFENSIVE_CODES else vote   # defending codes name the defender
        if row.code == "SET PIECES":
            frames = np.arange(row.start_frame, min(n, row.start_frame + int(TIME_CAP * FPS)))
            since_sp[frames] = np.minimum(since_sp[frames], (frames - row.start_frame) / FPS)
    # tag_gap: no phase tag covers the frame. Smart Tagging cuts its phases when play stops, so
    # gaps between phases tend to be stoppages; the models learn how much to trust that.
    return {"tag_team": np.sign(team),
            "tag_t_since_setpiece": np.minimum(since_sp, TIME_CAP).astype(np.float32),
            "tag_gap": (~phase).astype(np.float32)}


TAG_COLS = ["tag_team", "tag_t_since_setpiece", "tag_gap"]


# -- top level ---------------------------------------------------------------------------------

def build_features(m, tagging=True):
    """(n_frames, n_features) float32 DataFrame. Rows are video frames, like every other array."""
    sign, mid = attack_sign(m)
    f = {}
    f.update(shape_features(m, sign, mid))
    f.update(ball_features(m, sign, mid))
    for col in WINDOWED:
        for w in WINDOWS_S:
            f["{}_c{}".format(col, w)] = _roll(f[col], m, w)
    for col in SIDED:
        f["{}_past{}".format(col, SIDE_WINDOW_S)] = _roll(f[col], m, SIDE_WINDOW_S, "past")
        f["{}_fut{}".format(col, SIDE_WINDOW_S)] = _roll(f[col], m, SIDE_WINDOW_S, "future")
    if tagging:
        f.update(tagging_features(m))
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def _ball_cols():
    windowed = ["{}_c{}".format(c, w) for c in WINDOWED for w in WINDOWS_S if c in _BALL_BASE]
    sided = ["{}_{}{}".format(c, s, SIDE_WINDOW_S) for c in SIDED if c in _BALL_BASE
             for s in ("past", "fut")]
    return _BALL_BASE + windowed + sided


BALL_COLS = _ball_cols()
