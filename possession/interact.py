"""Interaction features: who will reach the ball, and how the teams press and mark.

    interaction_features(m, X)   DataFrame, one row per frame

The basic features say who is nearest the ball now. These say who will be nearest where the ball
is going (0.5 s and 1 s ahead at its current velocity), how tightly each team marks the other,
and how the teams' shapes change over 2 seconds.
"""

import numpy as np
import pandas as pd

import possession_metrics as pm
from possession.data import HOME, AWAY

FPS = 25
HORIZONS_S = (0.5, 1.0)
DIST_CAP = 30.0
TREND = 2 * FPS
MAX_PLAYER_SPEED = 12.0


def _vel(p):
    """Central-difference velocity (m/s) of an (n, ..., 2) track, 3-frame smoothed along time."""
    v = np.full_like(p, np.nan, dtype=float)
    v[1:-1] = (p[2:] - p[:-2]) * FPS / 2
    return v


def _nearest(pts, target, mask):
    """(n,) distance and slot of the nearest player of `mask` to target (n, 2)."""
    d = np.linalg.norm(pts[:, mask] - target[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), j]
    return np.where(np.isfinite(dist), np.minimum(dist, DIST_CAP), np.nan), np.flatnonzero(mask)[j]


def interaction_features(m, X):
    players = m.players.astype(float)
    ball = m.ball.astype(float)
    n = len(ball)
    vb = pd.DataFrame(_vel(ball)).rolling(3, center=True, min_periods=2).mean().values
    vp = _vel(players)
    vp[np.linalg.norm(vp, axis=2) > MAX_PLAYER_SPEED] = np.nan
    home, away = m.team == HOME, m.team == AWAY
    f = {}

    # who reaches where the ball is going
    for h in HORIZONS_S:
        target = ball + vb * h
        dh, _ = _nearest(players, target, home)
        da, _ = _nearest(players, target, away)
        tag = str(h).replace(".", "")
        f["ix_proj{}_dh".format(tag)], f["ix_proj{}_da".format(tag)] = dh, da
        f["ix_proj{}_margin".format(tag)] = da - dh
        f["ix_proj{}_switch".format(tag)] = (np.sign(da - dh) * np.sign(X["d_margin"].to_numpy(float)) < 0).astype(float)

    # ball speed relative to the nearest player of each team (control vs passing by)
    rows = np.arange(n)
    for name, mask in (("h", home), ("a", away)):
        _, j = _nearest(players, ball, mask)
        rel = np.linalg.norm(vb - vp[rows, j], axis=1)
        f["ix_relspeed_{}".format(name)] = rel
    near_h = X["d_h"].to_numpy(float) <= X["d_a"].to_numpy(float)
    f["ix_relspeed_near"] = np.where(near_h, f["ix_relspeed_h"], f["ix_relspeed_a"])
    acc = np.full(n, np.nan)
    sp = np.hypot(vb[:, 0], vb[:, 1])
    acc[1:-1] = (sp[2:] - sp[:-2]) * FPS / 2
    f["ix_ball_acc"] = acc

    # marking: each team's mean distance to its nearest opponent
    ph, pa = players[:, home].astype(np.float32), players[:, away].astype(np.float32)
    f["ix_mark_h"], f["ix_mark_a"] = np.full(n, np.nan), np.full(n, np.nan)
    for a in range(0, n, 10000):                                             # chunks: (k, 23, 23) at a time
        d = np.linalg.norm(ph[a:a + 10000, :, None, :] - pa[a:a + 10000, None, :, :], axis=3)
        d = np.where(np.isnan(d), np.inf, d)
        with np.errstate(all="ignore"):
            for key, dmin in (("ix_mark_h", d.min(axis=2)), ("ix_mark_a", d.min(axis=1))):
                f[key][a:a + 10000] = np.nanmean(np.where(np.isfinite(dmin), dmin, np.nan), axis=1)
    f["ix_mark_diff"] = f["ix_mark_h"] - f["ix_mark_a"]

    # 2 s shape trends
    for col in ("h_cx", "a_cx", "h_spread", "a_spread", "cx_gap", "h_line", "a_line"):
        x = X[col].to_numpy(float)
        f["ix_trend_{}".format(col)] = x - pd.Series(x).shift(TREND).values
    for col in ("ix_mark_h", "ix_mark_a"):
        f[col + "_trend"] = f[col] - pd.Series(f[col]).shift(TREND).values
    return pd.DataFrame(f).astype(np.float32)


# -- restart team ---------------------------------------------------------------------------------


