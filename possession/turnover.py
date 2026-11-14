"""The switch head: the chance that possession changes within 0.5 s of a frame.

    switch_target(labels)                 (n,) 1 near a change of team, 0 otherwise, -1 unlabelled
                                          (what the switch head was trained to predict)
    turnover_features(m, X)               extra columns that compare the second before a frame
                                          with the second after it
    candidate_switch_features(cands, p)   per candidate possession: the switch probability at its
                                          ends, before, after and inside it (read by the rescorer)

A change of possession is a change over time, so these features compare before and after: which
team is nearest the ball, ball speed and direction, ball in flight, and the teams' shape.
"""

import numpy as np
import pandas as pd

import possession_metrics as pm

FPS = 25
HALF_WIN = 12             # frames: a switch "near" a frame = within +-0.5 s
MAX_DEAD_GAP = 2 * FPS    # H -> short D -> A counts as a switch (the D is a quick stop)
SIDE = (12, 25)           # frames compared before vs after
FREE_M = 2.0              # m: ball farther than this from every player = in flight
TURN_WIN = 10             # frames: window for the peak ball turn / speed change


def switch_target(labels):
    """1 within HALF_WIN frames of an H<->A switch (or of a <= 2 s stoppage between the two
    teams), 0 on other labelled frames, -1 where the label is X."""
    labels = np.asarray(labels)
    y = np.where(labels == pm.X, -1, 0).astype(np.int8)
    for start, stop in pm.valid_runs(labels):
        segs = [s for s in pm.segments(labels, [(start, stop)]) if s[2] in (pm.H, pm.A)]
        for (_, e0, c0), (s1, _, c1) in zip(segs[:-1], segs[1:]):
            if c0 != c1 and s1 - e0 <= MAX_DEAD_GAP:
                a, b = max(start, e0 - HALF_WIN), min(stop, s1 + HALF_WIN)
                y[a:b] = 1
    return y


def _pre(x, w):
    """Mean over the w frames before each frame (NaNs skipped)."""
    return pd.Series(x).rolling(w, min_periods=1).mean().shift(1).values


def _post(x, w):
    """Mean over the w frames after each frame."""
    return pd.Series(x[::-1]).rolling(w, min_periods=1).mean().shift(1).values[::-1]


def _since_change(sign):
    """Frames since the last change of a +-1 series (NaN carried forward), and until the next."""
    s = pd.Series(sign).ffill().values
    n = len(s)
    ch = np.zeros(n, bool)
    ch[1:] = (s[1:] != s[:-1]) & ~np.isnan(s[1:]) & ~np.isnan(s[:-1])
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(ch, idx, -1))
    nxt = np.minimum.accumulate(np.where(ch, idx, n)[::-1])[::-1]
    cap = 10 * FPS
    return (np.minimum(np.where(last >= 0, idx - last, cap), cap),
            np.minimum(np.where(nxt < n, nxt - idx, cap), cap))


def turnover_features(m, X):
    """(n, k) float32 change-over-time columns from a Match and its 92-column frame features."""
    f = {}
    for col in ("d_margin", "near_team", "ctrl", "cx_gap", "h_vfwd", "a_vfwd", "ball_speed", "bx"):
        x = X[col].to_numpy(dtype=float)
        for w in SIDE:
            pre, post = _pre(x, w), _post(x, w)
            f["to_{}_d{}".format(col, w)] = post - pre
            if col in ("d_margin", "near_team", "ctrl"):
                f["to_{}_flip{}".format(col, w)] = (np.sign(pre) * np.sign(post) < 0).astype(float)
                f["to_{}_pre{}".format(col, w)] = pre
                f["to_{}_post{}".format(col, w)] = post
    since, until = _since_change(np.sign(X["d_margin"].to_numpy(dtype=float)))
    f["to_near_since"], f["to_near_until"] = since, until

    # ball kinematics: peak speed change and direction change in a +-TURN_WIN window
    b = m.ball.astype(float)
    v = np.full_like(b, np.nan)
    v[1:-1] = (b[2:] - b[:-2]) * FPS / 2
    v = pd.DataFrame(v).rolling(3, center=True, min_periods=2).mean().values
    sp = np.hypot(v[:, 0], v[:, 1])
    ang = np.arctan2(v[:, 1], v[:, 0])
    k = 4                                                    # frames between compared velocities
    turn = np.full(len(b), np.nan)
    dang = np.abs(np.angle(np.exp(1j * (ang[2 * k:] - ang[:-2 * k]))))
    ok = (sp[2 * k:] > 2) & (sp[:-2 * k] > 2)
    turn[k:-k] = np.where(ok, dang, 0.0)
    dsp = np.full(len(b), np.nan)
    dsp[k:-k] = np.abs(sp[2 * k:] - sp[:-2 * k])
    roll = lambda x: pd.Series(x).rolling(2 * TURN_WIN + 1, center=True, min_periods=1).max().values
    f["to_ball_turn"], f["to_ball_dspeed"] = roll(turn), roll(dsp)

    # free flight: ball seen and no player within FREE_M
    near = np.minimum(X["d_h"].to_numpy(dtype=float), X["d_a"].to_numpy(dtype=float))
    free = (near > FREE_M) & ~np.isnan(b[:, 0])
    run_id = np.cumsum(np.r_[True, free[1:] != free[:-1]])
    run_len = pd.Series(free).groupby(run_id).transform("size").values
    flen = np.where(free, run_len, 0).astype(float)
    f["to_free_len"] = flen
    f["to_free_max25"] = pd.Series(flen).rolling(51, center=True, min_periods=1).max().values
    return pd.DataFrame(f).astype(np.float32)


# -- rescorer side ------------------------------------------------------------------------------

class _RangeMax:
    """O(1) range max over a fixed array (sparse table)."""

    def __init__(self, x):
        self.t = [np.asarray(x, float)]
        j = 1
        while 2 * j <= len(x):
            p = self.t[-1]
            self.t.append(np.maximum(p[:-j], p[j:]))
            j *= 2

    def __call__(self, a, b):
        """max over [a, b) per pair; -inf where empty."""
        a, b = np.asarray(a), np.asarray(b)
        n = np.maximum(b - a, 1)
        k = np.floor(np.log2(n)).astype(int)
        out = np.full(len(a), -np.inf)
        for lev in np.unique(k):
            i = (k == lev) & (b > a)
            t = self.t[lev]
            out[i] = np.maximum(t[a[i]], t[b[i] - (1 << lev)])
        return np.where(b > a, out, -np.inf)


def candidate_switch_features(cands, p):
    """Per candidate (start, stop, team): switch probability at each end, inside it, and where the
    nearest peak sits relative to each end."""
    p = np.nan_to_num(np.asarray(p, float))
    n = len(p)
    rmax = _RangeMax(p)
    cs = np.concatenate(([0.0], np.cumsum(p)))
    s, t = cands[:, 0].astype(int), cands[:, 1].astype(int)
    clip = lambda x: np.clip(x, 0, n)
    f = {}
    for name, e in (("start", s), ("end", t)):
        f["sw_{}".format(name)] = rmax(clip(e - HALF_WIN), clip(e + HALF_WIN))
        f["sw_{}_wide".format(name)] = rmax(clip(e - 3 * HALF_WIN), clip(e + 3 * HALF_WIN))
        f["sw_{}_at".format(name)] = p[np.clip(e, 0, n - 1)]
    f["sw_pre"] = rmax(clip(s - 75), clip(s - HALF_WIN))           # a stronger switch just before
    f["sw_post"] = rmax(clip(t + HALF_WIN), clip(t + 75))
    a, b = clip(s + 2 * HALF_WIN), clip(t - 2 * HALF_WIN)
    inner = rmax(a, b)
    f["sw_in_max"] = np.where(b > a, inner, 0.0)
    f["sw_in_mean"] = np.where(b > a, (cs[b] - cs[np.minimum(a, b)]) / np.maximum(b - a, 1), 0.0)
    f["sw_ends_min"] = np.minimum(f["sw_start"], f["sw_end"])
    f["sw_ends_vs_in"] = f["sw_ends_min"] - f["sw_in_max"]
    out = pd.DataFrame(f)
    return out.replace(-np.inf, 0.0).astype(np.float32)
