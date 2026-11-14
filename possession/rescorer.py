"""Features of each candidate possession for the rescorer (the model that scores candidates).

  evidence      sums, minima and shares of the per-frame evidence for the candidate's team against
                the other team and against a dead ball, plus 10-frame windows at both edges.
  best path     what the best path says just before and after the candidate, and how far each end
                is from a boundary or a stoppage of the best path.
  competition   the overlapping candidates of each team: how many, and how well supported.

Inputs: an (n, 3) array of log-probabilities for H, A and D, the candidates (k, 3)
[start, stop, label] sorted by start, the best path, and decoder.spell_features' table.
"""

import numpy as np

import possession_metrics as pm  # noqa: E402
from possession import decoder as dm

REL_SAME, REL_OTHER, REL_DEAD, REL_EDGE = 0, 1, 2, 3


def _window_reduce(ufunc, x, starts, stops):
    """ufunc.reduce of x over each [start, stop) (start < stop), via reduceat on paired indices."""
    xp = np.concatenate((x, x[-1:]))                     # stop may equal len(x)
    idx = np.column_stack([starts, stops]).ravel()
    return ufunc.reduceat(xp, idx)[::2]


def _rel(code, team):
    """Code of a neighbouring frame relative to a candidate's team."""
    return np.where(code == pm.D, REL_DEAD, np.where(code == team, REL_SAME,
                    np.where((code == pm.H) | (code == pm.A), REL_OTHER, REL_EDGE)))


def overlap_index(cands):
    """(i, j) for every ordered pair of distinct overlapping candidates (any team)."""
    s, t = cands[:, 0], cands[:, 1]
    hi = np.searchsorted(s, t, side="left")              # candidates j with start_j < stop_i
    lo = np.arange(len(cands)) + 1
    cnt = np.maximum(hi - lo, 0)
    i = np.repeat(np.arange(len(cands)), cnt)
    j = lo[i] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    keep = s[j] < t[i]                                   # sorted by start: j starts after i
    i, j = i[keep], j[keep]
    return np.concatenate([i, j]), np.concatenate([j, i])


def _group_max(values, i, n, fill=0.0):
    out = np.full(n, fill)
    np.maximum.at(out, i, values)
    return out


def competition_features(cands, value, prefix, pairs=None):
    """Features of each candidate's overlapping rivals under a per-candidate score `value`."""
    i, j = pairs if pairs is not None else overlap_index(cands)
    n = len(cands)
    same = cands[i, 2] == cands[j, 2]
    f = {}
    for tag, m in (("same", same), ("other", ~same)):
        f["{}_{}_max".format(prefix, tag)] = _group_max(value[j[m]], i[m], n)
        f["{}_{}_n".format(prefix, tag)] = np.log1p(np.bincount(i[m], minlength=n))
        f["{}_{}_better".format(prefix, tag)] = np.bincount(i[m], weights=value[j[m]] > value[i[m]],
                                                            minlength=n)
    f["{}_margin".format(prefix)] = value - np.maximum(f[prefix + "_same_max"], f[prefix + "_other_max"])
    return f


def extra_features(cands, emission, map_segs, X, n_frames):
    """Extra candidate features (dict of (k,) arrays), added to decoder.spell_features' table X."""
    e = np.asarray(emission, dtype=float)
    s, t, c = cands[:, 0], cands[:, 1], cands[:, 2]
    home = c == pm.H
    dur = np.maximum(t - s, 1)
    lo_ha = e[:, 0] - e[:, 1]
    alive = np.logaddexp(e[:, 0], e[:, 1])
    lo_alive = alive - e[:, 2]
    f = {}
    # evidence: team vs other team, team vs D (the candidate's own team's log-prob minus the rival's)
    for name, x_h, x_a in (("lo_other", lo_ha, -lo_ha), ("lo_dead", e[:, 0] - e[:, 2], e[:, 1] - e[:, 2])):
        cs_h, cs_a = np.concatenate(([0.0], np.cumsum(x_h))), np.concatenate(([0.0], np.cumsum(x_a)))
        tot = np.where(home, cs_h[t] - cs_h[s], cs_a[t] - cs_a[s])
        f[name + "_sum"] = np.sign(tot) * np.log1p(np.abs(tot))
        f[name + "_mean"] = tot / dur
        mn_h = _window_reduce(np.minimum, np.convolve(x_h, np.ones(5) / 5, "same"), s, t)
        mn_a = _window_reduce(np.minimum, np.convolve(x_a, np.ones(5) / 5, "same"), s, t)
        f[name + "_min5"] = np.clip(np.where(home, mn_h, mn_a), -30, 30)
    arg = np.argmax(e, axis=1)
    for name, k in (("argmax_team", None), ("argmax_dead", 2)):
        if k is None:
            ind_h, ind_a = (arg == 0).astype(float), (arg == 1).astype(float)
            f[name] = np.where(home, dm._window_means(ind_h, s, t), dm._window_means(ind_a, s, t))
        else:
            f[name] = dm._window_means((arg == 2).astype(float), s, t)
    p = np.exp(e - np.logaddexp.reduce(e, axis=1, keepdims=True))
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)
    f["entropy_in"] = dm._window_means(ent, s, t)
    p_alive = 1 / (1 + np.exp(-np.clip(lo_alive, -50, 50)))
    p_h = 1 / (1 + np.exp(-np.clip(lo_ha, -50, 50)))
    for tag, a, b in (("pre10", np.maximum(s - 10, 0), s), ("post10", t, np.minimum(t + 10, n_frames)),
                      ("head10", s, np.minimum(s + 10, t)), ("tail10", np.maximum(t - 10, s), t),
                      ("pre150", np.maximum(s - 150, 0), s), ("post150", t, np.minimum(t + 150, n_frames))):
        v = dm._window_means(p_h, a, b)
        f["p_team_" + tag] = np.where(home, v, 1 - v)
        f["p_alive_" + tag] = dm._window_means(p_alive, a, b)
    f["jump_start"] = f["p_team_head10"] - f["p_team_pre10"]
    f["jump_end"] = f["p_team_tail10"] - f["p_team_post10"]
    f["dead_jump_start"] = f["p_alive_head10"] - f["p_alive_pre10"]
    f["dead_jump_end"] = f["p_alive_tail10"] - f["p_alive_post10"]
    # MAP context
    mapc = dm.segments_to_codes(map_segs, n_frames)
    before = np.where(s > 0, mapc[np.maximum(s - 1, 0)], pm.X)
    after = np.where(t < n_frames, mapc[np.minimum(t, n_frames - 1)], pm.X)
    f["map_before"] = _rel(before, c)
    f["map_after"] = _rel(after, c)
    bounds = np.unique(np.concatenate([map_segs[:, 0], map_segs[:, 1]]))
    for tag, x in (("start", s), ("stop", t)):
        k = np.clip(np.searchsorted(bounds, x), 1, len(bounds) - 1)
        d = np.minimum(np.abs(x - bounds[k - 1]), np.abs(bounds[k] - x))
        f["map_dist_" + tag] = np.log1p(d)
    dead = map_segs[map_segs[:, 2] == pm.D]
    k = np.searchsorted(dead[:, 1], s, side="right") - 1     # last MAP stoppage ending at or before s
    f["since_dead"] = np.log1p(np.where(k >= 0, s - dead[np.maximum(k, 0), 1], 1e5))
    k = np.searchsorted(dead[:, 0], t, side="left")           # first MAP stoppage starting at or after t
    f["until_dead"] = np.log1p(np.where(k < len(dead), dead[np.minimum(k, len(dead) - 1), 0] - t, 1e5))
    f["map_contains"] = np.log1p(np.searchsorted(bounds, t, "left") - np.searchsorted(bounds, s, "right"))
    # competition: overlapping candidates, ranked by how often the samples contain them
    pairs = overlap_index(cands)
    f.update(competition_features(cands, X["s0_q"].values, "s0", pairs))
    f.update(competition_features(cands, X["s1_q"].values, "s1", pairs))
    return f


