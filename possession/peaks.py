"""Extra candidate possessions cut where the switch head peaks.

The duration model's own samples mostly agree with its best path, so a possession the best path
misses rarely becomes a candidate (in held-out training matches, 25% of real possessions had no
close candidate, most of them shorter than 1 s). Here every peak of the switch probability is a
possible boundary, as is every boundary of the best path. Each stretch between a boundary and one
of the next few becomes a candidate for the team the per-frame models favour there, unless the
stretch looks like a dead ball.

    peak_candidates(p_switch, emission, map_segs, graded) -> (k, 3) array [start, stop, label]
"""

import numpy as np
from scipy.signal import find_peaks

import possession_metrics as pm

MIN_HEIGHT = 0.2       # p_switch peak height
MIN_DIST = 6           # frames between peaks
SPAN = 3               # a candidate may run over up to SPAN - 1 intermediate boundaries
MIN_LEN = 5            # frames
MAX_DEAD = 0.5         # mean P(D) above this: not a possession


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def peak_candidates(p_switch, emission, map_segs, graded, height=MIN_HEIGHT, span=SPAN):
    e = np.asarray(emission, float)
    p_dead = np.exp(np.clip(e[:, 2], -50, 0))
    alive = np.logaddexp(e[:, 0], e[:, 1])
    p_h = np.exp(e[:, 0] - alive)
    cs_d = np.concatenate(([0.0], np.cumsum(p_dead)))
    cs_h = np.concatenate(([0.0], np.cumsum(p_h * (1 - p_dead))))
    cs_a = np.concatenate(([0.0], np.cumsum(1 - p_dead)))
    peaks, _ = find_peaks(np.nan_to_num(p_switch), height=height, distance=MIN_DIST)
    map_b = np.unique(np.r_[map_segs[:, 0], map_segs[:, 1]])
    out = []
    for start, stop in _runs(np.asarray(graded, bool)):
        b = np.unique(np.r_[start, stop, peaks[(peaks > start) & (peaks < stop)],
                            map_b[(map_b > start) & (map_b < stop)]])
        for k in range(1, span + 1):
            s, t = b[:-k], b[k:]
            ok = t - s >= MIN_LEN
            s, t = s[ok], t[ok]
            dead = (cs_d[t] - cs_d[s]) / (t - s)
            ph = (cs_h[t] - cs_h[s]) / np.maximum(cs_a[t] - cs_a[s], 1e-9)
            keep = dead < MAX_DEAD
            code = np.where(ph >= 0.5, pm.H, pm.A)
            out.append(np.column_stack([s[keep], t[keep], code[keep]]))
    out = np.concatenate(out) if out else np.zeros((0, 3), int)
    return out.astype(np.int64)
