"""Make an external match look like BAR-ATH's automatic tracking, so a model trained on it transfers.

The external tracking is clean (11 players per team on every frame, fixed identities, smooth) and
its ball is blanked while play is dead (92-100% missing) but rarely in play (7-13%). BAR-ATH's ATD
is the opposite on both counts. Trained as is, a model would learn "ball missing = dead" and rely
on perfect player tracks. degrade() removes both shortcuts, using only *statistics* of BAR-ATH's
raw tracking (gap lengths, track-slot lifetimes, players per frame), never its labels:

  ball, dead     put a ball back during each stoppage: seen going out past the line where the event
                 says it crossed (BAR-ATH's seen-outside durations and distances), then moving to
                 the restart spot and waiting there; then hide a random share of it
  ball, in play  bridge gaps < 3 s (the ATD does), then cut gaps with BAR-ATH's real gap lengths
                 until a random target share of in-play frames has no ball
  ball, smoothing the ATD ball is smoothed (99th pct of a 3-frame velocity change 4 m/s vs 21 in the
                 external tracking); a Gaussian of ~6.5 frames matches its tail, so each copy
                 gets one drawn from BALL_SMOOTH_FRAMES. Without it touches look sharper than
                 they ever do on BAR-ATH
  players        position noise; tracks cut into slots with BAR-ATH's slot lifetimes (identity
                 breaks); some tracklets dropped, some duplicated next to the real player

Shares are drawn per copy from ranges (domain randomisation), so the model sees a spread of
tracking quality rather than one guess at BAR-ATH's.

    from possession.degrade import atd_stats, degrade, swap_teams
    stats = atd_stats(load_match())                 # BAR-ATH statistics, no labels
    m2 = degrade(m, labels, stats, np.random.default_rng(0))
"""

import dataclasses

import numpy as np
import pandas as pd

from possession.data import HOME, AWAY
from possession.utils import runs_of, FPS
from possession.external import speed_from_positions
import possession_metrics as pm  # noqa: E402

# Params
ALIVE_MISSING = (0.25, 0.45)   # share of in-play frames without a ball (BAR-ATH: ~41% overall)
DEAD_MISSING = (0.0, 0.5)      # hidden share of each refilled stoppage; stoppages with no ball at
                               # either edge stay empty, so the total lands higher
BRIDGE_S = 3.0                 # the ATD bridges shorter gaps: every BAR-ATH gap is ≥ 3 s
RESTART_WAIT = 0.4             # last share of a stoppage the ball waits at the restart spot
HANDLE_SPEED = (0.3, 3.0)      # m/s of a dead ball being handled before it rests (per stoppage)
BALL_SMOOTH_FRAMES = (5.5, 7.5)  # Gaussian sigma; BAR-ATH's velocity-change tail fits ~6.5 (10-23)
EDGE_SEARCH_S = 10.0           # how far to look for the ball before / after a stoppage
POSITION_NOISE_M = 0.3
NOISE_KNOT_S = 2.0             # s between noise knots (adds ~0.2 m/s of speed, not ~2)
DROP_P = 0.065                 # tracklets dropped  } tuned on players per frame: BAR-ATH 10.2,
DUP_P = 0.12                   # tracklets duplicated } <10 on 13-15%, >11 on 5-8%; here 10.5 / 12.8% / 5.9%
DUP_OFFSET_M = (0.5, 1.5)
DUP_LEN_S = (2.0, 20.0)


def atd_stats(m):
    """Statistics of BAR-ATH's raw tracking that degrade() copies. Uses no labels."""
    g = m.graded
    gaps = np.array([e - s for s, e in runs_of(~m.ball_seen & g) if e - s >= BRIDGE_S * FPS])
    slot_runs = np.array([e - s for k in range(m.players.shape[1])
                          for s, e in runs_of(m.player_seen[:, k] & g)])
    length, width = m.pitch
    x, y = m.ball[:, 0], m.ball[:, 1]
    outside = np.maximum.reduce([-x, x - length, -y, y - width])
    seen_out = (outside > 0.5) & m.ball_seen & g
    return {"gap_frames": gaps, "slot_run_frames": slot_runs,
            "out_run_frames": np.array([e - s for s, e in runs_of(seen_out)]),
            "out_dist_m": outside[seen_out]}


# -- ball --------------------------------------------------------------------------------------

def _bridge_short_gaps(ball, labels):
    """Linear interpolation over in-play gaps shorter than BRIDGE_S."""
    out = ball.copy()
    missing = np.isnan(ball[:, 0])
    for s, e in runs_of(missing):
        if e - s < BRIDGE_S * FPS and s > 0 and e < len(ball) and not (labels[s:e] == pm.D).any():
            w = np.linspace(0, 1, e - s + 2)[1:-1, None]
            out[s:e] = ball[s - 1] * (1 - w) + ball[e] * w
    return out


def _edge_position(ball, start, step, limit):
    """First seen ball position walking from `start` in direction `step`, within `limit` frames."""
    i = start
    for _ in range(limit):
        if i < 0 or i >= len(ball):
            return None
        if not np.isnan(ball[i, 0]):
            return ball[i]
        i += step
    return None


def out_points(events):
    """{frame the ball goes dead: (x, y) where it crossed the line} for balls out of play."""
    out = events[(events.type == "BALL OUT")
                 | ((events.type == "SHOT") & events.subtype.str.contains("OUT"))]
    points = {}
    for row in out.itertuples():
        x, y = (row.end_x, row.end_y) if not np.isnan(row.end_x) else (row.start_x, row.start_y)
        if not np.isnan(x):
            points[int(max(row.start_frame, row.end_frame))] = np.array([x, y])
    return points


def _ball_going_out(point, pitch, stats, n_frames, rng):
    """(k, 2) a ball seen leaving the pitch past `point`, with BAR-ATH's seen-outside statistics."""
    length, width = pitch
    x, y = point
    # outward direction: the nearest line
    gaps = {(-1, 0): x, (1, 0): length - x, (0, -1): y, (0, 1): width - y}
    direction = np.array(min(gaps, key=gaps.get), dtype=float)
    edge = np.array([np.clip(x, 0, length), np.clip(y, 0, width)])
    if direction[0]:
        edge[0] = 0 if direction[0] < 0 else length
    else:
        edge[1] = 0 if direction[1] < 0 else width
    k = int(min(rng.choice(stats["out_run_frames"]), n_frames // 2))
    dist = float(rng.choice(stats["out_dist_m"]))
    u = np.clip(np.arange(k) / max(1, k // 2), 0, 1)[:, None]   # moves out, then rests
    return edge + direction * dist * u


def _refill_stoppages(ball, labels, stats, rng, pitch, points):
    """A ball during each stoppage: seen going out (if it did), moving to the restart spot, waiting."""
    out = ball.copy()
    limit = int(EDGE_SEARCH_S * FPS)
    for s, e, code in pm.segments(labels, pm.valid_runs(labels)):
        if code != pm.D:
            continue
        start = s
        before = _edge_position(ball, s - 1, -1, limit)
        if s in points:
            going = _ball_going_out(points[s], pitch, stats, e - s, rng)
            out[s:s + len(going)] = going
            start = s + len(going)
            before = going[-1] if len(going) else points[s]
        after = _edge_position(ball, e, 1, limit)
        if before is None and after is None:
            continue
        before = after if before is None else before
        after = before if after is None else after
        u = np.clip(np.linspace(0, 1, e - start) / (1 - RESTART_WAIT), 0, 1)[:, None]
        # smooth motion, not per-frame jitter: 0.2 m iid noise reads as ~9 m/s of ball speed, which
        # taught the dead head "fast ball = dead"; a still ATD ball moves ~0.25 m/s (lag-1 r 0.99).
        # Until the restart wait the ball is handled (rolled, carried, placed), then it rests
        out[start:e] = (before * (1 - u) + after * u + _noise(e - start, rng)
                        + _handled(e - start, rng) * (1 - u))
        # hide a random share as one block, after the ball was seen going out
        hidden = int(rng.uniform(*DEAD_MISSING) * (e - start))
        if hidden:
            h0 = int(rng.integers(start, e - hidden + 1))
            out[h0:h0 + hidden] = np.nan
    return out


def _handled(n, rng):
    """(n, 2) a dead ball being handled: a smooth random walk at a per-stoppage speed in
    HANDLE_SPEED (walking pace while a player places or carries the ball; deliberately wide),
    one direction change a second."""
    knot = FPS
    n_knots = n // knot + 2
    speed = rng.uniform(*HANDLE_SPEED)
    angle = rng.uniform(0, 2 * np.pi, n_knots)
    steps = speed * np.column_stack([np.cos(angle), np.sin(angle)])
    steps[0] = 0
    values = np.cumsum(steps, axis=0)
    t = np.arange(n) / knot
    return np.column_stack([np.interp(t, np.arange(n_knots), values[:, i]) for i in range(2)])


def _smooth_ball(ball, rng):
    """Gaussian smoothing within each seen run, like the ATD's ball."""
    from scipy.ndimage import gaussian_filter1d
    sigma = rng.uniform(*BALL_SMOOTH_FRAMES)
    out = ball.copy()
    for s, e in runs_of(~np.isnan(ball[:, 0])):
        if e - s > 3:
            out[s:e] = gaussian_filter1d(ball[s:e], sigma, axis=0, mode="nearest")
    return out


def _cut_play_gaps(ball, labels, stats, rng):
    """Hide the ball in gaps of BAR-ATH's lengths until the in-play target share is reached."""
    out = ball.copy()
    alive = np.isin(labels, (pm.H, pm.A))
    target = rng.uniform(*ALIVE_MISSING)
    candidates = np.flatnonzero(labels != pm.X)
    for _ in range(10000):
        if np.isnan(out[alive, 0]).mean() >= target:
            break
        length = int(rng.choice(stats["gap_frames"]))
        start = int(rng.choice(candidates))
        out[start:start + length] = np.nan
    return out


# -- players -----------------------------------------------------------------------------------

def _noise(n, rng):
    """(n, 2) slow drift with about POSITION_NOISE_M standard deviation: random offsets every
    NOISE_KNOT_S seconds, linearly interpolated. Frame-to-frame jitter would add fake speed (a
    0.3 m jitter smoothed over 1 s adds ~2 m/s), and speed is the main dead-ball signal."""
    knot = int(NOISE_KNOT_S * FPS)
    n_knots = n // knot + 2
    values = rng.normal(0, POSITION_NOISE_M, (n_knots, 2))
    t = np.arange(n) / knot
    return np.column_stack([np.interp(t, np.arange(n_knots), values[:, i]) for i in range(2)])


def _tracklets(players, team, stats, rng):
    """[(team, start, positions)] after cutting, dropping and duplicating, from clean tracks."""
    out = []
    for k in range(players.shape[1]):
        for s, e in runs_of(~np.isnan(players[:, k, 0])):
            t = s
            while t < e:
                length = int(rng.choice(stats["slot_run_frames"]))
                stop = min(e, t + max(length, FPS))
                if rng.random() >= DROP_P:
                    pos = players[t:stop, k] + _noise(stop - t, rng)
                    out.append((team[k], t, pos))
                    if rng.random() < DUP_P:
                        d_len = min(stop - t, int(rng.uniform(*DUP_LEN_S) * FPS))
                        d0 = int(rng.integers(t, stop - d_len + 1))
                        angle = rng.uniform(0, 2 * np.pi)
                        offset = rng.uniform(*DUP_OFFSET_M) * np.array([np.cos(angle), np.sin(angle)])
                        out.append((team[k], d0, players[d0:d0 + d_len, k] + offset + _noise(d_len, rng)))
                t = stop
    return out


def _assign_slots(tracklets, n_frames):
    """Pack tracklets into slots per team (a slot is reused once it is free), like the ATD's."""
    slots = {HOME: [], AWAY: []}   # per team: list of (free_from_frame, [(start, pos)])
    for team, start, pos in sorted(tracklets, key=lambda t: t[1]):
        for slot in slots[team]:
            if slot[0] <= start:
                slot[0] = start + len(pos) + 1   # one empty frame, so speeds never bridge identities
                slot[1].append((start, pos))
                break
        else:
            slots[team].append([start + len(pos) + 1, [(start, pos)]])
    columns, team_of = [], []
    for team in (HOME, AWAY):
        for _, pieces in slots[team]:
            col = np.full((n_frames, 2), np.nan, dtype=np.float32)
            for start, pos in pieces:
                col[start:start + len(pos)] = pos
            columns.append(col)
            team_of.append(team)
    return np.stack(columns, axis=1), np.array(team_of, dtype=np.int8)


# -- top level ---------------------------------------------------------------------------------

def degrade(m, labels, stats, rng, events=None, smooth=False, touch_smooth=False):
    """A degraded copy of an external Match; labels are unchanged (same frames).
    smooth: ATD-like smoothing of the ball every feature sees (off reproduces v2 exactly).
    touch_smooth: smooth only the copy touch detection sees (m.touch_ball), drawn last so the
    rest of the copy is identical to touch_smooth=False."""
    points = out_points(events) if events is not None else {}
    ball = _bridge_short_gaps(m.ball.astype(float), labels)
    ball = _refill_stoppages(ball, labels, stats, rng, m.pitch, points)
    if smooth:
        ball = _smooth_ball(ball, rng)
    ball = _cut_play_gaps(ball, labels, stats, rng)
    players, team = _assign_slots(_tracklets(m.players.astype(float), m.team, stats, rng), m.n_frames)
    touch_ball = _smooth_ball(ball, rng).astype(np.float32) if touch_smooth else None
    return dataclasses.replace(m, ball=ball.astype(np.float32), players=players, team=team,
                               player_speed=speed_from_positions(players), touch_ball=touch_ball)


def swap_teams(m, labels):
    """The same match with home and away exchanged (removes home/away identity from training)."""
    swapped = labels.copy()
    swapped[labels == pm.H] = pm.A
    swapped[labels == pm.A] = pm.H
    return dataclasses.replace(m, team=(1 - m.team).astype(np.int8)), swapped
