"""Per-frame H / A / D labels for the external matches, derived from their events by rule.

Possession: a team has the ball from the start of its PASS, CARRY, RECOVERY, SET PIECE or SHOT
until the next such event. BALL LOST and CHALLENGE don't move possession on their own (the
opponent's RECOVERY does).
Dead ball: from the ball crossing a line (BALL OUT end frame, or the end of a shot that goes out),
a foul (FAULT RECEIVED) or a goal, until the next possession event after it (the restart).
Frames before the first event and after the last event of each period are pm.X.

No hand labelling: the events are Metrica's, the rules are below. check_labels() measures them
against the match's own tracking (close-control agreement, player speed while dead), so a change
to the rules is always measured the same way.

    from possession.ext_labels import event_labels, check_labels
    codes = event_labels(m, events)
"""

import warnings

import numpy as np
import pandas as pd

from possession.data import HOME, AWAY
from possession.utils import runs_of, FPS
import possession_metrics as pm  # noqa: E402

TEAM_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}


def _dead_starts(events):
    """Frames where the ball goes dead."""
    out = events[events.type == "BALL OUT"]
    shots_out = events[(events.type == "SHOT") & events.subtype.str.contains("OUT")]
    goals = events[(events.type == "SHOT") & events.subtype.str.endswith("GOAL")]
    fouls = events[events.type == "FAULT RECEIVED"]
    frames = np.concatenate([
        np.maximum(out.start_frame, out.end_frame),
        np.maximum(shots_out.start_frame, shots_out.end_frame),
        np.maximum(goals.start_frame, goals.end_frame),
        fouls.start_frame,
    ])
    return np.unique(frames.astype(int))


def event_labels(m, events):
    """(n,) int8 label codes from the events."""
    labels = np.full(m.n_frames, pm.X, dtype=np.int8)
    team = events[events.type.isin(TEAM_EVENTS)]
    marks = [(int(f), pm.H if t == "H" else pm.A) for f, t in zip(team.start_frame, team.team)]
    marks += [(int(f), pm.D) for f in _dead_starts(events)]
    # at the same frame a dead mark wins only if it is the later fact; sort dead after team marks
    marks.sort(key=lambda fm: (fm[0], fm[1] == pm.D))

    for period in np.unique(events.period):
        in_period = np.flatnonzero(m.half == period)
        lo, hi = in_period[0], in_period[-1] + 1
        period_marks = [(f, c) for f, c in marks if lo <= f < hi]
        if not period_marks:
            continue
        last_event = int(events[events.period == period][["start_frame", "end_frame"]].max().max())
        stop = min(hi, last_event + 1)
        for (f, c), (f_next, _) in zip(period_marks, period_marks[1:] + [(stop, None)]):
            labels[f:max(f, f_next)] = c
    return labels


def check_labels(m, labels, radius=1.5, min_frames=20):
    """Consistency of the labels with the match's own tracking (no ground truth exists here)."""
    seen = ~np.isnan(m.ball[:, 0])
    dist = np.linalg.norm(m.players - m.ball[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no player of a team
        d_h = np.nanmin(np.where(m.team[None, :] == HOME, dist, np.nan), axis=1)
        d_a = np.nanmin(np.where(m.team[None, :] == AWAY, dist, np.nan), axis=1)
    control = np.full(m.n_frames, pm.X, dtype=np.int8)
    for code, mine, theirs in ((pm.H, d_h, d_a), (pm.A, d_a, d_h)):
        c = seen & (mine <= radius) & (theirs >= mine + 1.0)
        for s, e in runs_of(c):
            if e - s >= min_frames:
                control[s:e] = code
    sel = (control != pm.X) & (labels != pm.X)
    alive = sel & np.isin(labels, (pm.H, pm.A))

    step = np.full(m.players.shape[:2], np.nan)
    step[1:] = np.linalg.norm(np.diff(m.players, axis=0), axis=2) * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = pd.Series(np.nanmean(np.where(step < 12, step, np.nan), axis=1)).rolling(
            50, center=True, min_periods=1).mean().values

    spells = pm.possession_spells(labels, pm.valid_runs(labels))
    stops = [s for s in pm.segments(labels, pm.valid_runs(labels)) if s[2] == pm.D]
    minutes = (labels != pm.X).sum() / FPS / 60
    return {
        "labelled_share": float((labels != pm.X).mean()),
        "control_agree": float((control[sel] == labels[sel]).mean()),
        "control_on_dead": float((labels[sel] == pm.D).mean()),
        "wrong_team_alive": float((control[alive] != labels[alive]).mean()),
        "speed_alive": float(np.nanmedian(speed[np.isin(labels, (pm.H, pm.A))])),
        "speed_dead": float(np.nanmedian(speed[labels == pm.D])),
        "ball_missing_alive": float((~seen[np.isin(labels, (pm.H, pm.A))]).mean()),
        "ball_missing_dead": float((~seen[labels == pm.D]).mean()),
        "possessions": len(spells),
        "per_min": len(spells) / minutes,
        "stoppages": len(stops),
        "dead_share": float((labels[labels != pm.X] == pm.D).mean()),
        "home_share": float((labels[np.isin(labels, (pm.H, pm.A))] == pm.H).mean()),
    }


if __name__ == "__main__":
    from possession.external import load_external, EXTERNAL_GAMES
    rows = {}
    for game in EXTERNAL_GAMES:
        m, events = load_external(game)
        rows[game] = check_labels(m, event_labels(m, events))
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).T.to_string())
