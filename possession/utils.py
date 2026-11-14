"""Small helpers shared by the pipeline."""

import os

import numpy as np
import pandas as pd

import possession  # noqa: F401  (sets the paths to the organisers' code)
import possession_metrics as pm
from games import GAME, GAMES

# The organisers' FAQ: the 2nd half is graded from frame 110808 but kicks off at 113742,
# so the frames in between are always a dead ball.
SECOND_HALF_KICKOFF = 113742
LETTERS = np.array(list(pm.LABELS))
FPS = 25


def runs_of(mask):
    """[(start, stop)] half-open runs where a boolean mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def known_dead(m):
    """(n,) bool: graded frames before the 2nd-half kick-off, which are always D."""
    mask = np.zeros(m.n_frames, dtype=bool)
    if m.n_frames == GAMES[GAME]["n_video_frames"]:
        mask[int(np.flatnonzero(m.half == 2)[0]):SECOND_HALF_KICKOFF] = True
    return mask


def write_submission(path, codes):
    """Write the frame,label CSV. Raises if the organisers' checks would reject it."""
    codes = np.asarray(codes).astype(np.int8)
    table = pd.DataFrame({"frame": np.arange(len(codes)), "label": LETTERS[codes]})
    reason = pm.validate_submission(table, len(codes))
    if reason:
        raise ValueError("submission would be rejected: " + reason)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    table.to_csv(path, index=False)
    return path
