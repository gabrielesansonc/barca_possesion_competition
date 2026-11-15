"""Train every model in models/ from open data of other matches.

    python train.py              # writes models/ (about an hour on a laptop with 8 GB)

Training data: 10 matches with per-frame possession labels (see training/README.md).
  Metrica sample data   3 matches, labels from their event data (ext_labels.py)
  DFL open data         7 Bundesliga matches, labels from the tracking flags BallStatus / BallPossession
Their tracking is cleaner than this match's automatic tracking, so every training match is
first made to look like it ("degraded", degrade.py): the ball goes missing in gaps of the same
lengths, players' tracks break into pieces, positions get noise. degrade.py uses summary
statistics of this match's tracking (gap lengths, track lifetimes), never its labels.
Each degraded copy is also used with home and away swapped, so no model can learn which side
is "home".

What is trained, in order:
  1. Head tables      degraded copies of each match -> frame features -> every 5th (Metrica) or
                      10th (DFL) labelled frame.
  2. Heads            dead head and team head (LightGBM) on all 10 matches        -> dead_head.txt,
                                                                                     team_head.txt
  3. Switch head      the same idea, target "possession changes within 0.5 s"     -> switch_head.txt
  4. Held-out copies  for each match, heads and switch head trained on the other 9 decode fresh
                      copies of it (3 seeds x 2 copies). These are realistic decodes with
                      realistic mistakes, which the next two models learn from.
  5. Rescorer         per candidate possession: how well it matches a real one   -> rescorer.txt
  6. Boundary model   per change of possession: how far it is from the real one   -> boundary.txt
                      (candidates chosen by rescorers trained without that match)
  7. Prior            how long possessions and stoppages last (Metrica labels)    -> prior.json
  8. Whistle count    whistled events per DFL match                               -> whistle.py

Everything is deterministic (fixed seeds, LightGBM deterministic mode): running this again gives
the same models, byte for byte.
"""

import dataclasses
import json
import os
import pickle
import time
import types

import numpy as np
import pandas as pd

from possession import DATA_DIR, MODELS_DIR
from possession import decoder as dm
from possession import pipeline as pl
from possession import turnover as to
from possession.degrade import (_assign_slots, _bridge_short_gaps, _cut_play_gaps, _refill_stoppages,
                                _tracklets, atd_stats, degrade, swap_teams)
from possession.ext_labels import event_labels
from possession.external import EXTERNAL_GAMES, load_external, speed_from_positions
from possession.extdata import DFL_MATCHES, load_dfl
from possession.hsmm import fit_prior
import possession_metrics as pm

CACHE = os.path.join(DATA_DIR, "training")
MATCHES = [("metrica", g) for g in EXTERNAL_GAMES] + [("dfl", g) for g in DFL_MATCHES]

# Degraded copies
HEAD_COPIES = {"metrica": 3, "dfl": 2}      # copies per match for the head tables (each also swapped)
HEAD_STEP = {"metrica": 5, "dfl": 10}       # keep every 5th / 10th labelled frame
SWITCH_COPIES = {"metrica": 3, "dfl": 2}    # copies per match for the switch tables (not swapped)
SWITCH_STEP = {"metrica": 5, "dfl": 10}
DFL_SEED_BASE = 100                         # DFL matches use seeds 100 + their index
OUT_LINE_M = 1.0                            # DFL's frozen dead ball this close to a line = it went out there
HELD_SEEDS, HELD_COPIES = (777, 888, 999), 2
DEV_SEEDS = (888, 999)                      # fold rescorers learn from these copies only

# Head columns left out: their distribution on this match is unlike any training match
# (a check of the inputs only, no labels involved)
SHIFTED = {"ix_ball_acc", "ix_relspeed_h", "ix_relspeed_a", "ix_relspeed_near", "ix_trend_cx_gap",
           "ix_trend_h_line", "ix_mark_h_trend", "ix_mark_a_trend"}

# LightGBM
HEAD_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "num_threads": 3, "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1}
HEAD_ROUNDS = 1500       # dead / team heads (held-out log loss bottoms out near here on the DFL matches)
SWITCH_ROUNDS = 300
RESCORER_PARAMS = {"objective": "cross_entropy", "learning_rate": 0.03, "num_leaves": 31,
                   "min_data_in_leaf": 300, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 3, "seed": 0,
                   "deterministic": True, "force_row_wise": True, "verbose": -1}
RESCORER_ROUNDS = 700
BOUNDARY_PARAMS = {"objective": "huber", "alpha": 10.0, "learning_rate": 0.05, "num_leaves": 15,
                   "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 3, "seed": 0,
                   "deterministic": True, "force_row_wise": True, "verbose": -1}
BOUNDARY_ROUNDS = 300
BOUNDARY_MATCH = 50      # frames: a real change of the same kind this close is "the" change
WHISTLED = ("Foul", "Offside", "Penalty", "RefereeBall", "KickOff", "FinalWhistle")


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _cached(name, build):
    """build() once, then read it back from data/training/r<radius>/<name>.pkl (the features
    depend on the control radius, so each radius has its own cache)."""
    from possession.features import CONTROL_RADIUS
    path = os.path.join(CACHE, "r{}".format(CONTROL_RADIUS), name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    out = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


# -- training matches -----------------------------------------------------------------------------

def match_labels(key):
    """(n,) per-frame labels of a training match (pm codes)."""
    src, g = key
    if src == "metrica":
        return event_labels(*load_external(g))
    return load_dfl(g)[1]


def _dfl_out_points(m, labels):
    """{dead-ball start frame: (x, y)} where DFL's frozen dead ball sits on or past a line."""
    length, width = m.pitch
    points = {}
    for s, e, c in pm.segments(labels, pm.valid_runs(labels)):
        if c != pm.D:
            continue
        x, y = m.ball[min(e - 1, s + 1)]
        if min(x, length - x, y, width - y) <= OUT_LINE_M:
            points[s] = np.array([x, y], dtype=float)
    return points


def _degrade_dfl(m, labels, stats, rng):
    """degrade() for a DFL match, whose ball is frozen while dead: that ball is dropped first and
    refilled like Metrica's (seen going out where it froze on a line, handled, waiting)."""
    ball = m.ball.astype(float).copy()
    ball[labels == pm.D] = np.nan
    ball = _bridge_short_gaps(ball, labels)
    ball = _refill_stoppages(ball, labels, stats, rng, m.pitch, _dfl_out_points(m, labels))
    ball = _cut_play_gaps(ball, labels, stats, rng)
    players, team = _assign_slots(_tracklets(m.players.astype(float), m.team, stats, rng), m.n_frames)
    return dataclasses.replace(m, ball=ball.astype(np.float32), players=players, team=team,
                               player_speed=speed_from_positions(players))


def copies(key, stats, n, seed, swapped=True):
    """Degraded copies of one training match: (match, labels), each followed by its home/away
    swap when swapped=True."""
    src, g = key
    if src == "metrica":
        m, events = load_external(g)
        labels = event_labels(m, events)
        game = g
    else:
        m, labels, _ = load_dfl(g)
        events = None
        game = DFL_SEED_BASE + DFL_MATCHES.index(g)
    for copy in range(n):
        rng = np.random.default_rng(seed + 1000 * game + copy)
        d = degrade(m, labels, stats, rng, events) if src == "metrica" else _degrade_dfl(m, labels, stats, rng)
        yield d, labels
        if swapped:
            yield swap_teams(d, labels)


# -- 1-2. heads -----------------------------------------------------------------------------------

def head_table(key, stats):
    """(X DataFrame, labels): the heads' training rows of one match."""
    src = key[0]
    step = HEAD_STEP[src]
    parts, labs, cols = [], [], None
    for i, (m, lab) in enumerate(copies(key, stats, HEAD_COPIES[src], 0)):
        X = pl.frame_features(m)
        cols = list(X.columns)
        rows = np.flatnonzero(lab != pm.X)[i % step::step]
        parts.append(X.values[rows])
        labs.append(lab[rows])
    X = pd.DataFrame(np.concatenate(parts), columns=cols, copy=False)
    return X[[c for c in cols if c not in SHIFTED]], np.concatenate(labs)


def fit_heads(tables):
    """{"dead", "team"} LightGBM boosters on the concatenated tables."""
    import lightgbm as lgb
    X = pd.concat([t[0] for t in tables], ignore_index=True)
    y = np.concatenate([t[1] for t in tables])
    dead = lgb.train(HEAD_PARAMS, lgb.Dataset(X, label=(y == pm.D).astype(int)), HEAD_ROUNDS)
    alive = y != pm.D
    team = lgb.train(HEAD_PARAMS, lgb.Dataset(X[alive], label=(y[alive] == pm.H).astype(int)), HEAD_ROUNDS)
    return {"dead": dead, "team": team}


# -- 3. switch head -------------------------------------------------------------------------------

def switch_table(key, stats):
    """DataFrame of the switch head's training rows of one match, target in column _y."""
    src = key[0]
    parts = []
    for i, (m, lab) in enumerate(copies(key, stats, SWITCH_COPIES[src], 0, swapped=False)):
        y = to.switch_target(lab)
        rows = np.flatnonzero(y >= 0)[i % SWITCH_STEP[src]::SWITCH_STEP[src]]
        T = pl.switch_features(m).iloc[rows].reset_index(drop=True)
        T["_y"] = y[rows]
        parts.append(T)
    return pd.concat(parts, ignore_index=True)


def fit_switch(tables):
    """(booster, columns) of the switch head."""
    import lightgbm as lgb
    cols = [c for c in tables[0].columns if c != "_y"]
    T = pd.concat(tables, ignore_index=True)
    return lgb.train(HEAD_PARAMS, lgb.Dataset(T[cols], label=T["_y"]), SWITCH_ROUNDS), cols


# -- 4. held-out copies ---------------------------------------------------------------------------

def heldout_copies(stats, head_tabs, switch_tabs):
    """[{key, source, match, seed, copy, emission, p_switch, labels, half}] for every training match:
    3 seeds x 2 fresh copies, decoded by heads and a switch head trained on the other 9 matches."""
    rows = []
    for held in MATCHES:
        _log("held-out", held)
        others = [k for k in MATCHES if k != held]
        heads = fit_heads([head_tabs[k] for k in others])
        switch, cols = fit_switch([switch_tabs[k] for k in others])
        for seed in HELD_SEEDS:
            gen = copies(held, stats, HELD_COPIES, seed, swapped=False)
            for c, (m, lab) in enumerate(gen):
                rows.append({"key": (held[0], str(held[1]), seed, c), "source": held[0],
                             "match": str(held[1]), "seed": seed, "copy": c,
                             "emission": pl.head_emission(heads, pl.frame_features(m)).astype(np.float32),
                             "p_switch": switch.predict(pl.switch_features(m)[cols]).astype(np.float32),
                             "labels": lab.astype(np.int8), "half": m.half.astype(np.int8)})
    return rows


def decoding_priors():
    """Prior per training match for decoding its held-out copies: a Metrica match uses the other
    two Metrica matches' labels, a DFL match all three. Returns (priors by match, all-Metrica prior)."""
    labs = {g: match_labels(("metrica", g)) for g in EXTERNAL_GAMES}
    priors = {str(g): fit_prior([labs[h] for h in EXTERNAL_GAMES if h != g]) for g in EXTERNAL_GAMES}
    return priors, fit_prior([labs[g] for g in EXTERNAL_GAMES])


# -- 5. rescorer ----------------------------------------------------------------------------------

def candidate_rows(heldout):
    """Per held-out copy: its candidates, their features and target y (mean over the IoU thresholds
    0.3-0.7 of "matches a real possession at this IoU")."""
    priors, prior_all = decoding_priors()
    out = []
    for r in heldout:
        mm = types.SimpleNamespace(half=r["half"], graded=r["half"] > 0, n_frames=len(r["half"]))
        prior = priors[r["match"]] if r["source"] == "metrica" else prior_all
        e = r["emission"].astype(float)
        cands, X, map_segs = pl.candidate_table(e, mm, prior, r["p_switch"])
        truth = dm.team_spells(dm.code_segments(r["labels"], pm.valid_runs(r["labels"])))
        y = (dm.best_ious(cands, truth)[:, None] >= dm.IOUS[None, :]).mean(axis=1)
        out.append({"key": r["key"], "source": r["source"], "match": r["match"], "seed": r["seed"],
                    "cands": cands, "X": X, "map": map_segs, "mm": mm, "y": y})
    return out


def fit_rescorer(rows, cols):
    import lightgbm as lgb
    X = np.concatenate([d["X"][cols].values for d in rows])
    y = np.concatenate([d["y"] for d in rows])
    return lgb.train(RESCORER_PARAMS, lgb.Dataset(X, label=y, feature_name=cols), RESCORER_ROUNDS)


def fold_rescorers(cands, cols):
    """{(source, match): rescorer trained on the dev copies of the other 9 matches}."""
    folds = sorted(set((d["source"], d["match"]) for d in cands))
    return {f: fit_rescorer([d for d in cands if (d["source"], d["match"]) != f and d["seed"] in DEV_SEEDS], cols)
            for f in folds}


# -- 6. boundary model ----------------------------------------------------------------------------

def _match_offsets(pb, tb):
    """Offset (real - predicted) to the nearest real change of the same kind within BOUNDARY_MATCH
    frames, NaN if none."""
    off = np.full(len(pb), np.nan)
    for kind in set(map(tuple, pb[:, 1:].tolist())):
        sel = (pb[:, 1] == kind[0]) & (pb[:, 2] == kind[1])
        tf = np.sort(tb[(tb[:, 1] == kind[0]) & (tb[:, 2] == kind[1]), 0])
        if not len(tf):
            continue
        f = pb[sel, 0]
        j = np.clip(np.searchsorted(tf, f), 1, len(tf) - 1) if len(tf) > 1 else np.zeros(len(f), int)
        cand = np.stack([tf[np.maximum(j - 1, 0)], tf[j]], axis=1)
        best = cand[np.arange(len(f)), np.argmin(np.abs(cand - f[:, None]), axis=1)]
        d = best - f
        off[np.flatnonzero(sel)] = np.where(np.abs(d) <= BOUNDARY_MATCH, d, np.nan)
    return off


def boundary_table(heldout, cands, rescorers, cols):
    """Rows of the boundary model: every change of possession in the held-out decodes (each copy
    decoded with the rescorer trained without its match), with its offset to the real one."""
    by_key = {d["key"]: d for d in cands}
    parts = []
    for r in heldout:
        d = by_key[r["key"]]
        q = rescorers[(d["source"], d["match"])].predict(d["X"][cols].values)
        codes = pl.choose(d["cands"], q, d["map"], d["mm"]).astype(np.int8)
        lab = r["labels"]
        pb = pl._boundaries(codes, d["mm"].graded)
        tb = pl._boundaries(lab, lab != pm.X)
        X = pl.boundary_features(codes, pb, r["emission"], r["p_switch"], len(codes))
        X["_off"] = _match_offsets(pb, tb)
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


def fit_boundary(table):
    import lightgbm as lgb
    cols = [c for c in table.columns if not c.startswith("_")]
    y = np.clip(table["_off"].fillna(0).to_numpy(), -pl.BOUNDARY_CLIP, pl.BOUNDARY_CLIP)
    return lgb.train(BOUNDARY_PARAMS, lgb.Dataset(table[cols], label=y), BOUNDARY_ROUNDS)


# -- 7-8. prior and whistle count -----------------------------------------------------------------

def _jsonable(v):
    if isinstance(v, np.ndarray):
        return {"__array__": v.tolist(), "dtype": str(v.dtype)}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return {"__tuple__": [_jsonable(x) for x in v]}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def whistles_per_match():
    """Mean count of whistled events per DFL match (the value of whistle.WHISTLES_PER_MATCH)."""
    counts = []
    for g in DFL_MATCHES:
        load_dfl(g)                                    # downloads and converts the match if needed
        ev = pd.read_csv(os.path.join(DATA_DIR, "external", "dfl", g + "_events.csv"))
        counts.append(int(ev.type.isin(WHISTLED).sum()))
    return int(round(np.mean(counts))), counts


# -- everything -----------------------------------------------------------------------------------

def train_all(out=MODELS_DIR):
    """Steps 1-8; writes the model files into `out`."""
    from possession.data import load_match
    os.makedirs(out, exist_ok=True)
    stats = atd_stats(load_match())                  # tracking statistics of this match, no labels
    _log("1. head tables")
    head_tabs = {k: _cached("head_table_{}_{}".format(*k), lambda k=k: head_table(k, stats)) for k in MATCHES}
    _log("2. heads")
    heads = fit_heads([head_tabs[k] for k in MATCHES])
    heads["dead"].save_model(os.path.join(out, "dead_head.txt"))
    heads["team"].save_model(os.path.join(out, "team_head.txt"))
    _log("3. switch head")
    switch_tabs = {k: _cached("switch_table_{}_{}".format(*k), lambda k=k: switch_table(k, stats)) for k in MATCHES}
    switch, switch_cols = fit_switch([switch_tabs[k] for k in MATCHES])
    switch.save_model(os.path.join(out, "switch_head.txt"))
    _log("4. held-out copies")
    heldout = _cached("heldout_h{}_s{}".format(HEAD_ROUNDS, SWITCH_ROUNDS),   # depends on the round counts
                      lambda: heldout_copies(stats, head_tabs, switch_tabs))
    del head_tabs, switch_tabs
    _log("5. rescorer")
    cands = candidate_rows(heldout)
    cols = list(cands[0]["X"].columns)
    fit_rescorer(cands, cols).save_model(os.path.join(out, "rescorer.txt"))
    _log("6. boundary model")
    rescorers = fold_rescorers(cands, cols)
    fit_boundary(boundary_table(heldout, cands, rescorers, cols)).save_model(os.path.join(out, "boundary.txt"))
    with open(os.path.join(out, "columns.json"), "w") as fh:
        json.dump({"switch_head": switch_cols, "rescorer": cols}, fh, indent=1)
    _log("7. prior")
    prior = fit_prior([match_labels(("metrica", g)) for g in EXTERNAL_GAMES])
    with open(os.path.join(out, "prior.json"), "w") as fh:
        json.dump(_jsonable(prior), fh, indent=1)
    _log("8. whistle count")
    k, counts = whistles_per_match()
    _log("whistled events per DFL match: {} -> mean {} (whistle.WHISTLES_PER_MATCH)".format(counts, k))
    _log("done ->", out)
