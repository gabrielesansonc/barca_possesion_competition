"""The whole method, from the match data to one label per frame.

    from possession.data import load_match
    from possession.pipeline import predict
    codes = predict(load_match())          # (199319,) label codes: H, A, D or X

Steps (each model was trained on other matches only, never on this one):
  1. Frame features      positions of players and ball -> ~110 numbers per frame (features.py,
                         interact.py).
  2. Two heads           LightGBM: P(dead ball) and P(home has it | ball in play), per frame.
  3. Match evidence      Smart Tagging (set pieces, untagged gaps) and referee whistles (whistle.py)
                         add log-odds towards "dead ball" where they apply.
  4. Switch head         LightGBM: P(possession changes within 0.5 s), per frame (turnover.py).
  5. Candidates          a duration model (hsmm.py, decoder.py) proposes possible possessions:
                         its best path, 120 sampled paths, and cuts at the switch head's peaks.
  6. Rescorer            LightGBM scores each candidate (chance it matches a real possession);
                         the best non-overlapping set is kept, gaps are filled from the best path.
  7. Boundary model      LightGBM moves each label change by up to 1 s to where it most likely is.
"""

import json
import os

import numpy as np
import pandas as pd

from possession import MODELS_DIR
from possession import decoder as dm
from possession import interact as ix
from possession import peaks as pk
from possession import rescorer as rs
from possession import turnover as to
from possession.features import TAG_COLS, build_features, tagging_features
from possession.hsmm import dead_insert_cost
from possession.utils import known_dead
from possession.whistle import WHISTLE_SPAN, add_whistle_evidence
import possession_metrics as pm

TEMPERATURE = (2, 8)          # (team, dead): how much the decoder trusts the heads
SAMPLE_SETS = (((2, 8), 0.0), ((2, 8), 1.5), ((2, 8), 3.0))   # (temperature, bias to shorter spells)
PEAK_HEIGHT, PEAK_SPAN = 0.2, 5   # switch-head peaks used to cut extra candidates
LAMBDA = 0.4                  # a candidate is kept only if its score beats this
EPS = 1e-4                    # probability floor before taking logs
# Smart Tagging evidence (log-odds per frame towards a dead ball)
SET_PIECE_WEIGHT = 3.0        # first 2 s of a set-piece tag: a set piece is a restart, so the ball is
                              # dead just before it. 50 frames x 3 outweighs the cost of entering and
                              # leaving a dead spell (2 x 40), so the tag is near-certain evidence.
TAG_GAP_WEIGHT = 0.6          # frames no Smart Tagging phase covers (mostly stoppages)
# Boundary model
BOUNDARY_OFFSETS = (-25, -15, -10, -6, -3, 0, 3, 6, 10, 15, 25)
BOUNDARY_CLIP = 25            # frames a boundary may move
BOUNDARY_MIN_KEEP = 5         # frames each neighbouring spell keeps


# -- models ---------------------------------------------------------------------------------------

def _booster(name):
    import lightgbm as lgb
    return lgb.Booster(model_file=os.path.join(MODELS_DIR, name + ".txt"))


def _from_json(v):
    if isinstance(v, dict) and "__array__" in v:
        return np.array(v["__array__"], dtype=v["dtype"])
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(_from_json(x) for x in v["__tuple__"])
    if isinstance(v, dict):
        return {k: _from_json(x) for k, x in v.items()}
    return v


def load_models():
    """Every trained piece, from models/."""
    with open(os.path.join(MODELS_DIR, "columns.json")) as fh:
        cols = json.load(fh)
    with open(os.path.join(MODELS_DIR, "prior.json")) as fh:
        prior = _from_json(json.load(fh))
    return {"dead": _booster("dead_head"), "team": _booster("team_head"),
            "switch": _booster("switch_head"), "switch_cols": cols["switch_head"],
            "rescorer": _booster("rescorer"), "rescorer_cols": cols["rescorer"],
            "boundary": _booster("boundary"), "prior": prior}


# -- steps 1-3: per-frame evidence ------------------------------------------------------------------

def frame_features(m):
    """Step 1: the heads' input table, one row per frame."""
    X = build_features(m, tagging=False)
    X = X[[c for c in X.columns if c not in TAG_COLS]].reset_index(drop=True)
    X = pd.concat([X, ix.interaction_features(m, X)], axis=1).astype(np.float32)
    return X


def head_emission(models, X):
    """Step 2: (n, 3) log-probabilities of H, A and D from the two heads."""
    p_dead = np.mean([models["dead"].predict(X[models["dead"].feature_name()])], axis=0)
    p_home = np.mean([models["team"].predict(X[models["team"].feature_name()])], axis=0)
    p_dead = np.clip(p_dead, EPS, 1 - EPS)
    p_home = np.clip(p_home, EPS, 1 - EPS)
    return np.column_stack([np.log1p(-p_dead) + np.log(p_home),
                            np.log1p(-p_dead) + np.log1p(-p_home),
                            np.log(p_dead)])


def add_tagging_evidence(emission, m):
    """Step 3a: Smart Tagging evidence. Matches without tagging are unchanged."""
    tags = tagging_features(m)
    out = emission.copy()
    set_piece = np.nan_to_num(tags["tag_t_since_setpiece"], nan=99.0) < 2.0
    out[:, 2] += SET_PIECE_WEIGHT * set_piece
    out[:, 2] += TAG_GAP_WEIGHT * np.nan_to_num(tags["tag_gap"])
    return out


def match_emission(models, m, raw):
    """Steps 2-3: head emission plus whistle and tagging evidence."""
    weight = dead_insert_cost(models["prior"], WHISTLE_SPAN) * TEMPERATURE[1] / WHISTLE_SPAN
    return add_tagging_evidence(add_whistle_evidence(raw, m, weight), m)


# -- steps 4-6: candidate possessions and the rescorer --------------------------------------------

def switch_features(m):
    """Step 4's input table: the frame features plus before-vs-after comparisons."""
    X = build_features(m, tagging=False)
    X = X[[c for c in X.columns if c not in TAG_COLS]].reset_index(drop=True)
    return pd.concat([X, to.turnover_features(m, X)], axis=1).astype(np.float32)


def switch_probability(models, m):
    """Step 4: P(possession changes within 0.5 s), per frame."""
    return models["switch"].predict(switch_features(m)[models["switch_cols"]])


def _merge(a, b):
    c = np.unique(np.concatenate([a, b]), axis=0)
    return c[np.lexsort((c[:, 1], c[:, 0]))]


def candidates(e, m, prior, p_switch, kd=None):
    """Step 5: candidate possessions with their features (cands, X, best path).
    kd: frames known to be dead (None for the training matches)."""
    map_segs = dm.viterbi_segments(e, m, prior, TEMPERATURE, kd)
    sets = [dm.posterior(e, m, dm.biased_prior(prior, tb), T, kd, n_samples=dm.N_SAMPLES, seed=k)["samples"]
            for k, (T, tb) in enumerate(SAMPLE_SETS)]
    base = dm.candidate_set(map_segs, sets)
    graded = m.graded if kd is None else m.graded & ~kd
    cands = _merge(base, pk.peak_candidates(p_switch, e, map_segs, graded, PEAK_HEIGHT, PEAK_SPAN))
    X = dm.spell_features(cands, e, map_segs, sets, m.n_frames)
    known = {tuple(c) for c in base.tolist()}
    X["from_peaks"] = np.array([tuple(c) not in known for c in cands.tolist()], np.float32)
    return cands, X, map_segs


def candidate_table(e, m, prior, p_switch, kd=None):
    """Steps 5-6 input: (candidates, their features for the rescorer, best path)."""
    cands, X, map_segs = candidates(e, m, prior, p_switch, kd)
    extra = rs.extra_features(cands, e, map_segs, X, m.n_frames)
    X = pd.concat([X.reset_index(drop=True), pd.DataFrame(extra)], axis=1).astype(np.float32)
    X = pd.concat([X.reset_index(drop=True), to.candidate_switch_features(cands, p_switch)], axis=1)
    return cands, X, map_segs


def choose(cands, q, map_segs, m, kd=None):
    """Step 6: the non-overlapping candidates with the highest total (q - LAMBDA), gaps from the
    best path; (n,) label codes."""
    chosen = cands[dm.schedule(cands, q - LAMBDA)]
    return dm.fill_gaps(chosen, dm.segments_to_codes(map_segs, m.n_frames), m, kd)


def rescore(models, e, m, p_switch, kd):
    """Steps 5-6: keep the best set of candidate possessions; (n,) label codes."""
    cands, X, map_segs = candidate_table(e, m, models["prior"], p_switch, kd)
    q = models["rescorer"].predict(X[models["rescorer_cols"]].values)
    return choose(cands, q, map_segs, m, kd)


# -- step 7: boundary model -----------------------------------------------------------------------

def _boundaries(codes, graded):
    """(k, 3) [frame, from, to] label changes inside graded runs."""
    out = []
    for s, t in pm.valid_runs(np.where(graded, codes, pm.X)):
        c = codes[s:t]
        i = np.flatnonzero(c[1:] != c[:-1]) + 1
        out.append(np.column_stack([s + i, c[i - 1], c[i]]))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def _kind_id(a, b):
    return {(pm.H, pm.A): 0, (pm.A, pm.H): 1, (pm.H, pm.D): 2, (pm.A, pm.D): 3, (pm.D, pm.H): 4,
            (pm.D, pm.A): 5}.get((int(a), int(b)), 6)


def boundary_features(codes, b, e, p_switch, n):
    """Evidence around each label change, seen from its kind (from-label a, to-label c)."""
    from scipy.stats import rankdata
    e = np.asarray(e, float)
    p_dead = np.exp(np.clip(e[:, 2], -50, 0))
    alive = np.logaddexp(e[:, 0], e[:, 1])
    p_h = np.exp(e[:, 0] - alive)
    ps = np.nan_to_num(np.asarray(p_switch, float))
    ps = (rankdata(ps, method="average") / len(ps)).astype(float)   # rank within the match
    f, a, c = b[:, 0].astype(int), b[:, 1], b[:, 2]
    ll = np.asarray(e, float)
    col = {pm.H: 0, pm.A: 1, pm.D: 2}
    ia = np.array([col.get(int(x), 2) for x in a])
    ic = np.array([col.get(int(x), 2) for x in c])
    X = {"kind": np.array([_kind_id(x, y) for x, y in zip(a, c)], float)}
    clipf = lambda x: np.clip(x, 0, n - 1)  # noqa: E731
    for o in BOUNDARY_OFFSETS:
        g = clipf(f + o)
        X["dll_{}".format(o)] = ll[g, ic] - ll[g, ia]
        X["psw_{}".format(o)] = ps[g]
        X["pd_{}".format(o)] = p_dead[g]
        X["ph_{}".format(o)] = p_h[g]
    win = np.stack([ps[clipf(f + k)] for k in range(-BOUNDARY_CLIP, BOUNDARY_CLIP + 1)], axis=1)
    X["psw_peak_k"] = np.argmax(win, axis=1) - BOUNDARY_CLIP
    X["psw_peak_v"] = win.max(axis=1)
    X["len_left"] = np.log1p(f - np.r_[0, f[:-1]])
    X["len_right"] = np.log1p(np.r_[f[1:], n] - f)
    return pd.DataFrame(X).astype(np.float32)


def _move(codes, b, shift, graded):
    """Move each label change b[i] by shift[i] frames, keeping BOUNDARY_MIN_KEEP frames each side."""
    c = codes.copy()
    f = b[:, 0].astype(int)
    prev = np.r_[-10 ** 9, f[:-1]]
    nxt = np.r_[f[1:], 10 ** 9]
    for i in range(len(f)):
        k = int(shift[i])
        if k == 0:
            continue
        nf = int(np.clip(f[i] + k, prev[i] + BOUNDARY_MIN_KEEP, nxt[i] - BOUNDARY_MIN_KEEP))
        if nf < f[i]:
            if not graded[nf:f[i]].all():
                continue
            c[nf:f[i]] = b[i, 2]
        elif nf > f[i]:
            if not graded[f[i]:nf].all():
                continue
            c[f[i]:nf] = b[i, 1]
    return c


def refine_boundaries(models, codes, raw, p_switch, m, kd):
    """Step 7: move each label change by the boundary model's predicted offset."""
    b = _boundaries(codes, m.graded)
    if not len(b):
        return codes
    X = boundary_features(codes, b, raw, p_switch, len(codes))
    shift = models["boundary"].predict(X[models["boundary"].feature_name()])
    out = _move(codes, b, np.round(1.0 * shift), m.graded & ~kd)
    out[kd & m.graded] = pm.D
    return out


# -- all steps ------------------------------------------------------------------------------------

def predict(m, models=None):
    """(n,) label codes for a match (possession_metrics codes: H, A, D, X)."""
    models = models or load_models()
    kd = known_dead(m)
    raw = head_emission(models, frame_features(m)).astype(float)
    e = match_emission(models, m, raw).astype(float)
    p_switch = switch_probability(models, m)
    codes = rescore(models, e, m, p_switch, kd)
    return refine_boundaries(models, codes, raw, p_switch, m, kd)
