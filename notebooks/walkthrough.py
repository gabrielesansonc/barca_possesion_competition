"""Numbers behind notebooks/pipeline_walkthrough.ipynb.

Two functions, both cached in data/notebook/ (match data: never committed):

  barath()    runs possession.pipeline step by step on Barcelona vs Athletic and keeps every
              intermediate (heads, evidence, switch head, best path, candidates and their q,
              chosen set, boundary moves). Its final labels must equal predict()'s.
  heldout()   the validation on the 10 external matches, from train.py's own held-out copies
              (data/training/r1.0/heldout_h1500_s300.pkl: 60 copies, each decoded by heads and a
              switch head trained without its match). Rescorers and boundary models are refitted
              leave-one-match-out here with train.py's recipe, so every number is held out.

Neither uses any label of Barcelona vs Athletic.

    python notebooks/walkthrough.py          # fills the cache (~30-60 min the first time)
"""

import os
import pickle
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import possession  # noqa: E402,F401
import possession_metrics as pm  # noqa: E402
from possession import decoder as dm  # noqa: E402
from possession import pipeline as pl  # noqa: E402
from possession import training as tr  # noqa: E402

CACHE = os.path.join(possession.DATA_DIR, "notebook")
HELDOUT = os.path.join(possession.DATA_DIR, "training", "r1.0", "heldout_h1500_s300.pkl")
N_CONTRIB = 20000            # BAR-ATH frames used for per-feature contributions (SHAP-style)


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t = time.time()
    out = build()
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("{} built in {:.0f}s".format(name, time.time() - t), flush=True)
    return out


def spell_f1(labels, codes):
    """spellF1 as the organisers compute it (X forward-filled, graded runs of the labels)."""
    runs = pm.valid_runs(labels)
    return pm.spell_scores(labels, pm.fill_abstentions(codes, runs), runs)["mean_f1"]


# -- Barcelona vs Athletic, step by step ------------------------------------------------------------

def _contrib(booster, X, rows):
    """LightGBM's exact per-feature contributions (TreeSHAP) on some rows, in log-odds."""
    Xs = X.iloc[rows][booster.feature_name()]
    c = booster.predict(Xs, pred_contrib=True)
    return {"cols": booster.feature_name(), "contrib": c[:, :-1].astype(np.float32),
            "bias": float(c[0, -1]), "X": Xs.to_numpy(np.float32)}


def _barath():
    from possession.data import load_match
    from possession.features import tagging_features
    from possession.hsmm import dead_insert_cost
    from possession.utils import known_dead
    from possession.whistle import WHISTLE_SPAN, stop_whistles

    models = pl.load_models()
    m = load_match()
    kd = known_dead(m)
    g = m.graded
    X = pl.frame_features(m)
    p_dead = models["dead"].predict(X[models["dead"].feature_name()])
    p_home = models["team"].predict(X[models["team"].feature_name()])
    raw = pl.head_emission(models, X).astype(float)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(np.flatnonzero(g & ~kd), N_CONTRIB, replace=False))
    contrib = {"dead": _contrib(models["dead"], X, rows), "team": _contrib(models["team"], X, rows),
               "rows": rows}
    del X

    # step 3: evidence, kept term by term
    w_whistle = dead_insert_cost(models["prior"], WHISTLE_SPAN) * pl.TEMPERATURE[1] / WHISTLE_SPAN
    whistles = stop_whistles(raw, m)
    tags = tagging_features(m)
    set_piece = np.nan_to_num(np.asarray(tags["tag_t_since_setpiece"]), nan=99.0) < 2.0
    tag_gap = np.nan_to_num(np.asarray(tags["tag_gap"], float))
    e = pl.match_emission(models, m, raw).astype(float)

    # step 4
    Xs = pl.switch_features(m)
    p_switch = models["switch"].predict(Xs[models["switch_cols"]])
    sw_rows = rows[::2]
    contrib["switch"] = _contrib(models["switch"], Xs[models["switch_cols"]], sw_rows)
    del Xs

    # steps 5-6
    cands, Xc, map_segs = pl.candidate_table(e, m, models["prior"], p_switch, kd)
    q = models["rescorer"].predict(Xc[models["rescorer_cols"]].values)
    chosen = dm.schedule(cands, q - pl.LAMBDA)
    resc_codes = pl.choose(cands, q, map_segs, m, kd)
    contrib["rescorer"] = {"cols": models["rescorer_cols"],
                           "contrib": models["rescorer"].predict(Xc[models["rescorer_cols"]].values,
                                                                 pred_contrib=True)[:, :-1].astype(np.float32),
                           "X": Xc[models["rescorer_cols"]].to_numpy(np.float32)}

    # step 7
    b = pl._boundaries(resc_codes, g)
    Xb = pl.boundary_features(resc_codes, b, raw, p_switch, len(resc_codes))
    shift = models["boundary"].predict(Xb[models["boundary"].feature_name()])
    final = pl.refine_boundaries(models, resc_codes, raw, p_switch, m, kd)
    contrib["boundary"] = {"cols": models["boundary"].feature_name(),
                           "contrib": models["boundary"].predict(Xb[models["boundary"].feature_name()],
                                                                 pred_contrib=True)[:, :-1].astype(np.float32),
                           "X": Xb[models["boundary"].feature_name()].to_numpy(np.float32)}

    argmax = np.full(m.n_frames, pm.X, np.int8)
    argmax[g] = raw[g].argmax(1)
    argmax[kd & g] = pm.D
    map_codes = dm.segments_to_codes(map_segs, m.n_frames)
    return {"n_frames": m.n_frames, "graded": g, "half": m.half, "known_dead": kd,
            "p_dead": p_dead.astype(np.float32), "p_home": p_home.astype(np.float32),
            "raw": raw.astype(np.float32), "e": e.astype(np.float32),
            "whistles": whistles, "whistle_weight": float(w_whistle), "set_piece": set_piece, "tag_gap": tag_gap,
            "p_switch": p_switch.astype(np.float32), "map_segs": map_segs, "cands": cands,
            "cand_X": Xc.astype(np.float32), "q": q, "chosen": chosen,
            "codes": {"argmax": argmax, "map": map_codes, "rescorer": resc_codes.astype(np.int8),
                      "final": final.astype(np.int8)},
            "boundaries": b, "shift": shift, "contrib": contrib}


def barath():
    out = _cached("barath", _barath)
    return out


# -- held-out validation on the 10 external matches ---------------------------------------------

def _heldout():
    import lightgbm as lgb  # noqa: F401
    with open(HELDOUT, "rb") as fh:
        held = pickle.load(fh)
    t = time.time()
    cands = tr.candidate_rows(held)
    print("candidates {:.0f}s".format(time.time() - t), flush=True)
    cols = list(cands[0]["X"].columns)
    folds = tr.fold_rescorers(cands, cols)
    print("fold rescorers {:.0f}s".format(time.time() - t), flush=True)

    by_key = {d["key"]: d for d in cands}
    per_copy, q_rows, bparts = [], [], []
    for r in held:
        d = by_key[r["key"]]
        fold = (d["source"], d["match"])
        q = folds[fold].predict(d["X"][cols].values)
        map_codes = dm.segments_to_codes(d["map"], len(r["labels"]))
        codes = pl.choose(d["cands"], q, d["map"], d["mm"]).astype(np.int8)
        lab = r["labels"]
        argmax = np.where(lab != pm.X, r["emission"].argmax(1), pm.X).astype(np.int8)
        pb = pl._boundaries(codes, d["mm"].graded)
        tb = pl._boundaries(lab, lab != pm.X)
        Xb = pl.boundary_features(codes, pb, r["emission"], r["p_switch"], len(codes))
        Xb["_off"] = tr._match_offsets(pb, tb)
        Xb["_fold"], Xb["_seed"], Xb["_k"] = [fold] * len(Xb), r["seed"], len(per_copy)
        bparts.append(Xb)
        chosen = dm.schedule(d["cands"], q - pl.LAMBDA)
        truth = dm.team_spells(dm.code_segments(lab, pm.valid_runs(lab)))
        per_copy.append({"key": r["key"], "source": r["source"], "match": r["match"], "seed": r["seed"],
                         "argmax": spell_f1(lab, argmax), "map": spell_f1(lab, map_codes),
                         "rescorer": spell_f1(lab, codes), "n_cands": len(q), "n_chosen": len(chosen),
                         "n_true": len(truth), "n_map": len(dm.team_spells(d["map"])),
                         "minutes": float((lab != pm.X).sum() / 1500), "_codes": codes, "_pb": pb})
        q_rows.append(pd.DataFrame({"q": q, "y": d["y"], "source": r["source"],
                                    "chosen": np.isin(np.arange(len(q)), chosen)}))
    print("rescorer decodes {:.0f}s".format(time.time() - t), flush=True)

    # boundary model, leave one match out (fold model: the other 9 matches' dev copies, as the rescorers)
    B = pd.concat(bparts, ignore_index=True)
    bcols = [c for c in B.columns if not c.startswith("_")]
    B["_pred"] = np.nan
    for fold in sorted(set(B["_fold"])):
        tr_rows = (B["_fold"] != fold) & B["_seed"].isin(tr.DEV_SEEDS)
        mdl = tr.fit_boundary(B.loc[tr_rows, bcols + ["_off"]])
        te = B["_fold"] == fold
        B.loc[te, "_pred"] = mdl.predict(B.loc[te, bcols])
    lab_of = {r["key"]: r["labels"] for r in held}
    for k, pc in enumerate(per_copy):
        rows = B["_k"] == k
        lab = lab_of[pc["key"]]
        moved = pl._move(pc["_codes"], pc["_pb"], np.round(B.loc[rows, "_pred"].to_numpy()), lab != pm.X)
        pc["final"] = spell_f1(lab, moved)
        pc["n_final"] = len(dm.team_spells(dm.code_segments(moved, pm.valid_runs(lab))))
        del pc["_codes"], pc["_pb"]
    print("boundary LOMO {:.0f}s".format(time.time() - t), flush=True)

    # per-frame head metrics: every 5th labelled frame of each copy
    hrows, srows = [], []
    for r in held:
        lab = r["labels"]
        ok = np.flatnonzero(lab != pm.X)[::5]
        e = r["emission"][ok].astype(float)
        hrows.append(pd.DataFrame({"y": lab[ok], "p_dead": np.exp(e[:, 2]),
                                   "p_home": np.exp(e[:, 0] - np.logaddexp(e[:, 0], e[:, 1])),
                                   "source": r["source"], "match": r["match"]}))
        from possession.turnover import switch_target
        ys = switch_target(lab)
        okk = np.flatnonzero(ys >= 0)[::5]
        srows.append(pd.DataFrame({"y": ys[okk], "p": r["p_switch"][okk], "source": r["source"]}))
    return {"per_copy": pd.DataFrame(per_copy), "q": pd.concat(q_rows, ignore_index=True),
            "bound": B[["kind", "_off", "_pred", "_fold", "_seed"]].rename(columns={"_off": "off", "_pred": "pred"}),
            "heads": pd.concat(hrows, ignore_index=True), "switch": pd.concat(srows, ignore_index=True),
            "rescorer_cols": cols}


def heldout():
    return _cached("heldout", _heldout)


if __name__ == "__main__":
    t0 = time.time()
    b = barath()
    from possession.pipeline import predict  # noqa: F401
    print("BAR-ATH done {:.0f}s".format(time.time() - t0), flush=True)
    h = heldout()
    print(h["per_copy"].groupby(["source", h["per_copy"].seed == 777])[["argmax", "map", "rescorer", "final"]].mean())
    print("done {:.0f}s".format(time.time() - t0))
