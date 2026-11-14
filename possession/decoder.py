"""Candidate possessions from the duration model (hsmm.py), and the final choice among them.

  HSMM / posterior   forward pass of the duration model and sampling of whole possession
                     sequences from it (the sampled paths give many alternative possessions).
  viterbi_segments   the single most likely sequence (the "best path").
  candidate_set      every distinct possession found in the best path and the samples.
  spell_features     per candidate: average evidence inside it and at its edges, and how often the
                     samples contain it (read by the rescorer).
  schedule           the set of non-overlapping candidates with the highest total score.
  fill_gaps          the final timeline: the chosen possessions, with the best path filling the
                     frames between them.

Inputs are an (n, 3) array of log-probabilities for H, A and D and the match (only its frame
count and graded frames are used).
"""

import numpy as np

import possession_metrics as pm  # noqa: E402
from possession.utils import runs_of
from possession.hsmm import (N_AGE, OUT, SA, SDA, SDH, SH, STEP, _log_transitions, _viterbi_blocks,
                       tempered)

IOUS = np.array(pm.SPELL_IOUS)


# -- the model --------------------------------------------------------------------------------

def biased_prior(prior, team_bias=0.0, dead_bias=0.0):
    """Prior with each hazard's logit shifted: >0 ends spells sooner (more, shorter spells)."""
    out = dict(prior)
    for key, b in (("team_hazard", team_bias), ("dead_hazard", dead_bias)):
        if b:
            lh, ls = prior[key]
            logit = lh - ls + b
            out[key] = (-np.logaddexp(0, -logit), -np.logaddexp(0, logit))
    return out


class HSMM:
    """Scaled forward-backward and FFBS for possession.hsmm's model on one run of block emissions."""

    def __init__(self, prior):
        lh = np.stack([prior["team_hazard"][0]] * 2 + [prior["dead_hazard"][0]] * 2)
        ls = np.stack([prior["team_hazard"][1]] * 2 + [prior["dead_hazard"][1]] * 2)
        self.h, self.stay = np.exp(lh), np.exp(ls)              # (4, N_AGE)
        self.trans = np.exp(_log_transitions(prior))            # (4, 4), 0 where impossible
        self.prior = prior

    def forward(self, em):
        """em (nb, 4) log block emissions. Sets alpha (nb, 4, N_AGE) normalised per block,
        the scaled emissions and the scale factors; returns log-likelihood."""
        nb = len(em)
        e = np.exp(em - em.max(axis=1, keepdims=True))           # (nb, 4) scaled emissions
        alpha = np.zeros((nb, 4, N_AGE), dtype=np.float32)
        c = np.empty(nb)
        a = np.zeros((4, N_AGE))
        a[:, 0] = e[0] / 4
        c[0] = a.sum()
        a /= c[0]
        alpha[0] = a
        h, stay, trans = self.h, self.stay, self.trans
        for t in range(1, nb):
            end = (a * h).sum(axis=1)                             # (4,) a spell ends after t-1
            cont = a * stay
            new = np.empty_like(a)
            new[:, 0] = end @ trans
            new[:, 1:] = cont[:, :-1]
            new[:, -1] += cont[:, -1]
            new *= e[t][:, None]
            c[t] = new.sum()
            a = new / c[t]
            alpha[t] = a
        self.alpha, self.e, self.c = alpha, e, c
        return float(np.log(c).sum() + em.max(axis=1).sum())

    def marginals(self):
        """(nb, 4) posterior P(state) per block (after forward)."""
        alpha, e, c = self.alpha, self.e, self.c
        h, stay, trans = self.h, self.stay, self.trans
        nb = len(alpha)
        gamma = np.empty((nb, 4))
        b = np.ones((4, N_AGE))
        g = alpha[-1] * b
        gamma[-1] = g.sum(axis=1) / g.sum()
        for t in range(nb - 2, -1, -1):
            eb = e[t + 1][:, None] * b                            # (4, N_AGE)
            start = trans @ eb[:, 0]                              # (4,) next block starts a spell
            shifted = np.empty_like(b)
            shifted[:, :-1] = eb[:, 1:]
            shifted[:, -1] = eb[:, -1]
            b = (stay * shifted + h * start[:, None]) / c[t + 1]
            g = alpha[t] * b
            gamma[t] = g.sum(axis=1) / g.sum()
        return gamma

    def sample(self, rng, n_samples):
        """n_samples state sequences, each as an (k, 3) int array of [start_block, stop_block, state]."""
        alpha, h, stay, trans = self.alpha, self.h, self.stay, self.trans
        nb = len(alpha)
        last = alpha[-1].astype(float).ravel()
        last_cdf = np.cumsum(last / last.sum())
        out = []
        for _ in range(n_samples):
            k = min(int(np.searchsorted(last_cdf, rng.random(), side="right")), len(last) - 1)
            s, a = divmod(k, N_AGE)
            t = nb - 1
            segs = []
            stop = nb
            while True:
                if a == N_AGE - 1:                                # in the tail: walk back
                    while t > 0:
                        w_tail = alpha[t - 1, s, N_AGE - 1] * stay[s, N_AGE - 1]
                        w_age = alpha[t - 1, s, N_AGE - 2] * stay[s, N_AGE - 2]
                        t -= 1
                        if rng.random() * (w_tail + w_age) >= w_tail:
                            a = N_AGE - 2
                            break
                    else:
                        a = 0
                start = t - a
                segs.append((start, stop, s))
                if start == 0:
                    break
                w = (alpha[start - 1] * h) * trans[:, s][:, None]  # (4, N_AGE) previous spell ends
                w = w.ravel()
                cdf = np.cumsum(w)
                k = min(int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right")), len(w) - 1)
                s, a = divmod(k, N_AGE)
                stop = start
                t = start - 1
            out.append(np.array(segs[::-1], dtype=np.int64))
        return out


# -- blocks <-> frames ------------------------------------------------------------------------

def block_emissions(e, start, stop):
    """(nb, 4) block sums of tempered (n, 3) emissions over [start, stop), as possession.hsmm."""
    n = stop - start
    nb = int(np.ceil(n / STEP))
    pad = np.zeros((nb * STEP, 3))
    pad[:n] = e[start:stop]
    block = pad.reshape(nb, STEP, 3).sum(axis=1)
    return block[:, [0, 1, 2, 2]]


def states_to_segments(states):
    """(nb,) block states -> (k, 3) [start_block, stop_block, state] runs."""
    breaks = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(states)]))
    return np.column_stack([starts, stops, states[starts]]).astype(np.int64)


def segments_to_label_segments(segs, start, n):
    """Block-state segments of one run -> frame segments [start, stop, pm code], merging D after H
    with D after A (both pm.D) and clipping the last block to the run."""
    codes = OUT[segs[:, 2]]
    s = start + segs[:, 0] * STEP
    e = np.minimum(start + segs[:, 1] * STEP, start + n)
    keep = np.concatenate(([True], codes[1:] != codes[:-1]))
    idx = np.flatnonzero(keep)
    stops = np.concatenate((s[idx[1:]], [e[-1]]))
    return np.column_stack([s[idx], stops, codes[idx]]).astype(np.int64)


def segments_to_codes(segs, n_frames):
    labels = np.full(n_frames, pm.X, dtype=np.int8)
    for a, b, c in segs:
        labels[a:b] = c
    return labels


def code_segments(codes, runs):
    """(k, 3) [start, stop, code] constant runs of codes inside each run."""
    out = []
    for start, stop in runs:
        w = codes[start:stop]
        if not len(w):
            continue
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        b = np.concatenate(([0], breaks, [len(w)]))
        out.append(np.column_stack([start + b[:-1], start + b[1:], w[b[:-1]]]))
    return np.concatenate(out).astype(np.int64) if out else np.zeros((0, 3), dtype=np.int64)


def team_spells(segs):
    return segs[(segs[:, 2] == pm.H) | (segs[:, 2] == pm.A)]


# -- fast exact spellF1 -----------------------------------------------------------------------

def overlap_pairs(g, p):
    """(iou, gi, pj) for every same-team overlapping pair of two (k, 3) spell arrays sorted by start."""
    ious, gis, pjs = [], [], []
    for code in (pm.H, pm.A):
        gi = np.flatnonzero(g[:, 2] == code)
        pj = np.flatnonzero(p[:, 2] == code)
        if not len(gi) or not len(pj):
            continue
        gs, ge = g[gi, 0], g[gi, 1]
        ps, pe = p[pj, 0], p[pj, 1]
        lo = np.searchsorted(pe, gs, side="right")             # first pred with stop > gt start
        hi = np.searchsorted(ps, ge, side="left")              # preds with start < gt stop
        cnt = np.maximum(hi - lo, 0)
        if not cnt.sum():
            continue
        rep = np.repeat(np.arange(len(gi)), cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        j = lo[rep] + off
        inter = np.minimum(ge[rep], pe[j]) - np.maximum(gs[rep], ps[j])
        union = np.maximum(ge[rep], pe[j]) - np.minimum(gs[rep], ps[j])
        ious.append(inter / union.astype(float))
        gis.append(gi[rep])
        pjs.append(pj[j])
    if not ious:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ious), np.concatenate(gis), np.concatenate(pjs)


# -- decoding ---------------------------------------------------------------------------------

def run_inputs(emission, m, temperature, known_dead=None):
    """Tempered emissions with known-dead frames forced, and the graded runs."""
    t_team, t_dead = temperature if np.ndim(temperature) else (temperature, temperature)
    e = tempered(emission, t_team, t_dead)
    if known_dead is not None:
        e[known_dead, 0] = e[known_dead, 1] = -1e4
    return e, [(int(a), int(b)) for a, b in runs_of(m.graded)]


def viterbi_segments(emission, m, prior, temperature, known_dead=None):
    """MAP timeline (frame segments with pm codes) under prior/temperature; == possession.hsmm.decode_hsmm."""
    e, runs = run_inputs(emission, m, temperature, known_dead)
    out = []
    for start, stop in runs:
        states = _viterbi_blocks(block_emissions(e, start, stop), prior)
        out.append(segments_to_label_segments(states_to_segments(states), start, stop - start))
    return np.concatenate(out)


def posterior(emission, m, prior, temperature, known_dead=None, n_samples=0, seed=0,
              marginals=False):
    """Per run: forward pass, optional block marginals (n_frames, 3 = H/A/D) and FFBS samples.
    Samples come back as frame segment arrays (pm codes) over all runs. Deterministic for a seed."""
    e, runs = run_inputs(emission, m, temperature, known_dead)
    rng = np.random.default_rng(seed)
    samples = [[] for _ in range(n_samples)]
    marg = np.zeros((m.n_frames, 3), dtype=np.float32) if marginals else None
    loglik = 0.0
    for start, stop in runs:
        n = stop - start
        model = HSMM(prior)
        loglik += model.forward(block_emissions(e, start, stop))
        if marginals:
            g = model.marginals()
            g3 = np.column_stack([g[:, 0], g[:, 1], g[:, 2] + g[:, 3]])
            marg[start:stop] = np.repeat(g3, STEP, axis=0)[:n]
        for i, segs in enumerate(model.sample(rng, n_samples)):
            samples[i].append(segments_to_label_segments(segs, start, n))
        del model
    samples = [np.concatenate(s) for s in samples]
    return {"samples": samples, "marginals": marg, "loglik": loglik, "runs": runs}


# -- minimum-Bayes-risk local search ----------------------------------------------------------
#
# Objective: U(P) = mean over posterior samples s of spellF1(P, s) = mean_s 2 V_s / (n_P + n_s),
# where V_s = sum over matched pairs of the share of IoU thresholds the pair clears (so the mean
# over thresholds of TP), n_P / n_s = number of possessions. The one-to-one greedy matching of pm
# decomposes over connected groups of overlapping spells, so an edit's effect on V_s is computed
# exactly enough on a local window (the edited spells plus one unchanged possession either side).
#
# Edits are "paint [a, b) with code c" (then merge equal neighbours): it covers inserting a spell
# (a team spell or a stoppage inside another spell), deleting one (paint it with a neighbour's
# code) and moving a boundary (paint the strip between old and new position). Candidates come
# from the samples: their segments, their boundary positions near ours, and deletions of ours.



# -- spell candidates, their support in the samples, and interval scheduling ------------------

def unique_spells(seg_lists):
    """Unique (start, stop, code) team spells over several frame-segment arrays, sorted by start."""
    allsp = np.concatenate([team_spells(s) for s in seg_lists])
    allsp = np.unique(allsp, axis=0)
    return allsp[np.lexsort((allsp[:, 1], allsp[:, 0]))]


def best_ious(cands, ref_spells):
    """(n_cand,) best same-team IoU of each candidate with a (k, 3) spell array (0 if none)."""
    iou, ci, _ = overlap_pairs(cands, ref_spells)
    out = np.zeros(len(cands))
    np.maximum.at(out, ci, iou)
    return out


def support(cands, sample_segs):
    """(n_cand, S) best IoU of each candidate with each sample's same-team spells."""
    return np.column_stack([best_ious(cands, team_spells(s)) for s in sample_segs])


def schedule(cands, value):
    """Max-sum set of non-overlapping candidates; same-team ones may not touch (they would merge
    into one possession), other-team ones may. Exact weighted interval scheduling.
    Returns chosen indices sorted by start."""
    n = len(cands)
    if not n:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((cands[:, 0], cands[:, 1]))          # by end, then start
    c = cands[order]
    v = value[order]
    ends = c[:, 1]
    best = np.empty(n)                 # best chain ending with candidate i (in end order)
    back = np.full(n, -1, dtype=np.int64)
    pref = np.full(n + 1, -np.inf)     # pref[p] = max(0, best[0..p-1]); arg in pref_arg
    pref[0] = 0.0
    pref_arg = np.full(n + 1, -1, dtype=np.int64)
    touch = {}                         # end frame -> {team: (best, index)}
    for i in range(n):
        start, team = int(c[i, 0]), int(c[i, 2])
        p = int(np.searchsorted(ends, start, side="left"))   # ends strictly before start
        cand_v, cand_j = pref[p], pref_arg[p]
        for t2, (bv, bj) in touch.get(start, {}).items():
            if t2 != team and bv > cand_v:
                cand_v, cand_j = bv, bj
        best[i] = v[i] + cand_v
        back[i] = cand_j
        d = touch.setdefault(int(ends[i]), {})
        if team not in d or best[i] > d[team][0]:
            d[team] = (best[i], i)
        if best[i] > pref[i]:
            pref[i + 1], pref_arg[i + 1] = best[i], i
        else:
            pref[i + 1], pref_arg[i + 1] = pref[i], pref_arg[i]
    chosen = []
    j = pref_arg[n]
    while j >= 0:
        chosen.append(order[j])
        j = back[j]
    return np.array(sorted(chosen, key=lambda k: cands[k, 0]), dtype=np.int64)


# -- learned spell rescorer: features of candidate spells --------------------------------------

def _window_means(x, starts, stops):
    """Mean of x over each [start, stop) (x 1-D), via a cumulative sum."""
    cs = np.concatenate(([0.0], np.cumsum(x)))
    n = np.maximum(stops - starts, 1)
    return (cs[stops] - cs[starts]) / n


def candidate_set(map_segs, sample_sets):
    """MAP spells plus every spell any posterior sample has."""
    return unique_spells([map_segs] + [s for ss in sample_sets for s in ss])


def spell_features(cands, emission, map_segs, sample_sets, n_frames):
    """Tabular features per candidate spell (n, k). emission: (n_frames, 3) H/A/D log-probs."""
    import pandas as pd
    e = np.asarray(emission, dtype=float)
    alive = np.logaddexp(e[:, 0], e[:, 1])
    p_dead = np.exp(np.clip(e[:, 2], -50, 0))
    p_h = np.exp(e[:, 0] - alive)                   # P(H | alive)
    s, t, c = cands[:, 0], cands[:, 1], cands[:, 2]
    home = c == pm.H
    dur = t - s
    f = {"log_dur": np.log(dur)}
    for name, x in (("p_team", None), ("p_dead", p_dead)):
        for tag, a, b in (("in", s, t), ("pre", np.maximum(s - 50, 0), s), ("post", t, np.minimum(t + 50, n_frames)),
                          ("head", s, np.minimum(s + 25, t)), ("tail", np.maximum(t - 25, s), t)):
            if name == "p_team":
                v = _window_means(p_h, a, b)
                v = np.where(home, v, 1 - v)
            else:
                v = _window_means(x, a, b)
            f["{}_{}".format(name, tag)] = v
    map_sp = team_spells(map_segs)
    f["map_iou"] = best_ious(cands, map_sp)
    map_other = map_sp.copy()
    map_other[:, 2] = 1 - map_other[:, 2]
    f["map_other_iou"] = best_ious(cands, map_other)        # overlaps a MAP spell of the other team
    for k, ss in enumerate(sample_sets):
        sup = support(cands, ss)                               # (n, S)
        share = np.stack([(sup >= th).mean(axis=1) for th in IOUS], axis=1)
        f["s{}_q".format(k)] = share.mean(axis=1)
        f["s{}_iou_mean".format(k)] = sup.mean(axis=1)
        f["s{}_p30".format(k)] = share[:, 0]
        f["s{}_p70".format(k)] = share[:, -1]
    return pd.DataFrame(f)


# -- the rescored decoder ---------------------------------------------------------------------
#
# The HSMM posterior is badly over-confident about spells (held-out external copies: samples agree
# with the MAP at spellF1 ~0.96 while the MAP scores ~0.62 against the truth), so MBR against its
# own samples can't find the possessions the MAP misses. Instead, a small LightGBM learns, on
# held-out external decodes, the probability that a candidate spell matches a true possession
# (mean over the IoU thresholds of "matched at this IoU"). The timeline is then the set of
# non-overlapping candidates maximising sum(q - lam): spellF1 = 2 TP / (n_pred + n_true), so a
# spell is worth predicting when its expected TP share beats lam ~ F / 2. Frames no chosen spell
# covers are D (the only label that adds no possession).

N_SAMPLES = 40


def fill_gaps(chosen_segs, map_codes, m, known_dead=None):
    """Timeline from chosen spells + the MAP for the frames between them, so the chosen spells
    don't get separated by artificial D gaps:
      gap with a MAP stoppage in it          -> D
      gap between spells of different teams  -> split where the MAP switches (else at its edge)
      gap next to one team spell             -> that team where the MAP says so, else D
      gap between spells of the same team    -> D (they must stay two possessions)"""
    codes = np.full(m.n_frames, pm.X, dtype=np.int8)
    for start, stop in runs_of(m.graded):
        start, stop = int(start), int(stop)
        sp = chosen_segs[(chosen_segs[:, 0] >= start) & (chosen_segs[:, 1] <= stop)]
        codes[start:stop] = pm.D
        for s, e, k in sp:
            codes[s:e] = k
        bounds = [(start, start, -1)] + [tuple(x) for x in sp.tolist()] + [(stop, stop, -1)]
        for (_, g0, cl), (g1, _, cr) in zip(bounds[:-1], bounds[1:]):
            if g1 <= g0:
                continue
            mp = map_codes[g0:g1]
            if (mp == pm.D).any():
                continue                                   # a stoppage: D
            if cl >= 0 and cr >= 0 and cl != cr:
                x = np.flatnonzero(mp == cr)               # MAP already on the right team
                cut = g0 + (int(x[0]) if len(x) else g1 - g0)
                codes[g0:cut], codes[cut:g1] = cl, cr
            elif cl >= 0 and cr != cl and (mp == cl).all():
                codes[g0:g1] = cl
            elif cr >= 0 and cl != cr and (mp == cr).all():
                codes[g0:g1] = cr
    if known_dead is not None:
        codes[known_dead & m.graded] = pm.D
    return codes
