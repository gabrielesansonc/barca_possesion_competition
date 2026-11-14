"""A duration model of football: how long possessions and stoppages last, and what follows what.

It turns noisy per-frame probabilities into a sequence of possessions. Everything in it is
learned from label sequences of the training matches (prior.json), not from this match.

  durations     each state knows how long it has lasted. The chance it ends now comes from the
                real distribution of possession and stoppage lengths, so a 0.2 s blip is as rare
                as in real football.
  restarts      the dead ball has two hidden states, "dead after home" and "dead after away", so
                the model remembers who had the ball and uses the real chance that the same team
                restarts. Both are written as D.
  temperature   neighbouring frames carry almost the same information, so their log-probabilities
                are divided by T before adding them up (one T for "which team", one for "dead or
                alive").

Frames are handled in blocks of STEP frames (0.2 s) to keep it fast.

    prior = fit_prior(label_sequences)      # how prior.json was made
"""

import numpy as np

import possession_metrics as pm  # noqa: E402

FPS = 25

# Params
STEP = 5                 # frames per block (0.2 s)
MAX_AGE_S = 60.0         # s of explicit age; longer spells use the tail hazard
SMOOTH_BLOCKS = 2.0      # Gaussian smoothing of the length histograms, in blocks
PSEUDO = 0.5             # pseudo-count per age bin, so no length is impossible
EPS = 1e-6

# Hidden states. Output labels: H, A, D, D
SH, SA, SDH, SDA = 0, 1, 2, 3
OUT = np.array([pm.H, pm.A, pm.D, pm.D], dtype=np.int8)
N_AGE = int(MAX_AGE_S * FPS / STEP)


def _segments(labels):
    """[(start, stop, code)] inside each valid run, plus whether each touches a run edge."""
    out = []
    for start, stop in pm.valid_runs(labels):
        segs = pm.segments(labels, [(start, stop)])
        for i, (s, e, c) in enumerate(segs):
            out.append((s, e, c, i == 0 or i == len(segs) - 1))
    return out


def _hazard(lengths_blocks):
    """(N_AGE,) log h and log (1 - h) per age (age 0 = first block), from observed lengths."""
    from scipy.ndimage import gaussian_filter1d
    lengths = np.clip(np.asarray(lengths_blocks), 1, None)
    counts = np.bincount(np.minimum(lengths, N_AGE + 1), minlength=N_AGE + 2)[1:].astype(float)
    tail = counts[N_AGE]                                   # lengths beyond the explicit ages
    body = gaussian_filter1d(counts[:N_AGE], SMOOTH_BLOCKS, mode="nearest") + PSEUDO
    survival = body[::-1].cumsum()[::-1] + tail            # P(length >= age), unnormalised
    h = np.clip(body / survival, EPS, 1 - EPS)
    long = lengths[lengths > N_AGE]
    h_tail = 1.0 / max(1.0, np.mean(long - N_AGE)) if len(long) else h[-1]
    h[-1] = np.clip(h_tail, EPS, 1 - EPS)                  # last age: constant hazard from here on
    return np.log(h), np.log1p(-h)


def fit_prior(label_sequences):
    """Durations and transitions from (n,) label arrays (pm codes; X outside the match).

    Lengths come from spells that start and end inside a run (edge spells are cut short).
    Team spells pool H and A, as do stoppages: the prior has no home side.
    """
    team_len, dead_len = [], []
    ends_in_dead = n_team_ends = 0
    restart_same = n_restarts = 0
    for labels in label_sequences:
        segs = _segments(labels)
        for i, (s, e, c, edge) in enumerate(segs):
            blocks = int(np.ceil((e - s) / STEP))
            if not edge:
                (dead_len if c == pm.D else team_len).append(blocks)
            if i + 1 < len(segs) and segs[i + 1][0] == e:
                nxt = segs[i + 1][2]
                if c in (pm.H, pm.A):
                    n_team_ends += 1
                    ends_in_dead += nxt == pm.D
                if c == pm.D and i > 0 and segs[i - 1][1] == s and segs[i - 1][2] in (pm.H, pm.A) \
                        and nxt in (pm.H, pm.A):
                    n_restarts += 1
                    restart_same += nxt == segs[i - 1][2]
    p_dead = (ends_in_dead + 1) / (n_team_ends + 2)
    p_same = (restart_same + 1) / (n_restarts + 2)
    return {"team_hazard": _hazard(team_len), "dead_hazard": _hazard(dead_len),
            "p_team_to_dead": p_dead, "p_restart_same": p_same,
            "n_team_spells": len(team_len), "n_dead_spells": len(dead_len),
            "n_restarts": n_restarts}


def _log_transitions(prior):
    """(4, 4) log P(next state | a spell of this state just ended). -inf where impossible."""
    t = np.full((4, 4), -np.inf)
    pd_, ps = prior["p_team_to_dead"], prior["p_restart_same"]
    t[SH, SA] = t[SA, SH] = np.log(1 - pd_)
    t[SH, SDH] = t[SA, SDA] = np.log(pd_)
    t[SDH, SH] = t[SDA, SA] = np.log(ps)
    t[SDH, SA] = t[SDA, SH] = np.log(1 - ps)
    return t


def _viterbi_blocks(em, prior):
    """Best hidden-state path for (n_blocks, 4) block emissions. Returns (n_blocks,) states."""
    n = len(em)
    log_h = np.stack([prior["team_hazard"][0]] * 2 + [prior["dead_hazard"][0]] * 2)    # (4, N_AGE)
    log_stay = np.stack([prior["team_hazard"][1]] * 2 + [prior["dead_hazard"][1]] * 2)
    trans = _log_transitions(prior)

    best = np.full((4, N_AGE), -np.inf)
    best[:, 0] = em[0] - np.log(4)
    from_state = np.zeros((n, 4), dtype=np.int8)     # for age 0: which state ended
    from_age = np.zeros((n, 4), dtype=np.int16)      #             at which age
    tail_stay = np.zeros((n, 4), dtype=bool)         # for the last age: stayed there (vs aged into it)
    for t in range(1, n):
        end = best + log_h                                   # a spell ends after this block
        end_age = end.argmax(axis=1)
        end_best = end[np.arange(4), end_age]                # (4,)
        cand = end_best[:, None] + trans                     # (from, to)
        src = cand.argmax(axis=0)
        new = np.empty_like(best)
        new[:, 0] = cand[src, np.arange(4)]
        from_state[t], from_age[t] = src, end_age[src]
        cont = best + log_stay                               # the spell goes on, one block older
        new[:, 1:] = cont[:, :-1]
        stay = cont[:, -1] > cont[:, -2]
        new[:, -1] = np.where(stay, cont[:, -1], cont[:, -2])
        tail_stay[t] = stay
        best = new + em[t][:, None]

    state = np.empty(n, dtype=np.int8)
    s, a = np.unravel_index(int(best.argmax()), best.shape)
    for t in range(n - 1, -1, -1):
        state[t] = s
        if t == 0:
            break
        if a == 0:
            s, a = int(from_state[t, s]), int(from_age[t, s])
        elif a == N_AGE - 1 and tail_stay[t, s]:
            pass
        else:
            a -= 1
    return state


def dead_insert_cost(prior, dead_frames, team_age_s=10.0, horizon_s=20.0):
    """Log-prob the prior charges for cutting a team spell aged team_age_s with a stoppage of
    dead_frames that the same team restarts, against the spell just going on (ages even out over
    horizon_s). The evidence a stoppage needs, in tempered units."""
    th, ts = prior["team_hazard"]
    dh, ds = prior["dead_hazard"]
    a, n, h = int(team_age_s * FPS / STEP), int(round(dead_frames / STEP)), int(horizon_s * FPS / STEP)
    age = lambda start, k: np.minimum(np.arange(start, start + k), N_AGE - 1)
    stay = ts[age(a, n + h)].sum()
    cut = (th[min(a, N_AGE - 1)] + np.log(prior["p_team_to_dead"]) + ds[age(0, n - 1)].sum()
           + dh[n - 1] + np.log(prior["p_restart_same"]) + ts[age(0, h)].sum())
    return float(stay - cut)


def tempered(emission, t_team, t_dead):
    """Split (n, 3) H / A / D log-emissions into alive-vs-dead and which-team parts, temper each.

    alive = log(e^H + e^A) is the log-probability of "alive"; H - alive and A - alive are the
    log-probabilities of each team given alive.
    """
    alive = np.logaddexp(emission[:, 0], emission[:, 1])
    out = np.empty_like(emission, dtype=float)
    out[:, 0] = alive / t_dead + (emission[:, 0] - alive) / t_team
    out[:, 1] = alive / t_dead + (emission[:, 1] - alive) / t_team
    out[:, 2] = emission[:, 2] / t_dead
    return out


