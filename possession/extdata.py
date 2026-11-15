"""More external training matches: the DFL / Sportec open data (IDSSE), as possession.data.Match objects.

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data
in elite soccer", Scientific Data 12:195, doi:10.1038/s41597-025-04505-y. Data on figshare
(doi:10.6084/m9.figshare.28196177, licence **CC BY 4.0**, data owner Deutsche Fussball Liga), mirrored
at huggingface.co/datasets/pysport/idsse-data (the URLs kloppy's sportec.load_open_* use). Seven
full matches of the 2022/23 Bundesliga and 2. Bundesliga, TRACAB optical tracking at 25 fps.

Why these matter: every ball frame carries **BallStatus** (1 alive / 0 dead) and **BallPossession**
(1 home / 2 away), set by the data provider from its own (corrected) tracking. BAR-ATH's ground truth
is made the same way ("D where the corrected tracking marks the ball dead, otherwise the team the
tracking names as owner"), so these per-frame labels mean what the target means, unlike the
event-derived labels of the Metrica matches (possession.ext_labels).

    from possession.extdata import DFL_MATCHES, load_dfl
    m, labels, events = load_dfl("J03WPY")   # Match, (n,) pm codes from the tracking flags, events

Conversion (once per match, ~1 min, python -m possession.extdata): download the three XMLs to
data/external/dfl/raw/, parse the positions XML line by line (no DOM: 3.4M frame lines), write
data/external/dfl/<id>.npz (float32) + <id>_events.csv, then delete the 350-420 MB positions XML.

Layout of the Match, as possession.external.load_external:
  * metres, origin at a pitch corner, x along the length (DFL positions are centred: x + L/2)
  * one slot per person (fixed identity, NaN while not on the pitch), team HOME / AWAY
  * the two halves back to back with GAP_FRAMES of half 0 between them (NaN, label X), so no
    window, decoder run or spell crosses half time
  * the ball is tracked on every frame, dead time included (unlike Metrica's, blanked while dead)
"""

import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from possession import DATA_DIR
from possession.data import Match, HOME, AWAY
from possession.external import speed_from_positions
import possession_metrics as pm  # noqa: E402

# Params
DFL_DIR = os.path.join(DATA_DIR, "external", "dfl")
RAW_DIR = os.path.join(DFL_DIR, "raw")
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c/"  # pinned
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
               "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
               "J03WR9": "DFL-COM-000002"}
DFL_MATCHES = tuple(sorted(COMPETITION))
FPS = 25
GAP_FRAMES = 250          # ungraded frames inserted between the halves
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
FILES = {"meta": "DFL_02_01_matchinformation_{c}_DFL-MAT-{m}.xml",
         "events": "DFL_03_02_events_raw_{c}_DFL-MAT-{m}.xml",
         "positions": "DFL_04_03_positions_raw_observed_{c}_DFL-MAT-{m}.xml"}
SET_PIECES = ("KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty")


def _raw_path(match_id, kind):
    return os.path.join(RAW_DIR, FILES[kind].format(c=COMPETITION[match_id], m=match_id))


def download(match_id, kinds=("meta", "events", "positions")):
    os.makedirs(RAW_DIR, exist_ok=True)
    for kind in kinds:
        path = _raw_path(match_id, kind)
        if not os.path.exists(path):
            tmp = path + ".part"
            urllib.request.urlretrieve(BASE_URL + os.path.basename(path), tmp)
            os.replace(tmp, path)


# -- parsing -----------------------------------------------------------------------------------

def _meta(match_id):
    from lxml import etree
    root = etree.parse(_raw_path(match_id, "meta")).getroot()
    env = root.find(".//Environment")
    teams, persons = {}, {}
    for team in root.iter("Team"):
        code = HOME if team.get("Role") == "home" else AWAY
        teams[team.get("TeamId")] = code
        for p in team.iter("Player"):
            persons[p.get("PersonId")] = code
    return {"pitch": (float(env.get("PitchX")), float(env.get("PitchY"))), "teams": teams,
            "persons": persons}


def _ts(text):
    return datetime.fromisoformat(text).timestamp()


def _parse_positions(path):
    """{(section, team_id, person_id): (N, X, Y, Z, possession, status, first T, last T)} arrays.

    Each <Frame .../> line split on '"' puts the values at odd indices in a fixed attribute order
    (N, T, X, Y, [Z], D, S, A, M, [BallPossession, BallStatus]); the order is checked per FrameSet."""
    out = {}
    key, rows, is_ball = None, [], False

    def flush():
        if key is None or not rows:
            return
        tok0, tok1 = rows[0], rows[-1]
        n = np.array([int(r[1]) for r in rows], dtype=np.int64)
        x = np.array([float(r[5]) for r in rows], dtype=np.float32)
        y = np.array([float(r[7]) for r in rows], dtype=np.float32)
        if is_ball:
            z = np.array([float(r[9]) for r in rows], dtype=np.float32)
            possession = np.array([int(r[-4]) for r in rows], dtype=np.int8)
            stat = np.array([int(r[-2]) for r in rows], dtype=np.int8)
        else:
            z = possession = stat = None
        out[key] = (n, x, y, z, possession, stat, _ts(tok0[3].decode()), _ts(tok1[3].decode()))

    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"<Frame "):
                tok = line.split(b'"')
                if not rows:   # check the attribute order once per FrameSet
                    names = [t.strip().rstrip(b"=") for t in tok[0:-1:2]]
                    want = [b"<Frame N", b"T", b"X", b"Y"] + ([b"Z"] if is_ball else [])
                    if names[:len(want)] != want or (is_ball and names[-2:] != [b"BallPossession", b"BallStatus"]):
                        raise ValueError("unexpected Frame attributes {} in {}".format(names, key))
                rows.append(tok)
            elif line.startswith(b"<FrameSet"):
                flush()
                attrs = dict(zip([t.strip().rstrip(b"=").split()[-1].decode() for t in line.split(b'"')[0:-1:2]],
                                 [t.decode() for t in line.split(b'"')[1::2]]))
                key = (attrs["GameSection"], attrs["TeamId"], attrs["PersonId"])
                is_ball = attrs["TeamId"] == "BALL"
                rows = []
        flush()
    return out


def _parse_events(match_id, meta, half_t0, half_n, offsets):
    """Events as a DataFrame: frame (index into the Match), period, team H/A/'', type, subtype,
    x, y (metres, origin at a corner, event coordinates as delivered), decision_frame, extras."""
    from lxml import etree
    root = etree.parse(_raw_path(match_id, "events")).getroot()
    team_code = {tid: ("H" if c == HOME else "A") for tid, c in meta["teams"].items()}

    def to_frame(t):
        for period in (1, 2):
            k = int(round((t - half_t0[period]) * FPS))
            if -FPS * 60 <= k < half_n[period] + FPS * 60:
                return period, offsets[period] + int(np.clip(k, 0, half_n[period] - 1))
        return 0, -1

    rows = []
    for ev in root:
        if ev.tag != "Event" or not len(ev):
            continue
        kind = ev[0]
        t = _ts(ev.get("EventTime"))
        period, frame = to_frame(t)
        sub = kind[0] if len(kind) else None
        team = (kind.get("Team") or kind.get("WinnerTeam") or kind.get("TeamFouled")
                or (sub.get("Team") if sub is not None else None) or "")
        decision = kind.get("DecisionTimestamp")
        row = {"frame": frame, "period": period, "type": kind.tag,
               "subtype": sub.tag if sub is not None else "",
               "team": team_code.get(team, ""),
               "x": float(ev.get("X-Position", "nan")), "y": float(ev.get("Y-Position", "nan")),
               "decision_frame": to_frame(_ts(decision))[1] if decision else -1,
               "possession_change": kind.get("PossessionChange", ""),
               "winner_role": kind.get("WinnerRole", ""),
               "evaluation": kind.get("Evaluation") or (sub.get("Evaluation") if sub is not None else "") or "",
               "x_source": float(ev.get("X-Source-Position", "nan")),
               "y_source": float(ev.get("Y-Source-Position", "nan")),
               "fouler_team": team_code.get(kind.get("TeamFouler", ""), "")}
        if sub is not None and len(sub):
            row["subtype"] = sub.tag + "/" + sub[0].tag
        rows.append(row)
    ev = pd.DataFrame(rows)
    return ev[ev.frame >= 0].sort_values("frame", kind="stable", ignore_index=True)


def convert(match_id, delete_raw=True):
    """Download (if needed), parse and write data/external/dfl/<id>.npz and <id>_events.csv."""
    download(match_id)
    meta = _meta(match_id)
    sets = _parse_positions(_raw_path(match_id, "positions"))
    length, width = meta["pitch"]

    # frame range per half from the ball's FrameSet
    half_n0, half_n, half_t0 = {}, {}, {}
    for (section, team_id, _), (n, *_rest, t_first, t_last) in sets.items():
        if team_id == "BALL" and section in SECTIONS:
            p = SECTIONS[section]
            half_n0[p], half_n[p], half_t0[p] = int(n[0]), int(n[-1] - n[0] + 1), t_first
            if abs((t_last - t_first) - (n[-1] - n[0]) / FPS) > 0.5:
                raise ValueError("{} {}: frame times are not 25 fps".format(match_id, section))
    offsets = {1: 0, 2: half_n[1] + GAP_FRAMES}
    n_frames = offsets[2] + half_n[2]

    people = sorted({pid for (_, tid, pid) in sets if tid != "BALL" and tid in meta["teams"]})
    slot = {pid: i for i, pid in enumerate(people)}
    team = np.array([meta["persons"].get(pid, HOME) for pid in people], dtype=np.int8)
    players = np.full((n_frames, len(people), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_z = np.full(n_frames, np.nan, dtype=np.float32)
    possession = np.zeros(n_frames, dtype=np.int8)
    status = np.full(n_frames, -1, dtype=np.int8)
    half = np.zeros(n_frames, dtype=np.int8)
    for p in (1, 2):
        half[offsets[p]:offsets[p] + half_n[p]] = p
    for (section, team_id, pid), (n, x, y, z, possession, stat, *_t) in sets.items():
        if section not in SECTIONS:
            continue
        p = SECTIONS[section]
        idx = offsets[p] + (n - half_n0[p])
        ok = (idx >= offsets[p]) & (idx < offsets[p] + half_n[p])
        idx = idx[ok]
        xy = np.column_stack([x[ok] + length / 2, y[ok] + width / 2])
        if team_id == "BALL":
            ball[idx], ball_z[idx], possession[idx], status[idx] = xy, z[ok], possession[ok], stat[ok]
        elif pid in slot:
            if meta["teams"].get(team_id) != team[slot[pid]]:
                raise ValueError("{}: team of {} disagrees with the metadata".format(match_id, pid))
            players[idx, slot[pid]] = xy
    events = _parse_events(match_id, meta, half_t0, half_n, offsets)

    os.makedirs(DFL_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DFL_DIR, match_id + ".npz"), players=players, team=team,
                        ball=ball, ball_z=ball_z, possession=possession, status=status, half=half,
                        pitch=np.array([length, width]), people=np.array(people),
                        half_t0=np.array([half_t0[1], half_t0[2]]),
                        half_n=np.array([half_n[1], half_n[2]]),
                        offsets=np.array([offsets[1], offsets[2]]))
    events.to_csv(os.path.join(DFL_DIR, match_id + "_events.csv"), index=False)
    if delete_raw:
        os.remove(_raw_path(match_id, "positions"))
    return os.path.join(DFL_DIR, match_id + ".npz")


# -- loading -----------------------------------------------------------------------------------

def tracking_labels(possession, status, half):
    """(n,) pm codes from the provider's flags: D where BallStatus is 0, else H (BallPossession 1)
    or A (2); X outside the halves or where the ball frame is missing."""
    labels = np.full(len(status), pm.X, dtype=np.int8)
    labels[(status == 0)] = pm.D
    labels[(status == 1) & (possession == 1)] = pm.H
    labels[(status == 1) & (possession == 2)] = pm.A
    labels[half == 0] = pm.X
    return labels


def load_dfl(match_id):
    """(Match, labels from the tracking flags, events DataFrame) for one DFL match."""
    path = os.path.join(DFL_DIR, match_id + ".npz")
    if not os.path.exists(path):
        convert(match_id)
    z = np.load(path)
    players = z["players"]
    n = len(z["ball"])
    m = Match(players=players, player_speed=speed_from_positions(players), team=z["team"],
              ball=z["ball"], ball_speed=None, half=z["half"], homography=None, reproj_error=None,
              tagging=pd.DataFrame(),
              pitch=tuple(float(v) for v in z["pitch"]))
    events = pd.read_csv(os.path.join(DFL_DIR, match_id + "_events.csv"))
    text = events.select_dtypes(exclude="number").columns
    events[text] = events[text].fillna("").astype(str)
    return m, tracking_labels(z["possession"], z["status"], z["half"]), events


def ball_height(match_id):
    return np.load(os.path.join(DFL_DIR, match_id + ".npz"))["ball_z"]


if __name__ == "__main__":
    import sys
    import time
    for mid in (sys.argv[1:] or DFL_MATCHES):
        if os.path.exists(os.path.join(DFL_DIR, mid + ".npz")):
            print(mid, "already converted")
            continue
        t = time.time()
        print(mid, "->", convert(mid), "in {:.0f}s".format(time.time() - t), flush=True)


# -- event-derived labels (for comparison with the tracking flags; possession.ext_labels' rules) ------

TEAM_TYPES = {"Play", "ShotAtGoal", "BallClaiming", "OtherBallAction"} | set(SET_PIECES)
LAG_RANGE = 60            # frames searched each way when syncing events to the tracking
GOAL_DELAY = FPS          # a goal's ball is dead ~1 s after the shot


def event_lag(m, events):
    """{period: frames to add to event frames}: the shift that brings Play events closest to the
    tracked ball (median distance). DFL event times are hand-taken and off by up to ~1.5 s."""
    lags = {}
    for period in (1, 2):
        e = events[(events.type == "Play") & (events.period == period) & events.x.notna()]
        f, x, y = e.frame.values, e.x.values, e.y.values
        best = min(range(-LAG_RANGE, LAG_RANGE + 1), key=lambda lag: np.nanmedian(np.hypot(
            m.ball[np.clip(f + lag, 0, m.n_frames - 1), 0] - x,
            m.ball[np.clip(f + lag, 0, m.n_frames - 1), 1] - y)))
        lags[period] = best
    return lags


def event_labels_dfl(m, events, sync=True):
    """(n,) pm codes from the DFL events, by possession.ext_labels' rules translated to DFL's types.

    Team: from each Play / ShotAtGoal / BallClaiming / OtherBallAction / set piece, and each
    TacklingGame with PossessionChange, until the next one. Dead: from a foul, an offside, a set
    piece's DecisionTimestamp (the referee's call: ball out, foul) or 1 s after a goal, until the
    next team event (the restart). X before the first and after the last event of each half."""
    lags = event_lag(m, events) if sync else {1: 0, 2: 0}
    ev = events[events.period > 0].copy()
    shift = ev.period.map(lags).values
    ev["f"] = ev.frame.values + shift
    ev["df"] = np.where(ev.decision_frame >= 0, ev.decision_frame.values + shift, -1)
    team = ev[(ev.type.isin(TEAM_TYPES) | ((ev.type == "TacklingGame") & (ev.possession_change == "true")))
              & ev.team.isin(["H", "A"])]
    marks = [(int(f), pm.H if t == "H" else pm.A) for f, t in zip(team.f, team.team)]
    dead = list(ev.f[ev.type.isin(["Foul", "Offside"])])
    dead += list(ev.df[ev.type.isin(SET_PIECES) & (ev.df >= 0)])
    dead += list(ev.f[(ev.type == "ShotAtGoal") & ev.subtype.str.startswith("SuccessfulShot")] + GOAL_DELAY)
    marks += [(int(f), pm.D) for f in dead]
    marks.sort(key=lambda fm: (fm[0], fm[1] == pm.D))

    labels = np.full(m.n_frames, pm.X, dtype=np.int8)
    for period in (1, 2):
        idx = np.flatnonzero(m.half == period)
        lo, hi = idx[0], idx[-1] + 1
        pm_ = [(min(max(f, lo), hi - 1), c) for f, c in marks if lo - FPS <= f < hi + FPS]
        if not pm_:
            continue
        last = int(ev.f[ev.period == period].max())
        stop = min(hi, last + 1)
        for (f, c), (f_next, _) in zip(pm_, pm_[1:] + [(stop, None)]):
            labels[f:max(f, f_next)] = c
    return labels


# -- owner timing: move DFL's switches into the ball's flight (research agent's "mid50") --------

RELEASE_M = 1.0           # the old team "releases" the ball: last frame it was this close
LOOKBACK_S = 5.0
MIN_FLIGHT = 10           # frames from release to DFL's switch; shorter = a duel, left alone


def mid_flight_labels(m, labels, a=0.5):
    """Labels with each in-play H<->A switch moved from DFL's receiving contact (s) to
    r + a * (s - r), r = the old team's last frame within RELEASE_M of the ball in the LOOKBACK_S
    before s (inside the old spell). BAR-ATH's GT switches mid-flight (PLAN §9), DFL's on contact.
    Uses the clean DFL tracking; D frames and switches through D are unchanged."""
    dist = np.linalg.norm(m.players - m.ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_team = {pm.H: dist[:, m.team == HOME].min(axis=1), pm.A: dist[:, m.team == AWAY].min(axis=1)}
    out = labels.copy()
    segs = pm.segments(labels, pm.valid_runs(labels))
    moved = 0
    for (s0, s, c_old), (s1, _e1, c_new) in zip(segs, segs[1:]):
        if s != s1 or {c_old, c_new} != {pm.H, pm.A}:
            continue
        lo = max(s0, s - int(LOOKBACK_S * FPS))
        close = np.flatnonzero(d_team[c_old][lo:s] < RELEASE_M)
        if not len(close):
            continue
        r = lo + int(close[-1])
        if s - r >= MIN_FLIGHT:
            new_s = int(round(r + a * (s - r)))
            out[new_s:s] = c_new
            moved += 1
    return out
