"""Plain-language description and kind of every model input in the pipeline (dead / team heads, switch head,
rescorer, boundary model), from the code that builds them: possession.features, possession.interact,
possession.turnover, possession.decoder.spell_features, possession.rescorer, possession.pipeline.boundary_features.

    from glossary import describe
    kind, text = describe("ctrl_c3")

Every model matrix is float32; `kind` says what the number means (unit, flag, count, ...).
Coordinates are home-attack: +x always points at the away goal (possession.features.attack_sign).
"""

import re

BASE = {
    "h_n": ("count", "home players tracked in this frame"),
    "h_cx": ("m", "home centroid x (home-attack coordinates, 0 = halfway line)"),
    "h_spread": ("m", "home team spread: RMS distance of home players from their centroid"),
    "h_spread_x": ("m", "home spread along the pitch (std of x)"),
    "h_spread_y": ("m", "home spread across the pitch (std of y)"),
    "h_vfwd": ("m/s", "home players' mean forward velocity (towards the away goal)"),
    "h_speed": ("m/s", "home players' mean speed"),
    "h_line": ("m", "home defensive line: x of the second-deepest home player"),
    "a_line": ("m", "away defensive line: x of the second-deepest away player"),
    "cx_gap": ("m", "home centroid x minus away centroid x (who is pushed up)"),
    "speed_all": ("m/s", "mean speed of all tracked players (play stops → players walk)"),
    "frac_slow": ("share 0–1", "share of players moving slower than 1 m/s"),
    "speed_change_1s": ("m/s", "speed_all now minus its mean over the last 1 s (sudden slow-down)"),
    "ball_seen": ("flag 0/1", "1 if the ATD has a ball position in this frame"),
    "bx": ("m", "ball x (home-attack coordinates)"),
    "by_abs": ("m", "ball distance from the pitch's long axis"),
    "ball_speed": ("m/s", "ball speed, 5-frame smoothed"),
    "ball_out": ("m", "how far the ball is outside the pitch lines (0 inside)"),
    "d_h": ("m", "distance from the ball to the nearest home player (capped 30 m)"),
    "d_a": ("m", "distance from the ball to the nearest away player (capped 30 m)"),
    "d_margin": ("m", "d_a − d_h: > 0 means a home player is closer to the ball"),
    "ctrl": ("−1 / 0 / +1", "+1 a home player within 1.0 m and closest, −1 the same for away, 0 nobody"),
    "near_team": ("−1 / +1", "sign of d_margin: which team is nearest the ball"),
    "n_h_near": ("count", "home players within 5 m of the ball"),
    "n_a_near": ("count", "away players within 5 m of the ball"),
    "t_since_seen": ("s", "seconds since the ball was last tracked (capped 60)"),
    "t_until_seen": ("s", "seconds until the ball is tracked again (capped 60)"),
    "gap_len": ("s", "length of the current ball-tracking gap (0 when seen)"),
    "last_out": ("m", "ball_out at the last frame the ball was seen"),
    "last_bx": ("m", "ball x at the last frame the ball was seen"),
    "next_bx": ("m", "ball x at the next frame the ball is seen"),
    "last_ctrl": ("−1 / +1", "team of the last player in control (ctrl ≠ 0) before this frame"),
    "next_ctrl": ("−1 / +1", "team of the next player in control after this frame"),
    "t_since_ctrl": ("s", "seconds since a player last controlled the ball"),
    "t_until_ctrl": ("s", "seconds until a player next controls the ball"),
}
BASE.update({"a" + k[1:]: (v[0], v[1].replace("home", "away", 1)) for k, v in BASE.items()
             if k.startswith("h_") and k not in ("h_line",)})

IX = {
    "ix_proj05_dh": ("m", "distance from where the ball will be in 0.5 s (current velocity) to the nearest home player"),
    "ix_proj05_da": ("m", "same for the nearest away player"),
    "ix_proj05_margin": ("m", "ix_proj05_da − ix_proj05_dh: > 0 home will reach the ball first"),
    "ix_proj05_switch": ("flag 0/1", "1 if the team nearest the ball in 0.5 s differs from the team nearest now"),
    "ix_proj10_dh": ("m", "as ix_proj05_dh, 1 s ahead"),
    "ix_proj10_da": ("m", "as ix_proj05_da, 1 s ahead"),
    "ix_proj10_margin": ("m", "as ix_proj05_margin, 1 s ahead"),
    "ix_proj10_switch": ("flag 0/1", "as ix_proj05_switch, 1 s ahead"),
    "ix_mark_h": ("m", "home marking: mean distance of each home player to the nearest away player"),
    "ix_mark_a": ("m", "away marking: mean distance of each away player to the nearest home player"),
    "ix_mark_diff": ("m", "ix_mark_h − ix_mark_a"),
    "ix_trend_h_cx": ("m", "change in home centroid x over the last 2 s"),
    "ix_trend_a_cx": ("m", "change in away centroid x over the last 2 s"),
    "ix_trend_h_spread": ("m", "change in home spread over the last 2 s"),
    "ix_trend_a_spread": ("m", "change in away spread over the last 2 s"),
    "ix_trend_a_line": ("m", "change in the away defensive line over the last 2 s"),
}

TO = {
    "to_near_since": ("frames", "frames since the nearest team (sign of d_margin) last changed (capped 250)"),
    "to_near_until": ("frames", "frames until the nearest team next changes (capped 250)"),
    "to_ball_turn": ("rad", "largest change of ball direction within ±10 frames (ball > 2 m/s)"),
    "to_ball_dspeed": ("m/s", "largest change of ball speed within ±10 frames"),
    "to_free_len": ("frames", "length of the current free-flight run (ball > 2 m from every player)"),
    "to_free_max25": ("frames", "longest free-flight run within ±25 frames"),
}

RESCORER = {
    "log_dur": ("log frames", "log of the candidate's length"),
    "map_iou": ("IoU 0–1", "best IoU with a same-team possession of the Viterbi (MAP) timeline"),
    "map_other_iou": ("IoU 0–1", "best IoU with an other-team MAP possession (conflict)"),
    "from_peaks": ("flag 0/1", "1 if proposed only by the switch-head peaks, not by the MAP or any sample"),
    "argmax_team": ("share 0–1", "share of the candidate's frames whose emission argmax is its team"),
    "argmax_dead": ("share 0–1", "share of the candidate's frames whose emission argmax is D"),
    "entropy_in": ("nats", "mean entropy of the H/A/D emission inside the candidate"),
    "jump_start": ("prob diff", "P(team) in the first 10 frames minus the 10 frames before: a sharp start"),
    "jump_end": ("prob diff", "P(team) in the last 10 frames minus the 10 frames after: a sharp end"),
    "dead_jump_start": ("prob diff", "P(alive) jump at the start"),
    "dead_jump_end": ("prob diff", "P(alive) jump at the end"),
    "map_before": ("code 0–3", "MAP label just before the candidate: 0 same team, 1 other team, 2 D, 3 edge"),
    "map_after": ("code 0–3", "MAP label just after the candidate (same codes)"),
    "map_dist_start": ("log frames", "distance from the start to the nearest MAP boundary"),
    "map_dist_stop": ("log frames", "distance from the end to the nearest MAP boundary"),
    "since_dead": ("log frames", "frames since the last MAP stoppage ended"),
    "until_dead": ("log frames", "frames until the next MAP stoppage starts"),
    "map_contains": ("log count", "number of MAP boundaries strictly inside the candidate"),
    "sw_start": ("prob", "max p_switch within ±12 frames of the start"),
    "sw_start_wide": ("prob", "max p_switch within ±36 frames of the start"),
    "sw_start_at": ("prob", "p_switch exactly at the start frame"),
    "sw_end": ("prob", "max p_switch within ±12 frames of the end"),
    "sw_end_wide": ("prob", "max p_switch within ±36 frames of the end"),
    "sw_end_at": ("prob", "p_switch exactly at the end frame"),
    "sw_pre": ("prob", "max p_switch 75–12 frames before the start (a stronger switch just before?)"),
    "sw_post": ("prob", "max p_switch 12–75 frames after the end"),
    "sw_in_max": ("prob", "max p_switch inside the candidate (away from its ends): a switch inside is bad news"),
    "sw_in_mean": ("prob", "mean p_switch inside the candidate"),
    "sw_ends_min": ("prob", "min(sw_start, sw_end): both ends look like switches"),
    "sw_ends_vs_in": ("prob diff", "sw_ends_min − sw_in_max"),
}
WIN = {"in": "inside the candidate", "pre": "in the 50 frames before", "post": "in the 50 frames after",
       "head": "in its first 25 frames", "tail": "in its last 25 frames", "pre10": "in the 10 frames before",
       "post10": "in the 10 frames after", "head10": "in its first 10 frames", "tail10": "in its last 10 frames",
       "pre150": "in the 150 frames before", "post150": "in the 150 frames after"}

KINDS = "0 H>A, 1 A>H, 2 H>D, 3 A>D, 4 D>H, 5 D>A, 6 other"


def describe(c):
    """(kind, description) of a model input column."""
    if c in BASE:
        return BASE[c]
    if c in IX:
        return IX[c]
    if c in TO:
        return TO[c]
    if c in RESCORER:
        return RESCORER[c]
    m = re.fullmatch(r"(.+)_c(\d+)", c)
    if m and m.group(1) in BASE:
        k, d = BASE[m.group(1)]
        return k, "{} — mean over ±{} s".format(d, m.group(2))
    m = re.fullmatch(r"(.+)_(past|fut)3", c)
    if m and m.group(1) in BASE:
        k, d = BASE[m.group(1)]
        return k, "{} — mean over the {} 3 s".format(d, "last" if m.group(2) == "past" else "next")
    m = re.fullmatch(r"to_(.+?)_(d|flip|pre|post)(12|25)", c)
    if m:
        col, op, w = m.groups()
        base = BASE.get(col, ("", col))[1]
        return {"d": ("Δ", "change in {}: mean of the {} frames after minus the {} before".format(col, w, w)),
                "flip": ("flag 0/1", "1 if {} changes sign between the {} frames before and after".format(col, w)),
                "pre": (BASE.get(col, ("",))[0], "mean {} over the {} frames before ({})".format(col, w, base)),
                "post": (BASE.get(col, ("",))[0], "mean {} over the {} frames after".format(col, w))}[op]
    m = re.fullmatch(r"(p_team|p_dead|p_alive)_(\w+)", c)
    if m:
        what = {"p_team": "mean P(candidate's team | alive)", "p_dead": "mean P(D)", "p_alive": "mean P(alive)"}[m.group(1)]
        return "prob", "{} {}".format(what, WIN.get(m.group(2), m.group(2)))
    m = re.fullmatch(r"s(\d)_(q|iou_mean|p30|p70)", c)
    if m:
        k = int(m.group(1))
        sset = ["sample set 0 (40 posterior samples, no bias)", "sample set 1 (team-hazard bias 1.5)",
                "sample set 2 (team-hazard bias 3.0: shorter spells)"][k]
        return "share 0–1", {"q": "mean over IoU 0.3–0.7 of the share of samples containing it — " + sset,
                             "iou_mean": "mean best IoU with a same-team spell across " + sset,
                             "p30": "share of samples with a same-team spell at IoU ≥ 0.3 — " + sset,
                             "p70": "share of samples with a same-team spell at IoU ≥ 0.7 — " + sset}[m.group(2)]
    m = re.fullmatch(r"s(\d)_(same|other)_(max|n|better)", c)
    if m:
        who = "same-team" if m.group(2) == "same" else "other-team"
        return {"max": ("share 0–1", "best s{}_q among overlapping {} rival candidates".format(m.group(1), who)),
                "n": ("log count", "number of overlapping {} rivals".format(who)),
                "better": ("count", "{} rivals with a higher s{}_q".format(who, m.group(1)))}[m.group(3)]
    m = re.fullmatch(r"s(\d)_margin", c)
    if m:
        return "share diff", "s{0}_q minus the best overlapping rival's s{0}_q".format(m.group(1))
    m = re.fullmatch(r"lo_(other|dead)_(sum|mean|min5)", c)
    if m:
        vs = "the other team" if m.group(1) == "other" else "D"
        return "log-odds", {"sum": "signed log of the summed log-odds of the candidate's team vs " + vs,
                            "mean": "mean per-frame log-odds of the candidate's team vs " + vs,
                            "min5": "weakest 5-frame stretch of log-odds of the team vs " + vs}[m.group(2)]
    if c == "kind":
        return "category", "boundary kind: " + KINDS
    m = re.fullmatch(r"(dll|psw|pd|ph)_(-?\d+)", c)
    if m:
        o = int(m.group(2))
        at = "at the boundary" if o == 0 else "{} frames {} the boundary".format(abs(o), "before" if o < 0 else "after")
        return {"dll": ("log-odds", "emission log-odds of the new label vs the old one " + at),
                "psw": ("rank 0–1", "p_switch as its rank within the match " + at),
                "pd": ("prob", "P(D) from the dead head " + at),
                "ph": ("prob", "P(H | alive) from the team head " + at)}[m.group(1)]
    if c == "psw_peak_k":
        return "frames", "where p_switch peaks within ±25 frames of the boundary (−25…+25)"
    if c == "psw_peak_v":
        return "rank 0–1", "height of that peak (rank within the match)"
    if c in ("len_left", "len_right"):
        return "log frames", "length of the spell on the {} of the boundary".format(c.split("_")[1])
    return "", c
