# How the models were trained

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python train.py                  # rewrites models/ (about an hour)
.venv/bin/python train.py --out new_models # or write elsewhere and compare with models/
```

Training is deterministic: it rebuilds the files in [models/](../models/) exactly. The code is
[possession/training.py](../possession/training.py); its header lists the steps in order.

## Training data: 10 matches, none of them Barcelona vs Athletic

| source | matches | labels: where they come from | who made them | how much |
|---|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) (commit `e706dd5`) | 3 anonymised matches | turned into per-frame `H` / `A` / `D` by our code from Metrica's event data ([possession/ext_labels.py](../possession/ext_labels.py)): a team has the ball from its pass, carry, recovery, set piece or shot until the next one; the ball is dead from a ball out, foul or goal until the restart | events: Metrica Sports (how they were made is not published); per-frame labels: our rules | 3 full matches |
| [DFL / Sportec Solutions open data](https://doi.org/10.1038/s41597-025-04505-y) (CC BY 4.0, mirror `huggingface.co/datasets/pysport/idsse-data`, commit `a715a38`) | 7 Bundesliga matches | the flags `BallStatus` (ball alive / dead) and `BallPossession` (home / away) delivered with the tracking | DFL's data provider, set during the match (how is not published) | 7 full matches |

We did not label anything by hand, on any match.

From Barcelona vs Athletic, training uses only **summary statistics of its tracking**: how long
the ball goes missing, how long a player's track lasts before it breaks, how many tracks there
are per frame ([possession/degrade.py](../possession/degrade.py), `atd_stats`). These make the
training matches look like this match's automatic tracking. No labels of this match are used,
and neither are its 5 public minutes.

## What each model learns

| step | model | one row is | target |
|---|---|---|---|
| 1-2 | dead head, team head | a frame of a degraded copy (every 5th frame for Metrica, every 10th for DFL; 3 or 2 copies per match, each also with home and away swapped) | ball dead? / home team has it? |
| 3 | switch head | a frame of a degraded copy | does possession change within 0.5 s? |
| 4 | (no model) | each match is decoded by heads trained on the other 9 matches, on 6 fresh copies | |
| 5 | rescorer | a candidate possession in those decodes | how well it overlaps a real possession (share of the IoU thresholds 0.3-0.7 it passes) |
| 6 | boundary model | a change of possession in those decodes, chosen by a rescorer trained without that match | frames to the real change (clipped to ±1 s) |
| 7 | prior | the label sequences of the 3 Metrica matches | how long possessions and stoppages last |
| 8 | whistle count | the event lists of the 7 DFL matches | whistled events per match (34) |

**Degraded copies.** The training matches have clean tracking. This match's automatic tracking
loses the ball on 41% of frames, breaks player tracks and adds noise. So each training match is
copied with the same kinds of damage before features are computed, with fixed random seeds.

**Held-out decodes.** The rescorer and the boundary model learn from mistakes, so their
training data must contain realistic ones. Each match is therefore decoded only by models that
never saw it.

**Columns left out.** 8 interaction columns of the heads (listed in `training.SHIFTED`) are not
used: their distribution on this match is unlike any training match. Only the inputs were
compared, never labels.

## How we checked the models (no labels of this match)

Leave one match out: train on 9 matches, decode the 10th, score spellF1 with its labels, and
repeat for all 10. Settings were chosen on copies with seeds 888 and 999 and confirmed on seed
777. The results are in the technical document.
