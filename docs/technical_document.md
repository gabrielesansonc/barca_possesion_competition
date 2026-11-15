# Who has the ball, frame by frame — team paloma

More than a Hack 2027, Phase 1 · Barcelona vs Athletic · code: github.com/gabrielesansonc/barca_possesion_competition

## 1. Summary

We label each of the 199,319 frames `H`, `A`, `D` or `X` with a pipeline of five LightGBM models
and a duration model of football, all trained on **10 other matches** of open data (3 Metrica
sample matches, 7 DFL Bundesliga matches). Nothing was trained or tuned on labels of this match.

Because the score counts whole possessions (spellF1), the pipeline does not stop at per-frame
predictions. It proposes candidate possessions and keeps the set most likely to match real ones.

| check (no labels of this match) | spellF1 |
|---|---|
| held-out Metrica matches (models trained without the match) | **0.738** |
| held-out DFL matches | **0.650** |
| public 5 minutes (17 possessions; a pipeline check only) | 0.759 |
| organisers' baseline, full match | 0.243 |

## 2. The approach, and why

**Framing.** We treat the task as tabular machine learning on tracking data. Features per frame
go into gradient-boosted trees, and a temporal decoder sits on top. Trees learn from a few
million rows in minutes on a laptop, and each step can be checked on its own.

**Training data from other matches.** The rules forbid labels of this match for training or
tuning. So we needed labelled matches whose tracking looks like this one:

- **Metrica sample data**: 3 matches, tracking + events. Our rules turn the events into per-frame
  labels (a team has the ball from its pass, carry, recovery, set piece or shot until the next
  one; the ball is dead from a ball out, foul or goal until the restart).
- **DFL open data**: 7 matches whose tracking carries the provider's per-frame flags `BallStatus`
  and `BallPossession`. These are the same kind of label as this challenge's ground truth.

**Degraded copies.** Open tracking is clean; this match's automatic tracking is not (the ball is
missing on 41% of graded frames, player tracks break, positions are noisy). Training on clean
data and predicting on noisy data fails. So every training match is copied with the same kinds
of damage, using only summary statistics of this match's tracking: ball gap lengths, track
lifetimes, tracks per frame. Each copy is also used with home and away swapped.

**The pipeline** ([possession/pipeline.py](../possession/pipeline.py)):

1. **Frame features** (~110 per frame, positions only): distances from each team to the ball,
   who controls it, ball speed, ball last seen outside the pitch, what was known at the edges of
   a ball gap, team shape and speed, who will reach the ball 0.5–1 s ahead, how tightly each team
   marks the other.
2. **Two heads** (LightGBM): P(ball dead) and P(home has it | ball in play).
3. **Match evidence**, added as log-odds towards `D`:
     - the first 2 s of a Smart Tagging set piece (a set piece is a restart, so the ball was dead);
     - frames no Smart Tagging phase covers;
     - **referee whistles** from the video's audio: the whistle at each tagged kick-off becomes a
       template of this referee's whistle, and later frames are scored by similarity to it.
       Its threshold gives as many whistles as a DFL match has whistled events (34).
4. **Switch head** (LightGBM): P(possession changes within 0.5 s), from features that compare the
   second before a frame with the second after.
5. **Candidate possessions.** A duration model learns from labels how long possessions and
   stoppages last and what follows what. It proposes the most likely sequence, 120 sampled
   sequences, and extra cuts at the switch head's peaks.
6. **Rescorer** (LightGBM): scores each candidate (how well it matches a real possession). The
   best non-overlapping set is kept, and the best sequence fills the gaps.
7. **Boundary model** (LightGBM): moves each change of possession by up to 1 s to where it most
   likely is.

**Why a rescorer and not just the most likely sequence.** spellF1 matches whole possessions one
to one. The most likely sequence under the duration model is over-confident and misses short
possessions. Its own samples agree with it (spellF1 0.96 between them) while it scores about 0.62
against the truth. A model trained to predict "will this candidate match a real possession",
on decodes of held-out matches, chooses better.

## 3. What each part added (held-out Metrica matches, spellF1)

| step added | held-out |
|---|---|
| per-frame heads on degraded Metrica copies, simple decoder | 0.622 |
| duration model of football as the decoder | 0.632 |
| DFL matches in training + candidate rescorer | 0.668 |
| richer rescorer features, wider pool of candidates | 0.670 |
| switch head, read by the rescorer | 0.687 |
| extra candidates cut at the switch head's peaks | 0.698 |
| interaction features in the heads + boundary model | 0.721 |
| control radius chosen on held-out matches (1.0 m instead of 1.5 m), all models retrained | 0.726 |
| dead / team heads trained for 1,500 trees instead of 300, rescorer and boundary model retrained | **0.738** |

Each step was accepted only if it improved on held-out matches. Settings were chosen on copies
with seeds 888/999 and confirmed on seed 777. The switch head and the last two steps each improved
every one of the 10 held-out matches. The last step was also confirmed on fresh copies (seed 555)
that no decision had used: +0.013 spellF1, all 10 matches better.

**How many trees in the heads.** The two heads used to stop at 300 boosting rounds (trees), a
default from early on. Watching the error on held-out matches as trees are added, the DFL matches
(whose labels are the same kind as this challenge's) kept improving until roughly 800–1,800
trees. At 1,500 the held-out score rose on every held-out match; at 4,000 the held-out error
started to rise again, so we stopped at 1,500. The switch head stays at 300: more trees made it
slightly better on its own but changed nothing after the rescorer.

## 4. Assets: used and dropped

| asset | use |
|---|---|
| tactical tracking (players + ball) | **used**: all frame features |
| Smart Tagging | **used**: set-piece starts and untagged gaps as dead-ball evidence; kick-offs to find the referee's whistle |
| tactical video | **used for its audio only** (referee whistles) |
| homography | not needed: features use pitch positions only |
| broadcast video + tracking | **tried, dropped**. We aligned it to the tactical video exactly (player residual 0.28 m), but it loses the ball in the same places (mostly stoppages), and replay / close-up cues gave no measurable gain |
| camera pan / zoom | **tried, dropped**: the camera follows the players, not the ball, so it adds nothing the tracking lacks |
| our own ball detection on the video | **not pursued**: a perfect in-play ball was worth ~+0.03 in simulation, and most ball gaps are in stoppages |

Also tried and dropped: decoding for expected spellF1 against the model's own samples (the
model is too sure of itself for this to help); a stillness rule (players slow for 5 s → dead
ball), neutral once the heads were trained on DFL; a model of which team restarts.

## 5. The possession rules we implemented

The models learn the definition from the training labels: the team named by the provider's flags
(DFL), or the team of the last possession event (Metrica). The explicit rules are:

- **Dead ball (`D`)**: ball out of play, fouls and offsides until the restart, set-piece setups,
  goals until the kick-off. The graded frames before the second-half kick-off (110,808–113,741,
  from the organisers' FAQ) are always `D`.
- **Loose balls and challenges**: no "nobody" label exists, so a loose ball stays with the last
  team in possession until the other team controls it. The boundary model learns where in the
  ball's flight the labels switch.
- **Restarts**: the duration model knows how often the same team restarts after a stoppage.
- **Short possessions**: allowed, but the duration model makes them rare unless evidence is
  strong. A wrong short possession costs as much as a missed one under spellF1.
- **`X`**: only on ungraded frames (before kick-off, half-time, after the final whistle).

## 6. Limitations, and what we would do with more time

- **Short possessions are under-predicted**: about 2.5 possessions per minute against roughly
  3.4 in the public window (4.6 in the DFL labels). Most misses last under 2 s: deflections and
  50/50 duels. Better ball tracking in crowded scenes would help most.
- **Stoppages before throw-ins and goal kicks** are sometimes run through as one possession, when
  the ball is not seen going out (about 5 of the ~58 tagged set pieces have no dead ball before them).
  The held-out matches cannot show this: they have no Smart Tagging and their stoppage ball is
  simulated.
- **Fouls and offsides** with the ball on the pitch are the hardest stoppages. Most referee
  whistles are not audible over the crowd (only ~10–14 of ~47 detectable).
- **Transition timing** has an unavoidable ±1 s uncertainty: the reference is set by an operator
  during the match.
- **Domain gap**: some ball-movement features differ between this match and the degraded
  training copies (this match's ball track is jerkier). We removed the 8 most affected
  interaction columns; a few base features remain affected.
- **Validation**: the public window has 17 possessions (a ±0.15 confidence interval), so it
  cannot choose between models. We relied on held-out matches and on a label-free occlusion
  test on this match (hide the ball where it is visible and check the answer holds).

## 7. Training data (required by the rules)

| data | matches | labels | who made the labels | used for |
|---|---|---|---|---|
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 | per-frame labels by our rules from Metrica's events | events by Metrica Sports (method not published); rules ours | training all models; the duration model's prior |
| DFL / Sportec Solutions open data (Bassek et al. 2025, CC BY 4.0) | 7 | the provider's `BallStatus` / `BallPossession` flags | DFL's data provider, during the match | training all models; the whistle count |
| Barcelona vs Athletic | — | **none** | — | only tracking statistics (gap lengths, track lifetimes) to make the copies look alike |

We labelled nothing by hand, on any match. The 5 public minutes were used only to check that the
pipeline runs correctly.

## 8. Reproduction

```bash
git clone --recurse-submodules https://github.com/gabrielesansonc/barca_possesion_competition.git
cd barca_possesion_competition
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the match data in data/ (data/README.md); ffmpeg must be installed
.venv/bin/python predict.py      # -> submission/paloma.csv, prints its MD5
```

`train.py` retrains every model from the open data (about an hour) and rebuilds `models/`
exactly. Details are in the repository's README and training/README.md.
