# Who has the ball? Frame-by-frame possession for Barcelona vs Athletic

Our entry to **More than a Hack 2027, Phase 1** (Metrica Sports × FC Barcelona).

For each of the 199,319 frames of the tactical video, we say who has the ball:
`H` (Barcelona), `A` (Athletic), `D` (dead ball) or `X` (not graded).

- **Input**: the organisers' tracking, Smart Tagging and the video's sound track.
- **Output**: one CSV, `frame,label`.
- **Every model was trained on other matches** (open data from Metrica and the DFL), never on this
  one. Nothing was trained or tuned on labels of this match. The 5 public minutes were only used
  to check that the pipeline works.

The method is described in [docs/technical_document.md](docs/technical_document.md).

---

## Reproduce the submission

You need Python 3.11, ffmpeg and the match data from the organisers.

```bash
# 1. Get the code (the organisers' repository comes in as a submodule)
git clone --recurse-submodules https://github.com/gabrielesansonc/barca_possesion_competition.git
cd barca_possesion_competition

# 2. Install the exact package versions we used
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Put the match data in data/ (see data/README.md for the file list)

# 4. Make the submission
.venv/bin/python predict.py
```

`predict.py` writes `submission/paloma.csv`, checks it with the organisers' validator and
prints its MD5. The file we submitted has MD5 **`688141c2bde5b9be562e87d3dd365ef1`**.

It takes under a minute on a laptop (Apple M-series, 8 GB). The first run also parses the
tracking into `data/cache/` and extracts the audio into `data/audio/` (we used ffmpeg 9.0.2).

**Retrain the models (optional).** `models/` holds the trained models. To rebuild them from the
open data (about an hour):

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python train.py        # downloads the open data, rewrites models/ with identical files
```

---

## What is in this repository

| path | what |
|---|---|
| [predict.py](predict.py) | the one command: data in, submission out |
| [possession/](possession/) | the method, as a Python package |
| [possession/pipeline.py](possession/pipeline.py) | **start here**: every step, in order |
| [models/](models/) | the trained models (LightGBM text files) and the decoder's prior |
| [train.py](train.py), [possession/training.py](possession/training.py) | retrain every model from the open data |
| [training/](training/) | what each model learned, and from which data |
| [docs/](docs/) | the technical document |
| [competition/](competition/) | the organisers' repository, unchanged (git submodule) |
| [data/](data/) | where the match data goes (not included: it is licensed to participants only) |

---

## How it works, in short

1. **Features per frame.** From player and ball positions: who is close to the ball, how fast
   players move, whether the ball was last seen outside the pitch, and so on.
2. **Two models per frame** (LightGBM): the chance the ball is dead, and the chance Barcelona has
   it if the ball is in play.
3. **Match evidence.** Smart Tagging set pieces and referee whistles heard in the audio push
   towards "dead ball".
4. **A switch model** (LightGBM): the chance possession changes within half a second.
5. **Candidate possessions.** A duration model (how long possessions and stoppages usually last)
   proposes many possible sequences of possessions.
6. **A rescorer** (LightGBM) scores each candidate possession and keeps the best set.
7. **A boundary model** (LightGBM) moves each change of possession to where it most likely
   happened.

All five LightGBM models learned from 10 matches of open data, made to look like this match's
automatic tracking (ball gaps, noise, track breaks).

---

## Results we can measure

The ranking uses the private full-match labels, which nobody has. These are our own checks:

| check | spellF1 |
|---|---|
| held-out Metrica matches (model trained without that match) | 0.738 |
| held-out DFL matches | 0.650 |
| the 5 public minutes (17 possessions: a sanity check only, never used to choose anything) | 0.759 |
| organisers' baseline, full match | 0.243 |

---

## Data and licences

- **Match data**: from the organisers, licensed to participants only. Not in this repository.
- **Metrica Sports sample data**: 3 matches,
  [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- **DFL / Sportec Solutions open data**: 7 Bundesliga matches, CC BY 4.0. Bassek et al., "An
  integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12:195
  (2025), doi:10.1038/s41597-025-04505-y.
- **Code**: MIT licence, see [LICENSE](LICENSE).
