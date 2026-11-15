# Trained models

Everything the pipeline loads. All of it was learned from 10 matches of open data (3 Metrica
sample matches, 7 DFL Bundesliga matches), never from Barcelona vs Athletic.

| file | what it predicts | trained on |
|---|---|---|
| `dead_head.txt` | per frame: the chance the ball is dead | sampled frames (every 5th–10th) of degraded copies of the 10 matches |
| `team_head.txt` | per frame: the chance the home team has the ball, if it is in play | the same frames, ball in play only |
| `switch_head.txt` | per frame: the chance possession changes within 0.5 s | sampled frames of degraded copies of the 10 matches |
| `rescorer.txt` | per candidate possession: how well it matches a real one | candidates from held-out decodes of the 10 matches |
| `boundary.txt` | per change of possession: how many frames to move it | held-out decodes of the 10 matches |
| `columns.json` | input columns of the switch head and the rescorer | |
| `prior.json` | how long possessions and stoppages last, and what follows what | labels of the 3 Metrica matches |

Trees per model: dead and team heads 1,500 each, switch head 300, rescorer 700, boundary model 300.

The `.txt` files are LightGBM models in its text format (LightGBM 4.5.0); open them with
`lightgbm.Booster(model_file=...)`. "Held-out decodes" means each match was decoded by models
trained without it, so the rescorer and boundary model learn from realistic mistakes.
