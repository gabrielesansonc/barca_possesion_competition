"""Train every model in models/ from open data of other matches.

About 40 minutes on a laptop with 8 GB, plus downloading the open data the first time.

    python train.py                  # overwrites models/
    python train.py --out new_models # writes somewhere else, to compare with models/

Needs, besides the match data in data/ (only its tracking statistics are used, never labels):
  data/external/metrica_sample/   Metrica Sports sample data, cloned here on first run
  data/external/dfl/              DFL open data, downloaded and converted on first run
Intermediate tables are cached in data/training/ (about 3 GB), so a second run is faster.
See training/README.md for what each step learns and from which data.
"""

import argparse
import os
import subprocess

import possession
from possession import DATA_DIR

METRICA_REPO = "https://github.com/metrica-sports/sample-data"
METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"   # last commit, 2021-04-15


def get_metrica():
    path = os.path.join(DATA_DIR, "external", "metrica_sample")
    if not os.path.exists(os.path.join(path, "data", "Sample_Game_1")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["git", "clone", "-q", METRICA_REPO, path], check=True)
        subprocess.run(["git", "-C", path, "checkout", "-q", METRICA_COMMIT], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=possession.MODELS_DIR)
    args = ap.parse_args()
    get_metrica()
    from possession.training import train_all
    train_all(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
