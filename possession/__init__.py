"""Frame-by-frame ball possession for More than a Hack 2027 (Barcelona vs Athletic).

Paths used by every module:
  ROOT         this repository
  COMPETITION  the organisers' repository (git submodule in competition/)
  DATA_DIR     the match data (data/ by default, or the folder in MTAH_DATA_DIR)
  MODELS_DIR   the trained models shipped with this repository (models/)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPETITION = os.path.join(ROOT, "competition")
os.environ.setdefault("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
DATA_DIR = os.environ["MTAH_DATA_DIR"]
MODELS_DIR = os.path.join(ROOT, "models")

# The organisers' helpers (games.py, starter/, evaluation/) are plain folders, not packages.
for _path in (COMPETITION, os.path.join(COMPETITION, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
