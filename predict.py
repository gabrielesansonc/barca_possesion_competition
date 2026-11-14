"""Make the submission file from the match data.

    python predict.py                       # writes submission/paloma.csv
    python predict.py --out my_file.csv

Steps: read the match data (and cache it in data/cache/), extract the audio with ffmpeg (once),
run the pipeline (possession/pipeline.py), write the CSV, check it with the organisers' validator
and print its MD5, which should match the one in README.md.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

import possession
from possession.data import load_match
from possession.pipeline import predict
from possession.utils import write_submission

TEAM_FILE = "paloma.csv"   # the form wants <team_name>.csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(possession.ROOT, "submission", TEAM_FILE))
    args = ap.parse_args()

    t = time.time()
    m = load_match()
    print("match loaded: {} frames ({:.0f}s)".format(m.n_frames, time.time() - t), flush=True)
    codes = predict(m)
    write_submission(args.out, codes)
    print("wrote {} ({:.0f}s)".format(args.out, time.time() - t), flush=True)

    validator = os.path.join(possession.COMPETITION, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, validator, args.out], check=True)
    print("MD5", hashlib.md5(open(args.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
