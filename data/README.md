# Match data (not included)

The match data is licensed to enrolled participants only, so it is not in this repository.
Put the files from the organisers' download here:

```
data/
  tactical/
    BAR-ATH_tactical.mp4                 # the tactical video (used for its audio only)
    BAR-ATH_tactical_atd.txt             # tracking
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
```

The broadcast files are not used. To keep the data somewhere else, set `MTAH_DATA_DIR` to that
folder (the same variable the organisers' code uses).

`predict.py` writes two folders here on its first run: `cache/` (the tracking parsed into arrays)
and `audio/` (the video's sound track as a WAV file, made with ffmpeg).
