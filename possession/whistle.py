"""Referee whistles from the match audio, used as extra evidence that play has stopped.

How it works:
  1. Audio: the tactical video's sound track, mono 16 kHz (ffmpeg), +40 dB because it is very quiet.
  2. Spectrum: one spectrum per video frame (25 fps). Each frequency cell is compared with its usual
     level over the whole match ("whitening"), so a whistle stands out as a narrow, loud tone.
  3. Template: at each kick-off found by Smart Tagging, the strongest narrow tone above 2 kHz is the
     referee's whistle. Its average spectrum is this referee's "whistle template".
  4. Score: for every frame, how similar its spectrum is to the template (cosine similarity).
  5. Events: runs of frames above a threshold. The threshold is set so the match gets as many
     whistle events as a typical match has whistled events. That number (34) is the mean count of
     fouls, offsides, penalties, drop balls, kick-offs and final whistles per match in the DFL open
     data (7 Bundesliga matches); see training/README.md.
  6. Evidence: a whistle heard while the model thinks the ball is in play adds "dead ball" evidence
     to the next 3 seconds. Whistles heard during a stoppage (restarts) are ignored.

No constant here was chosen by looking at this match's audio.
"""

import os
import subprocess

import numpy as np
import pandas as pd

from possession import DATA_DIR

FPS = 25
GAIN_DB = 40                 # the match audio is very quiet (peaks at -43 dBFS)
NBINS, FMAX = 256, 8000      # spectrum: 0-8 kHz in 31.25 Hz cells (16 kHz audio)
BAND_HZ = (2000, FMAX - 100)  # a referee's whistle is a narrow tone in the kHz range
NEAR = np.r_[-12:-5, 6:13]   # cells 190-375 Hz either side of the peak: the "background"
FLOOR_DB = 20                # a kick-off whistle candidate stands at least this far above background
GAP = 2                      # frames bridged inside one whistle event
MIN_FRAMES, MAX_FRAMES = 4, 75  # a whistle lasts 0.16-3 s
TEMPLATE_FROM_HZ = 2500
TEMPLATE_SHIFT = 3           # cells (+-94 Hz) of pitch drift allowed
WHISTLES_PER_MATCH = 34      # DFL open data, see the module docstring
WHISTLE_SPAN = 3 * FPS       # frames of dead-ball evidence after a whistle
WHISTLE_ALIVE_BEFORE = 2 * FPS  # the model must lean "in play" over these frames before the whistle


# -- audio ------------------------------------------------------------------------------------------

def audio_path(game="BAR-ATH", sr=16000):
    """Mono 16-bit WAV of the tactical video's audio, extracted with ffmpeg on first use."""
    path = os.path.join(DATA_DIR, "audio", f"{game}_audio_{sr // 1000}k.wav")
    if not os.path.exists(path):
        video = os.path.join(DATA_DIR, "tactical", f"{game}_tactical.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sr),
                        "-af", f"volume={GAIN_DB}dB", "-c:a", "pcm_s16le", path], check=True)
    return path


def read_wav(path):
    """(16-bit samples, sample rate) of a mono WAV, memory-mapped."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    sr = int.from_bytes(head[24:28], "little")
    off = head.index(b"data") + 8
    return np.memmap(path, dtype="<i2", mode="r", offset=off), sr


# -- spectrum -------------------------------------------------------------------------------------

def spectrogram(x, sr, n_frames):
    """(n_frames, NBINS) dB, one spectrum per video frame (1024-sample Hann window centred on it)."""
    n_fft, hop = 1024, sr / FPS
    assert sr == 2 * FMAX and n_fft // 2 == 2 * NBINS, "expects 16 kHz audio"
    w = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_frames, NBINS), np.float32)
    for a in range(0, n_frames, 20000):
        fr = np.arange(a, min(n_frames, a + 20000))
        s0 = np.clip(np.round((fr + .5) * hop).astype(int) - n_fft // 2, 0, len(x) - n_fft)
        seg = x[s0[:, None] + np.arange(n_fft)].astype(np.float32) / 32768 * w
        p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        p = p[:, 1:n_fft // 2 + 1].reshape(len(fr), NBINS, 2).max(2)
        out[a:a + len(fr)] = 10 * np.log10(p * 4 / w.sum() ** 2 + 1e-30)
    return out


def whiten(spec):
    """Each frequency cell minus its median over the match."""
    return spec - np.median(spec[::5], axis=0)


def tone_scores(z, band=BAND_HZ, min_peak=0):
    """Per frame: how much the loudest cell in the band stands above the cells around it (dB)."""
    cell = (np.arange(NBINS) + .5) * FMAX / NBINS
    idx = np.flatnonzero((cell >= band[0]) & (cell <= band[1]))
    arg = idx[z[:, idx].argmax(1)]
    rows = np.arange(len(z))
    peak = z[rows, arg]
    near = z[rows[:, None], np.clip(arg[:, None] + NEAR, 0, NBINS - 1)].mean(1)
    return np.where(peak >= min_peak, peak - near, 0.0)


def events(score, threshold, gap=GAP, min_frames=MIN_FRAMES, max_frames=MAX_FRAMES):
    """DataFrame(start, end, peak) of runs where score >= threshold."""
    idx = np.flatnonzero(score >= threshold)
    if not len(idx):
        return pd.DataFrame(columns=["start", "end", "peak"])
    br = np.flatnonzero(np.diff(idx) - 1 > gap)
    s, e = idx[np.r_[0, br + 1]], idx[np.r_[br, len(idx) - 1]]
    keep = (e - s + 1 >= min_frames) & (e - s + 1 <= max_frames)
    s, e = s[keep], e[keep]
    peak = [score[a:b + 1].max() for a, b in zip(s, e)]
    return pd.DataFrame({"start": s, "end": e, "peak": peak})


def threshold_for(score, n_events, graded, **kw):
    """Threshold giving about n_events events starting in graded frames (bisection)."""
    lo, hi = float(np.min(score)), float(np.max(score))
    for _ in range(40):
        mid = (lo + hi) / 2
        ev = events(score, mid, **kw)
        k = graded[ev.start.to_numpy()].sum() if len(ev) else 0
        lo, hi = (mid, hi) if k > n_events else (lo, mid)
    return hi


# -- template -------------------------------------------------------------------------------------

def template_scores(z, tone, kickoffs):
    """Per frame: similarity to the referee's whistle, learned from the kick-off whistles."""
    ev = events(tone, FLOOR_DB)
    frames = []
    for k in kickoffs:
        near = ev[(ev.start >= k - 2 * FPS) & (ev.start <= k + 4 * FPS)]
        if len(near):
            pick = near.loc[near.peak.idxmax()]          # the strongest tone near the kick-off
            frames += range(int(pick.start), int(pick.end) + 1)
    assert frames, "no kick-off whistle found for the template"
    cell = (np.arange(NBINS) + .5) * FMAX / NBINS
    zp = np.clip(z[:, cell >= TEMPLATE_FROM_HZ], 0, None)
    t = zp[frames].mean(0)
    t /= np.linalg.norm(t)
    norm = np.linalg.norm(zp, axis=1) + 1e-6
    return np.max([np.roll(zp, s, axis=1) @ t / norm for s in range(-TEMPLATE_SHIFT, TEMPLATE_SHIFT + 1)], axis=0)


def whistle_scores(m, kickoffs, game="BAR-ATH"):
    """(n_frames,) template score, cached in data/cache/<game>_whistle.npz."""
    path = os.path.join(DATA_DIR, "cache", f"{game}_whistle.npz")
    if os.path.exists(path):
        return np.load(path)["template"]
    x, sr = read_wav(audio_path(game, 16000))
    z = whiten(spectrogram(x, sr, m.n_frames))
    score = template_scores(z, tone_scores(z), kickoffs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, template=score)
    return score


# -- evidence -------------------------------------------------------------------------------------

def stop_whistles(emission, m):
    """Start frames of whistles heard while the model leans "in play", graded frames only.
    emission must be the model's own (log-probabilities), before any other evidence is added."""
    tags = m.tagging
    kickoffs = tags[tags.type == "Kick Off (Start)"].start_frame.to_numpy() if "type" in tags else []
    if not len(kickoffs):
        return np.zeros(0, int)
    score = whistle_scores(m, kickoffs)
    assert len(score) == m.n_frames, "whistle cache is for another match length"
    p_dead = np.exp(emission[:, 2])
    threshold = threshold_for(score, WHISTLES_PER_MATCH, m.graded)
    out = []
    for t in events(score, threshold).start.to_numpy():
        before = p_dead[max(0, t - WHISTLE_ALIVE_BEFORE):t]
        if m.graded[t] and len(before) and before.mean() < 0.5:
            out.append(int(t))
    return np.array(out, int)


def add_whistle_evidence(emission, m, weight):
    """Emission plus `weight` log-odds towards D on the WHISTLE_SPAN frames after each stop whistle."""
    out = emission.copy()
    for t in stop_whistles(emission, m):
        out[t:t + WHISTLE_SPAN, 2] += weight
    return out
