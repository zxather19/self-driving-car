# Self-Driving Car — Behavioral Cloning

A virtual car that learns to steer by copying a human driver. Record laps in
Udacity's free simulator, train NVIDIA's end-to-end network
([Bojarski et al., 2016](https://arxiv.org/abs/1604.07316)) on
camera-image → steering-angle pairs, then let the network drive in Autonomous
mode. No lane labels, no hand-built lane detector — just imitation.

**Status: the car completes full laps autonomously** (at a 10 mph target
speed; see the results story below).

## Results

| model | training data | held-out steering correlation | autonomous lap |
| --- | --- | --- | --- |
| v1 | 1,997 moments, fast centre-line laps | 0.556 (own data) / 0.300 on recovery-style data | failed at ~15% of the lap (twice) |
| v2 | 3,897 moments, + deliberate edge-recovery laps | **0.565 on data neither model trained on** (sign agreement on turns 80% → 92%) | reached ~65%, beached on a curb in the paved lakeside section |
| v2 + fixes | + pavement steering gain (below), 10 mph | — | **full laps** |

## What made it work (the interesting part)

Every failure in this project was diagnosed from data, not vibes:

1. **84% of recorded frames have steering exactly 0** (keyboard driving is
   mostly "no key pressed"). A network trained on that learns to always say 0.
   Fix: keep all turning moments, only 25% of straight ones.
2. **MSE checkpointing selects for collapse** — the lowest-loss snapshot is
   the one that hedges closest to 0. Fix: checkpoint on validation
   *correlation* instead (custom callback in `train.py`).
3. **Human steering lags the road by ~2 frames.** Pairing each image with the
   steering from 2 frames *later* (≈160 ms at the recording's ~12 fps —
   reaction time) took held-out correlation from 0.30 to 0.56. The probe is
   reproducible: `shift_probe.py`, results in `shift_probe_results.txt`.
4. **The simulator never sends a socket.io CONNECT packet.** Its 2017 client
   opens the connection and streams telemetry in an old dialect, so any modern
   socket.io server accepts the connection and silently drops every frame.
   `drive.py` therefore speaks engine.io (the transport layer) directly and
   handles the packets itself — compatible with both dialects.
5. **The model understeers specifically on pavement.** Measured on held-out
   turning frames: the human steers |0.41|, the model asks for |0.28|
   (least-squares optimum ×1.31); on dirt its scale is already correct.
   `drive.py` detects pavement per frame (road-patch colour saturation) and
   applies the ×1.3 gain only there.
6. **Speed is reaction budget.** At 15 mph the model's decisions arrive too
   late for the tight paved bends; at 10 mph it laps cleanly.

## How it works

- `utils.py` — the *one* preprocessing function (crop → resize 66×200 →
  RGB→YUV), shared by training and driving so they can never drift apart.
- `model.py` — NVIDIA architecture: 5 conv layers → dense 100/50/10 → 1
  steering output, MSE loss, Adam (252,219 parameters).
- `train.py` — grouped train/val split by log row (a moment's 3 camera views
  never straddle the split), +2-frame label shift, balancing, left/right
  cameras with ±0.2 steering correction, live flip augmentation,
  correlation checkpointing.
- `drive.py` — engine.io server on port 4567; per frame: preprocess → predict
  → section-aware gain → steer; fixed-throttle speed control.
- `eval_v2.py` / `compare_v1_v2.py` — head-to-head model evaluation on
  identical held-out frames.

## Run it yourself

### 1. Simulator (Udacity Term 1 — accept no substitutes)

Download the **Term 1** Mac build (118 MB):
https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip

It is linked from the `udacity/self-driving-car-sim` repo README's
"Available Game Builds" section. **Warning:** the Releases page's "Latest" is
the *Term 2* simulator — a different app with an incompatible protocol. On
macOS, if the app "can't be opened": strip quarantine
(`xattr -dr com.apple.quarantine <app>`) and `chmod +x` the binary inside
`Contents/MacOS/` (this beta ships without the execute bit), then right-click
→ Open. Record with Graphics Quality on **Good/Fast** (stutter = jittery
training frames).

### 2. Record

Training mode → Record → a new folder per session, e.g. `data/mylap1/`
(produces `driving_log.csv` + `IMG/`). Drive smooth centre-line laps *and*
deliberate recovery laps — drift to the edge, steer back to centre. Recovery
data is what turns a demo into a driver.

### 3. Train

Python 3.10–3.12 (on Apple Silicon use `tensorflow-macos`; note
tensorflow-macos has no builds for Python 3.13+):

```bash
pip install -r requirements.txt
python train.py --data data/mylap1 data/mylap2 --epochs 40 --save model.h5
```

### 4. Drive

```bash
python drive.py model.h5 --speed 10
```

Then launch the simulator → **Autonomous** mode.

## Not in this repo

The recordings (`data/`, ~190 MB) and trained model binaries (`*.h5`) are
excluded — recordings are personal driving data and the models are
regenerated by `train.py`. The v1/v2 models from the results table were
trained on 2026-10-20 with the exact commands above.

## License

MIT — see `LICENSE`.
