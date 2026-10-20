"""Compare two trained models on the turning frames of a recording.

Purpose: answer "did v2 learn to steer where v1 stayed flat?" -- the frames
where the driver was actually turning are exactly the failure-spot skill v1
lacked (it wedged at the dirt corner by the lake and at the start bridge).

Usage:
    python compare_v1_v2.py --data data/mylap2 --v1 model.h5 --v2 model_v2.h5

Method: reuse train.py's loading (center camera, steering paired 2 frames
later), take center-camera frames with |steering| > 0.15, and for each model
report correlation vs recorded steering, prediction spread, and the 15 frames
where the two models disagree most. Honest caveat: v2 has usually *seen*
this recording (it trained on it), so its score is optimistic -- the point is
the behavioral comparison, not the leaderboard number.
"""

import argparse
import os

import numpy as np

from train import load_groups, load_image
from tensorflow import keras

TURN_THRESHOLD = 0.15
TOP_DISAGREE = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    args = ap.parse_args()

    print("loading models...")
    # compile=False: we only need predictions, and Keras 3 can't always
    # deserialize the legacy training config (loss/metrics) from .h5 saves.
    m1 = keras.models.load_model(args.v1, compile=False)
    m2 = keras.models.load_model(args.v2, compile=False)

    print("loading center-camera frames...")
    groups = load_groups(args.data)
    centers = [(p, s) for g in groups for p, s in [g[0]]
               if abs(s) > TURN_THRESHOLD]
    if not centers:
        raise SystemExit("No turning frames found. Lower the threshold?")
    print(f"{len(centers)} turning frames (|steering| > {TURN_THRESHOLD}) "
          f"across {', '.join(args.data)}")

    x = np.stack([load_image(p) for p, _ in centers])
    y = np.array([s for _, s in centers], dtype=np.float32)

    print("predicting (this takes a couple of minutes on CPU)...")
    p1 = m1.predict(x, verbose=0).flatten()
    p2 = m2.predict(x, verbose=0).flatten()

    c1 = float(np.corrcoef(y, p1)[0, 1])
    c2 = float(np.corrcoef(y, p2)[0, 1])
    print(f"\ncorrelation vs recorded steering on turning frames:")
    print(f"  v1: {c1:.3f}  (pred std {p1.std():.3f})")
    print(f"  v2: {c2:.3f}  (pred std {p2.std():.3f})")
    sign1 = float(np.mean(np.sign(p1) == np.sign(y)))
    sign2 = float(np.mean(np.sign(p2) == np.sign(y)))
    print(f"correct-direction rate on turning frames:")
    print(f"  v1: {sign1:.1%}   v2: {sign2:.1%}")

    disagree = np.abs(p1 - p2)
    top = np.argsort(disagree)[-TOP_DISAGREE:][::-1]
    print(f"\n{TOP_DISAGREE} frames where v1 and v2 disagree most "
          f"(recorded steering | v1 pred | v2 pred | image file):")
    for k in top:
        path, actual = centers[k]
        print(f"  {actual:+.2f} | {p1[k]:+.2f} | {p2[k]:+.2f} | "
              f"{os.path.basename(path)}")

    print("\nReading the numbers: what you want is v2's correlation and "
          "correct-direction rate clearly above v1's, with the top-"
          "disagreement list showing v2 matching the recorded steering "
          "where v1 sat near 0. If v2 still hugs 0 on real turns, the "
          "recovery data didn't take and the fix is more recovery reps.")


if __name__ == "__main__":
    main()
