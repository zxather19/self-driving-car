"""Train the NVIDIA network on simulator recordings.

Data layout (put each recording in its own folder under data/):
    data/mylap1/
        driving_log.csv        <- center,left,right,steering,throttle,brake,speed
        IMG/                   <- the three camera pictures per row

Usage:
    python train.py --data data/mylap1 --epochs 30 --batch 128 --save model.h5
    python train.py --data data/mylap1 data/mylap2 --epochs 30

Design choices (v2 -- learned the hard way):
  * The split is by LOG ROW, so the three camera views of one moment are
    never divided between train and validation (that leaks and inflates scores).
  * The checkpoint metric is validation error on CENTER images only.
    Scoring against the side-camera labels (which carry a built-in +-0.2
    shift) rewards a model that hedges toward 0 and never really steers.
  * Augmentation: left/right cameras with a +-0.2 steering correction, plus
    random sideways flips done live per batch (never stored).
  * Everything is loaded into memory once (a recording is ~6k small images),
    which keeps batches honest and fast enough on CPU.
"""

import argparse
import csv
import os
import random

import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import Callback

from model import nvidia_model
from utils import preprocess, flip_augment


class CorrCheckpoint(Callback):
    """Save the model whenever validation CORRELATION improves.

    With sparse, noisy steering labels, the lowest-MSE snapshot is the model
    that hedges closest to always predicting 0 -- MSE checkpointing was
    actively selecting for collapse. Correlation measures what we actually
    care about: does the output move when the road does?
    """

    def __init__(self, x_val, y_val, save_path):
        super().__init__()
        self.x_val, self.y_val, self.save_path = x_val, y_val, save_path
        self.best = -2.0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x_val, verbose=0).flatten()
        corr = float(np.corrcoef(self.y_val, pred)[0, 1])
        if epoch % 5 == 4 or corr > self.best:
            print(f"  epoch {epoch + 1}: val corr {corr:.3f} "
                  f"(pred std {pred.std():.3f})")
        if corr > self.best:
            self.best = corr
            self.model.save(self.save_path)

STEERING_CORRECTION = 0.2
VAL_FRACTION = 0.15
# Pair each recorded moment with the steering from SHIFT_FRAMES rows LATER.
# A probe (shift_probe.py) tested -4..+4: no shift scored corr 0.30 on held-out
# images, +2 scored 0.56. Human steering is a *response* to the road ~160ms
# earlier -- at the recording's ~12fps, the label an image actually caused
# lands about 2 frames after it. This was the single biggest fix in the project.
SHIFT_FRAMES = 2
# Keyboard driving is mostly "no key pressed": in our first recording 84% of
# frames had steering exactly 0, and a network trained on that learns to just
# say 0. So keep every turning moment, but only this fraction of the
# straight-line moments (their side-camera pairs still teach recovery).
STRAIGHT_KEEP_FRACTION = 0.25


def load_groups(data_dirs):
    """-> list of moments; each moment = [(img_path, steering) x3 cameras].

    Steering for a moment comes from the log row SHIFT_FRAMES later
    (see the note at SHIFT_FRAMES). The last SHIFT_FRAMES rows are dropped.
    """
    groups = []
    for d in data_dirs:
        with open(os.path.join(d, "driving_log.csv"), newline="") as f:
            raw = list(csv.reader(f))

        def p(fname):
            # Log stores absolute paths from the recording machine;
            # resolve by filename inside this folder's IMG/.
            return os.path.join(d, "IMG", os.path.basename(fname.strip()))

        for i in range(len(raw) - SHIFT_FRAMES):
            center, left, right, *_ = raw[i]
            s = float(raw[i + SHIFT_FRAMES][3].strip() or 0.0)
            groups.append([(p(center), s),
                           (p(left), s + STEERING_CORRECTION),
                           (p(right), s - STEERING_CORRECTION)])
    return [g for g in groups if all(os.path.exists(p) for p, _ in g)]


def load_image(path):
    return preprocess(np.asarray(Image.open(path).convert("RGB")))


class BatchGen(Sequence):
    """In-memory batches with live flip augmentation."""

    def __init__(self, pairs, batch_size, augment):
        self.pairs = pairs  # list of (image_array, steering)
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        chunk = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        xs, ys = [], []
        for x, s in chunk:
            if self.augment and random.random() < 0.5:
                x, s = flip_augment(x, s)
            xs.append(x)
            ys.append(s)
        return np.stack(xs), np.array(ys, dtype=np.float32)

    def on_epoch_end(self):
        random.shuffle(self.pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save", default="model.h5")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="0.5 (the paper-era default) helped this sparse-label "
                         "data collapse to always predicting 0; 0 works better here")
    args = ap.parse_args()

    groups = load_groups(args.data)
    if not groups:
        raise SystemExit("No usable log rows found. Check --data folders "
                         "contain driving_log.csv + IMG/.")

    random.seed(42)
    random.shuffle(groups)
    cut = int(len(groups) * (1 - VAL_FRACTION))
    train_groups, val_groups = groups[:cut], groups[cut:]
    print(f"{len(groups)} recorded moments -> "
          f"{len(train_groups)} train / {len(val_groups)} validation")

    turning = [g for g in train_groups if abs(g[0][1]) > 0.02]
    straight = [g for g in train_groups if abs(g[0][1]) <= 0.02]
    keep_straight = random.sample(
        straight, int(len(straight) * STRAIGHT_KEEP_FRACTION))
    balanced = turning + keep_straight
    random.shuffle(balanced)
    print(f"balancing: kept all {len(turning)} turning moments + "
          f"{len(keep_straight)} of {len(straight)} straight moments")

    print("loading images into memory...")
    train_pairs = [(load_image(p), s) for g in balanced for p, s in g]
    # Validation = CENTER camera only, unaugmented: the honest score.
    val_pairs = [(load_image(g[0][0]), g[0][1]) for g in val_groups]
    x_val = np.stack([x for x, _ in val_pairs])
    y_val = np.array([s for _, s in val_pairs], dtype=np.float32)
    print(f"train pairs (3 cameras): {len(train_pairs)} | "
          f"validation center images: {len(val_pairs)}")

    model = nvidia_model(dropout=args.dropout, learning_rate=args.lr)
    model.summary()

    ckpt = CorrCheckpoint(x_val, y_val, args.save)
    model.fit(
        BatchGen(train_pairs, args.batch, augment=True),
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[ckpt],
    )
    print(f"best validation correlation: {ckpt.best:.3f} "
          f"(model from that epoch saved to {args.save})")
    print("\nThe loss is a guide, not the verdict: the verdict is the "
          "autonomous lap in the simulator.")


if __name__ == "__main__":
    main()
