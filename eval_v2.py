"""Head-to-head: v1 vs v2 on identical held-out frames.

Rebuilds the exact pooled split train.py used (same seed-42 shuffle of the
combined mylap1+mylap2 groups), takes the validation moments, and scores both
models on the SAME center images. Two views:
  * all validation moments (v2 saw none of these in training; v1 saw the
    mylap1 ones, so this view is generous to v1)
  * mylap2-only validation moments (frames NEITHER model trained on --
    the fair fight, and the data containing the recovery driving)
"""
import random

import numpy as np
from tensorflow.keras.models import load_model

from train import load_groups, load_image

random.seed(42)
groups = load_groups(["data/mylap1", "data/mylap2"])
random.shuffle(groups)
cut = int(len(groups) * 0.85)
val_groups = groups[cut:]

views = {
    "ALL val": val_groups,
    "mylap2-only val (fair)": [g for g in val_groups if "mylap2" in g[0][0]],
}

models = {}
for name, path in [("v1", "model.h5"), ("v2", "model_v2.h5")]:
    models[name] = load_model(path, compile=False)

for view_name, gs in views.items():
    xs = np.stack([load_image(g[0][0]) for g in gs])
    ys = np.array([g[0][1] for g in gs], dtype=np.float32)
    turn = np.abs(ys) > 0.1
    print(f"\n== {view_name}: {len(gs)} center frames "
          f"({int(turn.sum())} turning) ==")
    for name, m in models.items():
        pred = m.predict(xs, verbose=0).flatten()
        corr = float(np.corrcoef(ys, pred)[0, 1])
        mae = float(np.mean(np.abs(ys - pred)))
        sign = float(np.mean(np.sign(ys[turn]) == np.sign(pred[turn]))) if turn.sum() else float("nan")
        print(f"  {name}: corr {corr:.3f} | MAE {mae:.3f} | pred std {pred.std():.3f} "
              f"(actual {ys.std():.3f}) | turning sign agreement {sign*100:.0f}%")
