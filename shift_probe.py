import sys, csv, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from utils import preprocess
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

rows = list(csv.reader(open('data/mylap1/driving_log.csv')))
labels = np.array([float(r[3]) for r in rows])
imgs = [preprocess(np.asarray(Image.open(os.path.join('data/mylap1/IMG', os.path.basename(r[0].strip()))).convert('RGB'))) for r in rows]
n = len(rows)
random.seed(7); idx = list(range(n)); random.shuffle(idx)
test_idx = set(idx[int(n * 0.85):])

def build(shift):
    xs, ys, ids = [], [], []
    for t in range(n):
        lt = t + shift
        if 0 <= lt < n:
            xs.append(imgs[t]); ys.append(labels[lt]); ids.append(t)
    return np.stack(xs), np.array(ys, dtype=np.float32), ids

def arch():
    m = Sequential([Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=(66, 200, 3)),
                    Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
                    Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
                    Conv2D(64, (3, 3), activation='elu'), Conv2D(64, (3, 3), activation='elu'),
                    Flatten(), Dense(100, activation='elu'), Dense(50, activation='elu'),
                    Dense(10, activation='elu'), Dense(1)])
    m.compile(optimizer=Adam(1e-3), loss='mse')
    return m

out = open('shift_probe_results.txt', 'w')
for shift in (-4, -2, 0, 2, 4):
    xs, ys, ids = build(shift)
    tr = [i for i, t in enumerate(ids) if t not in test_idx]
    te = [i for i, t in enumerate(ids) if t in test_idx]
    m = arch()
    m.fit(xs[tr], ys[tr], epochs=15, batch_size=64, verbose=0)
    p = m.predict(xs[te], verbose=0).flatten()
    line = 'label shift %+d frames: held-out corr %.3f | pred std %.3f' % (
        shift, np.corrcoef(ys[te], p)[0, 1], p.std())
    print(line, flush=True)
    out.write(line + '\n'); out.flush()
out.write('DONE\n'); out.close()
