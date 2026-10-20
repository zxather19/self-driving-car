"""Image preparation, shared by train.py and drive.py.

WHY THIS FILE EXISTS: the single most common behavioral-cloning failure is the
trainer and the driver preparing pictures differently. Both import preprocess()
from here, so they can never drift apart.
"""

import numpy as np
from PIL import Image

IMG_H, IMG_W = 66, 200  # NVIDIA's input size (rows, cols)
# Rows of the raw 160-tall frame to keep. The classic [60:135] road-only crop
# was tuned for non-letterboxed captures; these recordings ARE letterboxed
# (black bars top/bottom), and an A/B test showed the wider crop -- which keeps
# the horizon and scenery landmarks (lake side, hills) -- learns measurably
# better on this track. Only the car hood at the very bottom is excluded.
CROP_TOP, CROP_BOTTOM = 30, 140


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Raw simulator frame (160x320x3, uint8 RGB) -> model input (66x200x3, float32 in [0,1]).

    Steps, same as the NVIDIA paper:
    1. Crop the sky (top) and the car hood (bottom).
    2. Resize to 66x200.
    3. Convert RGB -> YUV (the network gets brightness separated from color).
    4. Scale pixels to 0..1.
    """
    img = img_rgb[CROP_TOP:CROP_BOTTOM, :, :]
    img = np.asarray(Image.fromarray(img).resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.float32)

    # RGB -> YUV (BT.601)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
    img = np.stack([y, u, v], axis=-1)
    return (img / 255.0).astype(np.float32)


def flip_augment(img_model_input: np.ndarray, steering: float):
    """Mirror an already-preprocessed image sideways; steering sign flips too."""
    return np.fliplr(img_model_input).copy(), -steering
