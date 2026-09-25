"""Deterministic, CPU-only features for a 64x64 RGB satellite tile.

We are not trying to win a vision benchmark. These features are cheap,
fully offline, and good enough to prove the ingest -> classify -> store path
on EuroSAT-style tiles, where colour and coarse spatial layout already
separate most of the seven land-use classes.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.config import HIST_BINS, IMAGE_SIZE, SPATIAL_SIZE


def load_rgb_tile(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def extract_features(image: Image.Image) -> np.ndarray:
    """Turn a tile into a fixed-length vector.

    Layout:
    - 16-bin RGB histograms (colour signature)
    - 8x8 downsampled RGB (coarse spatial layout)
    - per-channel mean/std (brightness / contrast)
    """
    tile = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(tile, dtype=np.float32) / 255.0

    histograms = []
    stats = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=HIST_BINS, range=(0.0, 1.0), density=True)
        histograms.append(hist.astype(np.float32))
        stats.extend([float(arr[:, :, channel].mean()), float(arr[:, :, channel].std())])

    spatial = (
        np.asarray(tile.resize((SPATIAL_SIZE, SPATIAL_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    ).reshape(-1)

    return np.concatenate([*histograms, spatial, np.asarray(stats, dtype=np.float32)])


def features_from_bytes(data: bytes) -> np.ndarray:
    return extract_features(load_rgb_tile(data))
