from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Load PNG, resize to 80×80 if needed, convert to grayscale, return flat (6400,) float32."""
    img = Image.open(path).convert("RGB")
    if img.size != (80, 80):
        img = img.resize((80, 80), Image.BILINEAR)
    return np.array(img, dtype=np.float32).mean(axis=2).flatten()
