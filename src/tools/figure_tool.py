from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "1.0.0"


def run(image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image)
    nonwhite = np.any(array < 245, axis=2)
    colors = Counter(map(tuple, array[nonwhite].reshape(-1, 3).tolist()))
    dominant = [
        {"rgb": list(color), "pixel_count": count}
        for color, count in colors.most_common(12)
        if max(color) - min(color) > 15
    ][:8]
    # Connected dark-pixel bands provide conservative panel/axis structure only.
    dark = np.all(array < 100, axis=2)
    row_density = dark.mean(axis=1)
    col_density = dark.mean(axis=0)
    horizontal_bands = int(np.sum((row_density[1:] >= 0.25) & (row_density[:-1] < 0.25)))
    vertical_bands = int(np.sum((col_density[1:] >= 0.25) & (col_density[:-1] < 0.25)))
    return {
        "image_size": {"width": image.width, "height": image.height},
        "mode": "RGB",
        "nonwhite_pixel_fraction": float(nonwhite.mean()),
        "dominant_colored_regions": dominant,
        "ocr_text": [],
        "ocr_status": "unknown_unavailable",
        "axis_ranges": "unknown",
        "axis_ticks": "unknown",
        "legend_entries": "unknown",
        "series_count": "unknown",
        "bar_or_point_geometry": "unknown",
        "error_bars_present": "unknown",
        "panel_count": 2 if vertical_bands >= 2 else 1,
        "structural_bands": {"horizontal": horizontal_bands, "vertical": vertical_bands},
        "parse_confidence": 0.25,
        "limitations": ["No OCR engine is installed; semantic labels and exact geometry are not asserted."],
    }
