"""Inspect original depth bytes without converting to RGB or overwriting metadata.

Run: python scripts/inspect_depth.py /path/to/one/depth.png
The report is diagnostic only. Decide a depth normalization AFTER inspecting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def inspect(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        original = np.asarray(image)
        data = original.astype(np.float64, copy=False)
        finite = data[np.isfinite(data)]
        return {
            "path": str(path),
            "format": image.format,
            "mode": image.mode,
            "size": list(image.size),
            "dtype": str(original.dtype),
            "shape": list(original.shape),
            "bands": list(image.getbands()),
            "finite_fraction": float(np.isfinite(data).mean()),
            "zero_fraction": float(np.mean(data == 0)),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "percentiles_1_50_99": [float(v) for v in np.percentile(finite, (1, 50, 99))] if finite.size else None,
            "unique_values": int(np.unique(original).size),
            "image_info_keys": sorted(image.info.keys()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_file", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.depth_file), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
