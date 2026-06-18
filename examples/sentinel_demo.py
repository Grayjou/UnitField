"""
Selective TO_BORDER feathering with a negative multiplier sentinel.

Demonstrates:
  - Identity coordinate maps.
  - FeatheringMode.TO_BORDER, where feathering happens inside the image.
  - A negative left-edge multiplier disables the left transition to border.
  - The right edge keeps normal feathering and fades to the border colour.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unitfield import (
    U2DE,
    BorderConfig,
    BorderMode,
    FeatheringMode,
    InterpMethod,
    upbm_2d,
)

parent_dir = Path(__file__).parent
(parent_dir / "output").mkdir(exist_ok=True)

WIDTH, HEIGHT = 240, 120


def main():
    # --- source: red/white vertical stripes ---
    src = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    for i in range(8):
        x0 = i * WIDTH // 8
        x1 = (i + 1) * WIDTH // 8
        src[:, x0:x1] = [1.0, 0.0, 0.0] if i % 2 == 0 else [1.0, 1.0, 1.0]

    # --- identity unit-space coordinate map ---
    identity = upbm_2d(WIDTH, HEIGHT)
    endo = U2DE(identity, interp_method=InterpMethod.LINEAR)

    # --- TO_BORDER: left disabled, right enabled ---
    bc = BorderConfig(
        mode=BorderMode.CONSTANT,
        constant_value=np.array([0.0, 0.0, 1.0]),       # blue border
        feathering_width=0.25,
        feathering_mode=FeatheringMode.TO_BORDER,
        feathering_x_undershoot_multiplier=-1.0,       # left: disabled/full internal
        feathering_x_overshoot_multiplier=1.0,         # right: fade to blue
        #feathering_y_undershoot_multiplier=-1.0,       # ignore top edge for this demo
        #feathering_y_overshoot_multiplier=-1.0,        # ignore bottom edge for this demo
        feather_dims=(True, True, True),
    )

    remapped = endo.remap(src, interpolation=1, border_config=bc)
    out = (np.clip(remapped, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(out)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 4), "left sentinel: internal", fill=(0, 0, 0), font=font)
    draw.text((WIDTH - 112, 4), "right: to border", fill=(0, 0, 0), font=font)

    out_path = parent_dir / "output" / "sentinel_demo.png"
    img.save(out_path)
    print(f"Saved {out_path}")
    print("Left edge: x undershoot multiplier = -1, so TO_BORDER contributes nothing.")
    print("Right edge: x overshoot multiplier = 1, so it fades to blue inside the image.")


if __name__ == "__main__":
    main()
