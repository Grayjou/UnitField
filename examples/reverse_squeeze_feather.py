"""
Reverse squeeze with feathering: compress a circular image horizontally and
feather the out-of-bounds region into the background.

Demonstrates:
  - Unit2DMappedEndomorphism (U2DE) with a custom coordinate map
  - remap() with BorderConfig feathering for smooth OOB blending
  - Pillow → numpy → Pillow round-trip
"""

from pathlib import Path

import numpy as np
from PIL import Image

from unitfield import U2DE, BorderConfig, BorderMode, InterpMethod, upbm_2d

parent_dir = Path(__file__).parent
(parent_dir / "output").mkdir(exist_ok=True)

def rgba_circle(size, color_circle, color_bg=(0, 0, 0, 0)):
    """Create a circular RGBA image."""
    img = Image.new("RGBA", size, color_bg)
    draw = ImageDraw.Draw(img)
    min(size) // 2
    draw.ellipse((0, 0, size[0], size[1]), fill=color_circle)
    return img


WIDTH, HEIGHT = 100, 100


def main():
    # --- create source image ---
    circle_img = rgba_circle(
        (WIDTH, HEIGHT),
        color_circle=(255, 0, 0, 255),
        color_bg=(255, 255, 255, 255),
    )
    src = np.asarray(circle_img, dtype=np.float64) / 255.0   # (H, W, 4) in [0,1]

    # --- build unit-space coordinate map ---
    upbm = upbm_2d(WIDTH, HEIGHT)                             # (H, W, 2), (x,y)
    squeezed = upbm.copy()
    x = squeezed[..., 0]                                   # x in [0, 1]
    x = 1 - x
    x *= 2.0
    squeezed[..., 0] = x                               # compress x → [0, 2]

    # --- create endomorphism ---
    endo = U2DE(squeezed, interp_method=InterpMethod.LINEAR)

    # --- remap with feathering ---
    bc = BorderConfig(
        mode=BorderMode.CONSTANT,
        constant_value=np.array([0.0, 0.0, 0.0, 0.0]),        # black fully transparent background
        feathering_width=0.15,
        feathering_x_undershoot_multiplier=1.0,
        feathering_x_overshoot_multiplier=1.0,
        feathering_y_undershoot_multiplier=0.0,  # only feather along x
        feathering_y_overshoot_multiplier=0.0,
        feather_dims=(False, False, False, True),  # only feather alpha channel
    )
    remapped = endo.remap(src, interpolation=1, border_config=bc)

    # --- save ---
    out = (np.clip(remapped, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(parent_dir / "output" / "reverse_squeeze_feather_out.png")
    print(f"Saved {parent_dir / 'output' / 'reverse_squeeze_feather_out.png'}")


if __name__ == "__main__":
    from PIL import ImageDraw
    main()
