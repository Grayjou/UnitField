"""
Selective edge feathering with negative multiplier sentinel.

Demonstrates:
  - Setting a per-edge feather multiplier to a negative value (= sentinel)
    to disable feathering on that edge while keeping it on others.
  - CONSTANT border with feathering_width controls the blend region.
  - Visual comparison: the left edge stays at the sampled interior colour
    (sentinel), while the right edge blends smoothly to the border colour.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from unitfield import U2DE, BorderConfig, BorderMode, InterpMethod, upbm_2d

parent_dir = Path(__file__).parent
(parent_dir / "output").mkdir(exist_ok=True)

WIDTH, HEIGHT = 200, 100


def main():
    # --- source: vertical stripes (red on white) ---
    stripe = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    for i in range(4):
        x0 = i * WIDTH // 4
        x1 = (i + 1) * WIDTH // 4
        if i % 2 == 0:
            stripe[:, x0:x1] = [1.0, 0.0, 0.0]   # red
        else:
            stripe[:, x0:x1] = [1.0, 1.0, 1.0]   # white

    # --- coordinate map: identity stretched to [-0.5, 1.5] ---
    upbm = upbm_2d(WIDTH, HEIGHT)
    stretched = upbm.copy()
    x = stretched[..., 0]
    x = 2.0 * x - 0.5                                             # [0,1] → [-0.5, 1.5]
    stretched[..., 0] = x
    stretched[..., 1] = upbm[..., 1]                              # y stays identity [0,1]

    endo = U2DE(stretched, interp_method=InterpMethod.LINEAR)

    # --- left edge sentinel, right edge normal ---
    bc = BorderConfig(
        mode=BorderMode.CONSTANT,
        constant_value=np.array([0.0, 0.0, 1.0]),          # blue border
        feathering_width=0.3,
        feathering_x_undershoot_multiplier=-1.0,            # left:  sentinel → no feather
        feathering_x_overshoot_multiplier=1.0,              # right: normal feather
        feather_dims=(True, True, True),
    )

    remapped = endo.remap(stripe, interpolation=1, border_config=bc)
    out = (np.clip(remapped, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(out)

    # label regions with text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 2),  "sentinel", fill=(0, 0, 0), font=font)
    draw.text((150, 2), "normal",  fill=(0, 0, 0), font=font)

    out_path = parent_dir / "output" / "sentinel_demo.png"
    img.save(out_path)
    print(f"Saved {out_path}")
    print()
    print("Left side  (x<0, sentinel=-1):  stays red/white — no border blend")
    print("Right side (x>1, normal=1):     blends to blue border")


if __name__ == "__main__":
    main()
