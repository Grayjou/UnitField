"""
Sinusoidal morphing example with feathering.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from unitfield import U2DE, BorderConfig, BorderMode, InterpMethod, upbm_2d

parent_dir = Path(__file__).parent
(parent_dir / "output").mkdir(exist_ok=True)

def rgba_centered_circle(size, color_circle, color_bg=(0, 0, 0, 0), circle_radius=None,
    smaller_circle_color=None, smaller_circle_radius_ratio=0.5):
    """Create a circular RGBA image."""
    img = Image.new("RGBA", size, color_bg)
    draw = ImageDraw.Draw(img)
    radius = circle_radius if circle_radius is not None else min(size) // 2
    center = (size[0] // 2, size[1] // 2)
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    #draw.ellipse((0, 0, size[0], size[1]), fill=color_bg)  # fill background
    draw.ellipse(bbox, fill=color_circle)  # draw centered circle
    if smaller_circle_color is not None:
        smaller_radius = int(radius * smaller_circle_radius_ratio)
        smaller_bbox = [center[0] - smaller_radius, center[1] - smaller_radius, center[0] + smaller_radius, center[1] + smaller_radius]
        draw.ellipse(smaller_bbox, fill=smaller_circle_color)  # draw smaller centered circle
    return img


WIDTH, HEIGHT = 100, 100


def main():
    # --- create source image ---
    circle_img = rgba_centered_circle(
        (WIDTH, HEIGHT),
        color_circle=(255, 0, 0, 255),
        color_bg=(255, 255, 255, 255),
        circle_radius=40,
        smaller_circle_color=(128, 0, 255, 255),
        smaller_circle_radius_ratio=0.5,
    )
    src = np.asarray(circle_img, dtype=np.float64) / 255.0   # (H, W, 4) in [0,1]

    # --- build unit-space coordinate map ---
    upbm = upbm_2d(WIDTH, HEIGHT)                             # (H, W, 2), (x,y)
    morphed = upbm.copy()
    x = morphed[..., 0]                                   # x in [0, 1]
    y = morphed[..., 1]                                   # y in [0, 1]

    SQUEEZE = 2.0
    x_min = (SQUEEZE - 1) / SQUEEZE / 2
    x_max = (SQUEEZE + 1) / SQUEEZE / 2
    if x_max - x_min < 1e-6:
        x_norm = np.full_like(x, 10.0)
    else:
        mx = 1/(x_max-x_min)
        cx = 1 - x_max*mx
        x_norm = mx*x + cx
    y_min = (SQUEEZE - 1) / SQUEEZE / 2
    y_max = (SQUEEZE + 1) / SQUEEZE / 2
    if y_max - y_min < 1e-6:
        y_norm = np.full_like(y, 10.0)
    else:
        my = 1/(y_max-y_min)
        cy = 1 - y_max*my
        y_norm = my*y + cy
    clipped_x_norm = np.clip(x_norm, 0.0, 1.0)

    x_norm += 0.10 * np.sin(2 * np.pi * y_norm * 3)*(2*np.sin(1*np.pi*clipped_x_norm))  # add sinusoidal morph along x based on y
    morphed[..., 0] = x_norm
    morphed[..., 1] = y_norm
    # --- create endomorphism ---
    endo = U2DE(morphed, interp_method=InterpMethod.LINEAR)
    bc = BorderConfig(
        mode=BorderMode.CONSTANT,
        constant_value=np.array([0.0, 0.0, 0.0, 0.0]),        # black fully transparent background
        feathering_width=0.15,
        feathering_x_multiplier=1.0,
        feathering_y_multiplier=1.0,
        feather_dims=(False, False, False, True),  # only feather alpha channel
    )
    remapped = endo.remap(src, interpolation=1, border_config=bc)
    remapped_img = Image.fromarray((remapped * 255).astype(np.uint8))
    remapped_img.save(parent_dir / "output" / "sinusoidal_morph_feather_out.png")
    print(f"Saved {parent_dir / 'output' / 'sinusoidal_morph_feather_out.png'}")

    # create side by side comparison image
    comparison_img = Image.new("RGBA", (WIDTH * 2, HEIGHT))
    comparison_img.paste(circle_img, (0, 0))
    comparison_img.paste(remapped_img, (WIDTH, 0))
    comparison_img.save(parent_dir / "output" / "sinusoidal_morph_feather_comparison.png")
    print(f"Saved {parent_dir / 'output' / 'sinusoidal_morph_feather_comparison.png'}")
if __name__ == "__main__":
    from PIL import ImageDraw
    main()
