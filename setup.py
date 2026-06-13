"""
Build script for UnitField Cython extensions.

Usage:
    pip install -e .            # Editable install (recommended for dev)
    pip install -e ".[dev]"     # Editable install with test dependencies
    python setup.py build_ext --inplace  # Build extensions in-place
"""

import os
import platform
from pathlib import Path

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# ---------------------------------------------------------------------------
# Platform-specific compiler flags
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    OPENMP_COMPILE = ["/openmp"]
    OPENMP_LINK: list[str] = []
else:
    OPENMP_COMPILE = ["-fopenmp"]
    OPENMP_LINK = ["-fopenmp"]

COMMON_COMPILE = ["/O2"] if IS_WINDOWS else ["-O3"]

# ---------------------------------------------------------------------------
# Extension discovery
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PKG = ROOT / "unitfield"

_SKIP_PYX = {
    "border_handling.pyx",
    "interp_utils.pyx",
    "clip_abs_utils.pyx",
}


def _should_compile(path: Path) -> bool:
    """Return True if a .pyx file should produce a compiled extension."""
    if path.name in _SKIP_PYX:
        return False
    if path.name == "__init__.pyx":
        return False
    if path.stat().st_size == 0:
        return False
    return True


def discover_pyx_files() -> list[Path]:
    """Find all .pyx files recursively under unitfield/ that need compilation."""
    return sorted(p for p in PKG.rglob("*.pyx") if _should_compile(p))


def make_extension(pyx_path: Path) -> Extension:
    """Create a setuptools Extension from a .pyx file path."""
    rel = pyx_path.relative_to(ROOT)
    module_name = str(rel.with_suffix("")).replace(os.sep, ".")
    source_rel = str(rel)

    source_text = pyx_path.read_text(encoding="utf-8", errors="ignore")
    uses_openmp = "prange" in source_text

    extra_compile = list(COMMON_COMPILE)
    extra_link: list[str] = []
    if uses_openmp:
        extra_compile += OPENMP_COMPILE
        extra_link += OPENMP_LINK

    return Extension(
        name=module_name,
        sources=[source_rel],
        include_dirs=[
            np.get_include(),
            str(ROOT),
        ],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )


pyx_files = discover_pyx_files()
extensions = [make_extension(p) for p in pyx_files]

if extensions:
    print(f"\n[unitfield] Discovered {len(extensions)} Cython extension(s):")
    for ext in extensions:
        print(f"  \xb7 {ext.name}")
    print()
else:
    print("\n[unitfield] No Cython extensions to compile.\n")


if __name__ == "__main__":
    setup(
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
                "initializedcheck": False,
            },
            nthreads=1,
        ),
    )
