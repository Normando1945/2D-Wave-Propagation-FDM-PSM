"""
wave_propagation_2d

Python package for 1D and 2D wave propagation simulations
using Finite Difference Method (FDM) and Pseudo-Spectral Method (PSM).
"""

from .core_ws1d_2d import (
    FFt_src,
    Fourier_derivate_n_order,
    animation1D,
    Safe_animation_1DW,
    animation2D_FDM,
    animation2D_PeudoSpectral,
)

__all__ = [
    "FFt_src",
    "Fourier_derivate_n_order",
    "animation1D",
    "Safe_animation_1DW",
    "animation2D_FDM",
    "animation2D_PeudoSpectral",
]

__version__ = "0.1.0"