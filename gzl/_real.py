# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""The rule that turns a lattice sum's complex arithmetic into its real value.

The INFINITE sum :math:`\zeta_G(\boldsymbol{k})` is real whenever three
things hold: the lattice has one site per cell, and the kernels are real
and even.  The inversion :math:`\boldsymbol{x} \to -\boldsymbol{x}` then
maps every configuration to its own conjugate while leaving the kernel
product alone, so the sum equals its conjugate.  ``Interaction`` enforces
the evenness, and a power law is even by construction.

A TRUNCATION need not inherit that.  The balanced label window is
inversion-closed at odd ``n``, and on an orthogonal cell at any ``n``,
but not at even ``n`` on a cell with a cross term -- the same asymmetry
:func:`gzl.slab.lattice_window_group` is built around.  So
:func:`graph_zeta_general` at finite momentum (``space="k"``) does carry
an imaginary part, measured up to 1.5e-2 of scale on a triangular cell
at even ``n``, and it is not this function's business.  What passes
through here is the value at :math:`\boldsymbol{k} = \boldsymbol{0}`,
where the phase is 1, and the box sums, whose weights are real cosines.

The engines work in complex arithmetic (the Fourier part, the FFTs, the
Epstein zeta), so the collapse has to happen somewhere.  :func:`_as_real`
is that place for the values this library returns, and it CHECKS rather
than assumes: an imaginary part above the round-off it expects means one
of the three hypotheses above is gone -- a basis of several sites per
cell, whose sublattice phases survive inversion, or a complex kernel such
as a Peierls phase -- and then the real part alone is not the answer, so
it raises instead of discarding the rest.
"""

from __future__ import annotations

import numpy as np

__all__: list[str] = []

#: An imaginary part above this fraction of the result's own scale is
#: not round-off.  Every value that reaches :func:`_as_real` today has
#: an imaginary part of exactly zero, so this bound is about telling a
#: genuinely complex value from noise, not about policing last digits.
_REAL_RTOL = 1e-8


def _as_real(value, *, where: str):
    """The real value of a lattice sum that must be real.

    ``value`` may be a scalar or an array, complex or already real.  A
    real input comes back as ``float`` / ``float64``, bit for bit, and a
    ``float64`` array is not copied.  A complex one is checked against
    ``_REAL_RTOL`` times the largest finite ``|Re|`` in the same result
    and then returned as its real part; above that bound, or if the
    imaginary part is NaN, it raises ``ValueError`` naming ``where``.

    A result whose real part is entirely zero (or not finite) has no
    scale of its own, and any imaginary part above the smallest normal
    double is then refused rather than guessed at.
    """
    arr = np.asarray(value)
    if not np.iscomplexobj(arr):
        return float(arr) if arr.ndim == 0 else np.asarray(arr, dtype=np.float64)

    im = np.abs(arr.imag)
    im_max = float(im.max()) if im.size else 0.0
    re = np.abs(arr.real)
    finite = re[np.isfinite(re)]
    scale = float(finite.max()) if finite.size else 0.0
    bound = _REAL_RTOL * max(scale, np.finfo(float).tiny)
    # ``not (<=)`` rather than ``>``: a NaN imaginary part must raise,
    # and every comparison against NaN is False.
    if not (im_max <= bound):
        raise ValueError(
            f"{where}: the value is not real -- max |Im| = {im_max:.3e} "
            f"against a scale of {scale:.3e}.  A graph zeta is real on a "
            "Bravais lattice with real, even kernels; a complex value means "
            "the kernel is not even or not real, or the cell carries more "
            "than one site.  Neither is supported, and the real part alone "
            "would not be the answer."
        )
    return (float(arr.real) if arr.ndim == 0
            else np.ascontiguousarray(arr.real, dtype=np.float64))
