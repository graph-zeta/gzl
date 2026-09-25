# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

r"""Brute-force reference oracles for general per-edge kernels.

The library evaluates

.. math::

    \zeta_G(\boldsymbol{k}) \;=\;
    \sum_{\{\boldsymbol{x}_v \in \Lambda\},\, v \ne p}
    e^{-2\pi i (\boldsymbol{x}_t - \boldsymbol{x}_s) \cdot \boldsymbol{k}}
    \prod_{e \in E} V_e(\boldsymbol{x}_{e^+} - \boldsymbol{x}_{e^-})

on a Bravais lattice :math:`\Lambda = A \mathbb{Z}^d` with one vertex
pinned.  Historically every :math:`V_e` was the regularised power law
:math:`K_\nu(x) = |x|^{-\nu}`, :math:`K_\nu(0) = 0`.  The multi-kernel
feature widens this to

.. math::

    V_e(x) = a_e(x) + \sum_j b_j K_{\nu_j}(x)

with :math:`a_e` compactly supported, even, and — unlike the power law
— **allowed to be non-zero at the origin**, so a configuration whose
two endpoints coincide is weighted by :math:`a_e(0)` instead of being
dropped.

These oracles are deliberately dumb: an explicit enumeration of every
vertex assignment in a finite label box, with the per-edge weight
obtained by *calling the kernel function*.  Nothing here masks
coincident endpoints — returning ``0`` at ``dz = 0`` is the power-law
kernel's own job, and returning ``a(0)`` is the compact table's.  That
is precisely the property the library change must preserve, so the
oracle must not silently enforce it.

What each oracle is EXACT for
-----------------------------

``brute_force_zeta`` / ``box_enumeration`` evaluate the sum over the
finite label box :math:`[-L, L]^d` verbatim.  Therefore:

* For a **compactly supported** kernel (a pure table of reach ``R``)
  the box sum is *exact* — equal to the infinite-lattice value — as
  soon as ``L`` exceeds the reach that the graph can accumulate along
  a path from the pin.  The bound is graph- and pin-dependent: for a
  star around the pin ``L >= R`` suffices; for a path of ``h`` edges
  hanging off the pin the vertex at hop ``h`` needs ``L >= h·R``.  The
  tests that use the exactness say which bound they rely on.
* For a **power law** the box sum is truncation-limited: the residual
  falls off algebraically, leading order ``L^(d - nu_cut)`` where
  ``nu_cut`` is the minimum weighted cut separating an escaping vertex
  cluster from the pin.  Such a call is a *reference value with a
  systematic error*, never an exact oracle — choose ``L`` and ``nu``
  so the residual sits under the test tolerance, and say so.
* For a **mixed** kernel both statements apply term by term: the sum
  is truncation-limited by its power-law part alone.

``cycle_torus_reference`` is exact for the **torus truncation** it
names, and is *not* the infinite-lattice value (see its docstring).

Cost
----

``brute_force_zeta`` enumerates ``(2L+1)^(d·(V-1))`` configurations,
each carrying ``V-1`` label vectors of ``d`` int64 and one float64
accumulator, and calls every kernel function once on an array of that
length.  Peak memory is therefore
``(2L+1)^(d·(V-1)) · (8·d·(V-1) + O(1)·8)`` bytes; the guard refuses
more than ``_MAX_CONFIGS`` (5e6) configurations, which is ~240 MB of
labels at ``d·(V-1) = 6``.  Nothing in the implementation is
dimension-specific, so any ``d >= 1`` is accepted and the
configuration cap is what bites: ``d ∈ {1, 2, 3}`` is the practical
range for multi-vertex graphs, while ``d = 4`` still reaches the
2-path at ``L = 1`` (the case ``tests/test_box_nu_inf.py`` uses to
gate the box engine's own d-guard lift).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


__all__ = [
    "brute_force_zeta",
    "box_enumeration",
    "power_law_kernel",
    "table_kernel",
    "mixed_kernel",
    "cycle_torus_reference",
]


# Memory guard, matching ``tests.test_finite_k_validation._direct_sum_finite_k``.
_MAX_CONFIGS = 5_000_000


KernelFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def power_law_kernel(nu: float, A: np.ndarray) -> KernelFn:
    r"""The regularised power law :math:`K_\nu(A\,dz) = |A\,dz|^{-\nu}`
    as a function of **integer label differences**.

    The returned callable maps an integer array of shape ``(..., d)``
    of label differences to a float array of shape ``(...)``.  It
    returns exactly ``0.0`` at ``dz = 0`` — the kernel's regularisation
    at the origin, which is what drops coincident-endpoint
    configurations.  No other value is special-cased.

    ``A`` fixes the metric: the physical displacement of the label
    difference ``dz`` is ``A @ dz``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"power_law_kernel: A must be square, got {A.shape}.")
    nu = float(nu)

    def kernel(dz: np.ndarray) -> np.ndarray:
        dz = np.asarray(dz)
        if dz.shape[-1] != A.shape[0]:
            raise ValueError(
                f"power_law_kernel: label difference has trailing dim "
                f"{dz.shape[-1]}, expected d = {A.shape[0]}."
            )
        r = np.linalg.norm(dz.astype(float) @ A.T, axis=-1)
        zero = r == 0.0
        # Evaluate on a safe operand, then impose K(0) = 0 exactly.
        return np.where(zero, 0.0, np.where(zero, 1.0, r) ** (-nu))

    return kernel


def table_kernel(table: Mapping[tuple[int, ...], float]) -> KernelFn:
    r"""A compactly supported kernel given by an explicit table.

    ``table`` maps an integer label difference (a tuple of length
    ``d``) to its weight; every label difference absent from the table
    has weight ``0``.  ``table[(0,) * d]`` — the value at the origin —
    is honoured like any other entry: this is the ``a_e(0)`` of the
    multi-kernel feature, and a configuration with coincident
    endpoints is weighted by it rather than dropped.

    Evenness of ``a`` is *not* enforced here; the oracle evaluates the
    table it is handed, so a test that relies on ``a(-x) = a(x)`` must
    supply an even table itself.

    Cost: one boolean mask over the configuration axis per table entry,
    so a table of ``T`` entries costs ``T`` passes over the enumeration.
    """
    if not table:
        raise ValueError("table_kernel: empty table (use a zero entry).")
    keys = [tuple(int(c) for c in k) for k in table.keys()]
    dims = {len(k) for k in keys}
    if len(dims) != 1:
        raise ValueError(
            f"table_kernel: keys must all have the same length, got {sorted(dims)}."
        )
    d = dims.pop()
    key_arr = np.array(keys, dtype=np.int64)              # (T, d)
    val_arr = np.array([float(v) for v in table.values()])  # (T,)

    def kernel(dz: np.ndarray) -> np.ndarray:
        dz = np.asarray(dz)
        if dz.shape[-1] != d:
            raise ValueError(
                f"table_kernel: label difference has trailing dim "
                f"{dz.shape[-1]}, expected d = {d}."
            )
        dz = dz.astype(np.int64, copy=False)
        out = np.zeros(dz.shape[:-1], dtype=float)
        for j in range(key_arr.shape[0]):
            mask = np.all(dz == key_arr[j], axis=-1)
            out[mask] = val_arr[j]
        return out

    return kernel


def mixed_kernel(
    table: Mapping[tuple[int, ...], float],
    b: Sequence[float],
    nu: Sequence[float],
    A: np.ndarray,
) -> KernelFn:
    r"""``table_kernel(table) + Σ_j b[j] · power_law_kernel(nu[j], A)``.

    The full multi-kernel shape ``V(x) = a(x) + Σ_j b_j K_{ν_j}(x)``.
    Because every power law vanishes at the origin, ``V(0) = a(0)``
    exactly — the compact part alone decides the weight of a
    coincident-endpoint configuration.
    """
    b = [float(x) for x in b]
    nu = [float(x) for x in nu]
    if len(b) != len(nu):
        raise ValueError(
            f"mixed_kernel: len(b) = {len(b)} != len(nu) = {len(nu)}."
        )
    a_fn = table_kernel(table)
    pl_fns = [power_law_kernel(nu_j, A) for nu_j in nu]

    def kernel(dz: np.ndarray) -> np.ndarray:
        out = a_fn(dz)
        for b_j, fn in zip(b, pl_fns):
            out = out + b_j * fn(dz)
        return out

    return kernel


# ---------------------------------------------------------------------------
# The enumeration
# ---------------------------------------------------------------------------

def _free_labels(n_free: int, d: int, L: int, n_configs: int) -> list[np.ndarray]:
    """The ``n_free`` label arrays of shape ``(n_configs, d)`` enumerating
    ``[-L, L]^(d·n_free)`` in C-order.

    Digit ``j`` of the mixed-radix index (``j = 0`` most significant)
    is decoded directly rather than materialised by ``np.meshgrid``,
    which halves the peak memory of the enumeration.  The ordering is
    the same as ``np.meshgrid(..., indexing='ij')`` followed by
    ``ravel()``; a sum does not care, but keeping the convention makes
    the two oracles trivially comparable term by term.
    """
    coords = np.arange(-L, L + 1, dtype=np.int64)
    M = 2 * L + 1
    K = n_free * d
    idx = np.arange(n_configs, dtype=np.int64)
    digits = []
    for j in range(K):
        stride = M ** (K - 1 - j)
        digits.append(coords[(idx // stride) % M])
    return [
        np.stack(digits[f * d:(f + 1) * d], axis=-1) for f in range(n_free)
    ]


def brute_force_zeta(
    edges,
    kernel_fns: Sequence[KernelFn],
    A,
    L: int,
    *,
    source: int = 0,
    terminal: int | None = None,
    k_frac=None,
) -> float:
    r"""Brute-force finite-box lattice sum with **arbitrary per-edge kernels**.

    .. math::

        \zeta_G(\boldsymbol{k}) \;\approx\;
        \sum_{\boldsymbol{m}^{(v)} \in [-L, L]^d,\; v \ne s}
        e^{-2\pi i\, \boldsymbol{k} \cdot
           (\boldsymbol{m}^{(t)} - \boldsymbol{m}^{(s)})}
        \prod_{e \in E} V_e\!\left(
            \boldsymbol{m}^{(e^+)} - \boldsymbol{m}^{(e^-)}\right)

    with ``source`` pinned at label ``0`` and every other vertex summed
    over the integer label box.  Returns the real part, as the library
    does (lattice inversion symmetry makes the imaginary part vanish up
    to the truncation).

    This generalises ``tests.test_finite_k_validation._direct_sum_finite_k``:
    with ``kernel_fns = [power_law_kernel(nu_i, A) for nu_i in nu_vec]``
    the two agree to round-off.

    Parameters
    ----------
    edges
        ``(E, 2)`` integer array.  Edge ``i`` is ``(u, v)`` and its
        kernel argument is the label difference ``m_v - m_u`` — the
        same orientation ``_direct_sum_finite_k`` uses.  Kernels of the
        feature are even, so the orientation is immaterial for them,
        but the oracle is explicit about it so an *odd* kernel could be
        checked too.
    kernel_fns
        One callable per edge, mapping an integer array of label
        differences of shape ``(n_configs, d)`` to a float array of
        shape ``(n_configs,)``.  **The oracle does not mask coincident
        endpoints**: whatever the kernel returns at ``dz = 0`` is the
        weight, which is exactly how ``a_e(0) != 0`` enters.
    A
        ``(d, d)`` lattice matrix.  Used here only to fix ``d`` — the
        metric enters through the kernel functions, which is what lets
        a table kernel be metric-free.
    L
        Half-width of the label box, in lattice labels.
    source
        The pinned vertex, held at label ``0``.  On a *finite* box the
        pin is value-relevant (the box is not translation-closed), so
        the caller must choose it deliberately; the infinite-lattice
        value is pin-independent.
    terminal, k_frac
        Give both to weight each configuration by
        ``exp(-2πi k_frac · (m_t - m_s))`` with ``k_frac`` in
        fractional Brillouin-zone coordinates — so the phase uses the
        integer labels directly, with no metric.  Give neither for the
        vacuum sum.

    Notes
    -----
    Exactness and cost: see the module docstring.
    """
    edges = np.atleast_2d(np.asarray(edges, dtype=int))
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"brute_force_zeta: edges must have shape (E, 2), got {edges.shape}."
        )
    kernel_fns = list(kernel_fns)
    if len(kernel_fns) != edges.shape[0]:
        raise ValueError(
            f"brute_force_zeta: {len(kernel_fns)} kernel functions for "
            f"{edges.shape[0]} edges."
        )
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"brute_force_zeta: A must be square, got {A.shape}.")
    d = int(A.shape[0])
    if d < 1:
        raise ValueError(f"brute_force_zeta: d = {d} is not a lattice.")
    L = int(L)
    if L < 0:
        raise ValueError(f"brute_force_zeta: L must be >= 0, got {L}.")

    V = int(edges.max()) + 1
    if not (0 <= source < V):
        raise ValueError(
            f"brute_force_zeta: source {source} outside the vertex range "
            f"[0, {V})."
        )
    if (terminal is None) != (k_frac is None):
        raise ValueError(
            "brute_force_zeta: give both `terminal` and `k_frac`, or neither."
        )
    if terminal is not None and not (0 <= terminal < V):
        raise ValueError(
            f"brute_force_zeta: terminal {terminal} outside the vertex range "
            f"[0, {V})."
        )

    free = [v for v in range(V) if v != source]
    n_free = len(free)
    M = 2 * L + 1
    n_configs = M ** (n_free * d)
    if n_configs > _MAX_CONFIGS:
        raise ValueError(
            f"brute_force_zeta: {n_configs:.2e} configurations "
            f"((2L+1)^(d(V-1)) with L = {L}, d = {d}, V = {V}) would blow "
            f"memory; lower V, d, or L."
        )

    lab = {source: np.zeros((n_configs, d), dtype=np.int64)}
    for fi, arr in enumerate(_free_labels(n_free, d, L, n_configs)):
        lab[free[fi]] = arr

    weight = np.ones(n_configs, dtype=float)
    for i, (u, v) in enumerate(edges):
        dz = lab[int(v)] - lab[int(u)]
        w_e = np.asarray(kernel_fns[i](dz), dtype=float)
        if w_e.shape != (n_configs,):
            raise ValueError(
                f"brute_force_zeta: kernel {i} returned shape {w_e.shape}, "
                f"expected ({n_configs},)."
            )
        weight *= w_e

    if terminal is None:
        return float(weight.sum())

    k_arr = np.asarray(k_frac, dtype=float).reshape(d)
    dm = (lab[int(terminal)] - lab[int(source)]).astype(float)
    phase = np.exp(-2j * np.pi * (dm @ k_arr))
    return float((weight * phase).sum().real)


def box_enumeration(
    edges,
    kernel_fns: Sequence[KernelFn],
    A,
    L: int,
    *,
    root: int = 0,
) -> float:
    r"""The zero-momentum sum over exactly the box the library's
    ``direct_sum`` engine uses: every vertex label in ``[-L, L]^d``
    with ``root`` pinned at label ``0``.

    This is the twin of the ``itertools.product`` references in
    ``tests/test_box_nu_inf.py`` (``TestDGuardLift``), lifted to
    arbitrary kernels, and it is the same enumeration as
    :func:`brute_force_zeta` at zero momentum — the two names exist
    because they answer different questions.  ``brute_force_zeta`` is
    the *infinite-lattice* reference (its box is a truncation whose
    error must be argued about); ``box_enumeration`` is the
    *box-engine* reference (its box is the definition, and agreement
    is expected to round-off at every ``L``, with no truncation
    argument at all).

    On a finite box the pin is value-relevant — ``[-L, L]^d`` is not
    translation-closed — so ``root`` must match the root the engine
    under test used.
    """
    return brute_force_zeta(edges, kernel_fns, A, L, source=root)


# ---------------------------------------------------------------------------
# Cycle on the torus
# ---------------------------------------------------------------------------

def cycle_torus_reference(kernel_on_grid: np.ndarray, E: int) -> float:
    r"""Closed-form value of an ``E``-cycle on a periodic ``n^d`` torus
    whose every edge carries the same kernel.

    ``kernel_on_grid`` is the real array ``V[i_1, …, i_d] = V(z(i_1),
    …, z(i_d))`` sampled on the balanced label axis (index ``0`` is the
    zero displacement — the layout of
    ``gzl.tensor_network._balanced_z_axis``).  Pinning one
    vertex, the cycle sum is the ``E``-fold **cyclic** convolution of
    ``V`` evaluated at the origin, and the convolution theorem turns it
    into

    .. math::

        \zeta_{C_E} = \frac{1}{N} \sum_{\boldsymbol{k}}
                      \hat{V}(\boldsymbol{k})^E
                    = \operatorname{mean}\big(
                      \mathrm{fft}_n(V)^E \big),
        \qquad N = n^d,

    which is what this returns (real part).

    **This is the torus TRUNCATION of the cycle, not the
    infinite-lattice value.**  Two things are periodised: the vertex
    sums run over ``n^d`` sites rather than ``Z^d``, and the label
    difference of an edge is taken modulo ``n`` into the balanced
    window.  For a compactly supported kernel of reach ``R`` the
    truncation is exact once ``n > E R``: a closed ``E``-step walk of
    displacements of size ``<= R`` can wind around the torus only if its
    total displacement, at most ``E R``, reaches ``n``, so above that no
    wrapped walk is counted and no site with non-zero weight is missing.
    The weaker ``n > 2R`` (no single displacement wraps) is NOT enough —
    at ``E = 3, R = 1`` the ``n = 3`` truncation counts the two walks
    that wind once (pinned in ``tests/test_oracles.py``).  For a power
    law it converges
    algebraically in ``n`` (a pure ``ν``-cycle at ``E`` edges converges
    like ``n^(d - 2ν)``; e.g. ``d = 2, ν = 4.5`` gives ``n^-7``, which
    is why ``n = 256`` already sits at 3e-15 relative to
    ``zeta_circle``).

    Cost: one ``n^d`` FFT plus an elementwise power — negligible up to
    ``n^d ~ 1e7``.
    """
    V_grid = np.asarray(kernel_on_grid)
    if V_grid.ndim < 1:
        raise ValueError("cycle_torus_reference: kernel_on_grid must be an array.")
    if len(set(V_grid.shape)) != 1:
        raise ValueError(
            f"cycle_torus_reference: kernel_on_grid must be a cube, got "
            f"{V_grid.shape}."
        )
    E = int(E)
    if E < 2:
        raise ValueError(
            f"cycle_torus_reference: a cycle needs E >= 2 edges, got {E}."
        )
    return float(np.mean(np.fft.fftn(V_grid.astype(float)) ** E).real)
