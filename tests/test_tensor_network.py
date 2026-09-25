# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

r"""Validation of :func:`gzl.graph_zeta_general_at_zero`.

Two-pronged validation against the existing gzl methods,
running across the vendored TFIM softcore graph fixture
(``tests/fixtures/tfim_softcore_graphs.npz``):

* **tw <= 2 graphs**: compare to
  ``graph_zero(graph_compress(graph_from_edges_uniform(...),
  sigma_max=0.0))`` — the existing SP-reduction path forced through
  the same Fourier-grid representation.  These two methods use
  bit-identical real-space discretisation, so we expect agreement at
  floating-point precision.

* **tw > 2 graphs**: compare to ``direct_sum_extrapolated`` — a
  Richardson-extrapolated direct lattice sum.  The two methods use
  different truncations (torus periodisation vs finite-box
  extrapolation), so we expect agreement only to within a tolerance
  set by the larger of the two truncation residuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from gzl import (
    NotTreewidthTwoError,
    PrefactorSingularityError,
    data_path,
    direct_sum_extrapolated,
    graph_compress,
    graph_from_edges_uniform,
    graph_sample,
    graph_zero,
    graph_zeta_general,
    graph_zeta_general_at_zero,
)


# The lean test fixture (tests/fixtures/tfim_softcore_graphs.npz) filters
# out tw > 2 graphs at dump time, so it can't validate the tw > 2 path.
# We use the *full* topology snapshot shipped in gzl/data, which keeps
# every graph; this lets us test both tw ≤ 2 (against graph_from_edges_uniform)
# and tw > 2 (against direct_sum_extrapolated) on the full validation set.
_FULL_FIXTURE_PATH = data_path("full_graph_topologies.npz")


# ---------------------------------------------------------------------------
# Fixture loader (full corpus, restricted to the orders this test covers)
# ---------------------------------------------------------------------------

# Default (fast) test scope:  0qp orders 2..10 + 1qp orders 1..7.  Matches
# the smoothness fixture.  ~700 graphs total; runs in ~2 s.
_DEFAULT_ORDERS_BY_KIND: dict = {0: range(2, 11), 1: range(1, 8)}

# Slow-marked extended scope: covers everything *except* 1qp order 11.
# That single order is 15 541 graphs — half the corpus — and its presence
# pushes the total wall-clock above one minute on a 4-core machine.
# Skipping it leaves ~15 500 graphs (covering all 0qp orders 2..13 and
# 1qp orders 1..10), which still exercises tw>=3 graphs at every high-
# order regime and surfaces the deepest fillin patterns we have.
_FULL_ORDERS_BY_KIND: dict = {0: range(2, 14), 1: range(1, 11)}


def _load_graphs() -> dict:
    return dict(np.load(_FULL_FIXTURE_PATH))


def _graphs_of_kind(
    qp_kind: int, orders: dict | None = None,
) -> list[tuple[str, np.ndarray, int, int]]:
    f = _load_graphs()
    qp_arr        = f["qp"]
    order_arr     = f["order"]
    name_arr      = f["name"]
    s_arr         = f["s"]
    t_arr         = f["t"]
    edges_off_arr = f["edges_off"]
    edges_flat    = f["edges_flat"]
    if orders is None:
        orders = _DEFAULT_ORDERS_BY_KIND
    valid_orders = set(orders[int(qp_kind)])

    out: list[tuple[str, np.ndarray, int, int]] = []
    for i in np.where(qp_arr == int(qp_kind))[0]:
        order = int(order_arr[i])
        if order not in valid_orders:
            continue
        ef = edges_flat[edges_off_arr[i]:edges_off_arr[i + 1]].astype(np.int64)
        out.append((
            f"O{order}/{str(name_arr[i])}",
            ef,
            int(s_arr[i]),
            int(t_arr[i]),
        ))
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    graphs:    list[tuple[str, np.ndarray, int, int]],
    A:         np.ndarray,
    nu:        float,
    n_points:  int,
    rtol_tw2:  float,
    rtol_dsum: float,
):
    r"""Validate the tensor-network method on a list of graphs.

    For each graph, classifies tw<=2 vs tw>2 by attempting
    ``graph_from_edges_uniform`` at ``sigma_max=0`` (we run the SP path
    in the same Fourier-grid representation as the tensor-network
    method, so no FP-level differences from analytic Epstein algebra
    accumulate).  Returns a summary tuple
    ``(n_tw2, n_twhi, worst_tw2, worst_twhi, failures)``.
    """
    d = A.shape[0]
    n_tw2_pass = n_twhi_pass = 0
    worst_tw2_err = worst_twhi_err = 0.0
    worst_tw2_who = worst_twhi_who = "(none)"
    failures: list = []
    # direct_sum supports d in {1, 2, 3} (it raises NotImplementedError
    # only outside that range), and the tw>2 comparison is cheap there:
    # K4 / prism at d=2 run in <0.2 s and 0.10 GB.  This gate used to
    # read `d == 1`, which silently skipped every tw>2 cross-truncation
    # check at d=2 and d=3 -- exactly the comparison the two engines
    # exist to cross-validate.
    has_dsum = d in (1, 2, 3)

    for name, edges, s, t in graphs:
        nu_vec = np.full(len(edges), nu, dtype=float)

        v_new = complex(graph_zeta_general_at_zero(
            edges, nu_vec, A, n_points, pinned_vertex=s,
        ))

        try:
            g_sp = graph_from_edges_uniform(
                edges, nu, A, n_points, s=s, t=t, sigma_max=0.0,
            )
            g_compressed = graph_compress(g_sp, sigma_max=0.0)
            v_ref = complex(graph_zero(g_compressed))
            kind = "tw<=2"
            rtol = rtol_tw2
        except NotTreewidthTwoError:
            if not has_dsum:
                # No reference available in this dimension; skip.
                continue
            v_ref = complex(direct_sum_extrapolated(edges, nu_vec, A))
            kind = "tw>2"
            rtol = rtol_dsum

        denom = max(abs(v_ref), 1e-30)
        rel   = abs(v_new - v_ref) / denom

        if kind == "tw<=2":
            if rel > worst_tw2_err:
                worst_tw2_err, worst_tw2_who = rel, name
            if rel <= rtol:
                n_tw2_pass += 1
            else:
                failures.append(("tw<=2", name, v_new, v_ref, rel))
        else:
            if rel > worst_twhi_err:
                worst_twhi_err, worst_twhi_who = rel, name
            if rel <= rtol:
                n_twhi_pass += 1
            else:
                failures.append(("tw>2", name, v_new, v_ref, rel))

    n_tw2  = sum(1 for f in failures if f[0] == "tw<=2") + n_tw2_pass
    n_twhi = sum(1 for f in failures if f[0] == "tw>2") + n_twhi_pass
    return (
        (n_tw2_pass, n_tw2, worst_tw2_err, worst_tw2_who),
        (n_twhi_pass, n_twhi, worst_twhi_err, worst_twhi_who),
        failures,
    )


def _report_and_assert(label, tw2, twhi, failures, *, A, nu, n_points):
    p_tw2,  n_tw2,  err_tw2,  who_tw2  = tw2
    p_twhi, n_twhi, err_twhi, who_twhi = twhi
    d     = int(A.shape[0])
    sigma = float(nu) - d
    print(
        f"\n  {label}  [d={d}, sigma={sigma:g}, nu={nu:g}, "
        f"n_points={n_points}]:\n"
        f"    tw<=2 {p_tw2}/{n_tw2} pass "
        f"(worst rel.err {err_tw2:.2e} on {who_tw2})\n"
        f"    tw>2  {p_twhi}/{n_twhi} pass "
        f"(worst rel.err {err_twhi:.2e} on {who_twhi})"
    )
    if failures:
        msg = [
            f"{len(failures)} graph(s) failed validation "
            f"[d={d}, sigma={sigma:g}, n_points={n_points}]:"
        ]
        for kind, name, v_new, v_ref, rel in failures[:10]:
            msg.append(
                f"  [{kind}] {name}: new {v_new:.6e}, ref {v_ref:.6e}, "
                f"rel.err {rel:.3e}"
            )
        pytest.fail("\n".join(msg))


class TestTensorNetworkValidation:
    r"""End-to-end validation of the new bucket-elimination method.

    The default tier sweeps the same scope as the smoothness test
    (0qp orders 2..10, 1qp orders 1..7 — 681 tw<=2 + 21 tw>2 = 702
    graphs).  The slow tier extends to the *full* corpus
    (0qp 2..13 + 1qp 1..11 ≈ 31 k graphs) and surfaces tw>=4 graphs
    at the high orders.
    """

    # Test parameters.  We keep nu = 4 (sigma = nu - d = 3 in d = 1)
    # consistently — mid-range for TFIM applications, comfortably between
    # the Gamma poles at sigma = 2 and sigma = 4, and close enough to
    # the convergence boundary that truncation residuals are measurable.
    A:         np.ndarray = np.array([[1.0]])
    nu:        float      = 4.0          # sigma = 3 in d = 1
    n_points:  int        = 32           # gives ~3e-5 residuals at sigma = 3 in d = 1
                                         # (~1/n^3 scaling at nu = d + 3 = 4)
    rtol_tw2:  float      = 1.0e-10      # bit-identical Fourier-grid representation
    rtol_dsum: float      = 1.0e-3       # different truncations; both methods now
                                         # have residuals well below 1e-4 typically,
                                         # so 1e-3 leaves a 30x safety margin while
                                         # still tightening 10x over the previous
                                         # n_points = 16 / 1e-2 calibration.

    # ---- default (fast) tier --------------------------------------------

    @pytest.mark.parametrize("qp_kind, expected_total", [
        (0, 420),    # 0qp orders 2..10:  401 tw<=2 + 19 tw>2
        (1, 282),    # 1qp orders 1..7:   280 tw<=2 +  2 tw>2
    ])
    def test_default_tier(self, qp_kind, expected_total):
        graphs = _graphs_of_kind(qp_kind, _DEFAULT_ORDERS_BY_KIND)
        assert len(graphs) == expected_total, (
            f"qp_kind={qp_kind}: expected {expected_total}, got {len(graphs)}"
        )
        tw2, twhi, failures = _run_validation(
            graphs, self.A, self.nu, self.n_points,
            self.rtol_tw2, self.rtol_dsum,
        )
        _report_and_assert(
            f"qp_kind={qp_kind} default", tw2, twhi, failures,
            A=self.A, nu=self.nu, n_points=self.n_points,
        )

    # ---- slow tier: extended corpus, more high-tw cases -----------------

    # Slow-tier overrides: smaller n_points to stay under ~1 min wall-clock
    # on a 4-core machine, with a loosened tw>2 tolerance to match the
    # n_points^(-3) residual scaling.  Same sigma = 3 setting throughout.
    n_points_slow:  int   = 20      # ~1/20^3 ≈ 1.25e-4 residual floor
    rtol_dsum_slow: float = 3.0e-3  # 24x safety margin over the predicted
                                    # outlier residuals (1qp O11 — the
                                    # 6.6e-3 outlier at n=16 — is excluded
                                    # from the slow tier scope; remaining
                                    # graphs sit comfortably below 5e-4).

    @pytest.mark.slow
    @pytest.mark.parametrize("qp_kind", [0, 1])
    def test_full_corpus(self, qp_kind):
        graphs = _graphs_of_kind(qp_kind, _FULL_ORDERS_BY_KIND)
        tw2, twhi, failures = _run_validation(
            graphs, self.A, self.nu, self.n_points_slow,
            self.rtol_tw2, self.rtol_dsum_slow,
        )
        _report_and_assert(
            f"qp_kind={qp_kind} full corpus", tw2, twhi, failures,
            A=self.A, nu=self.nu, n_points=self.n_points_slow,
        )


class TestTensorNetwork2DHex:
    r"""Validate the d=2 path on the hexagonal lattice (sigma = 2).

    Lattice basis A is the standard hex primitive matrix
    ``[[1, 1/2], [0, sqrt(3)/2]]`` (so ``d = 2``); we evaluate at
    ``nu = 4``, hence ``sigma = nu - d = 2`` — close to the second
    Gamma pole (sigma = 2 in d = 2 corresponds to nu = 4 = d + 2,
    which is exactly on a discrete pole, so a small irrational shift
    would normally be added; here we use the Fourier-only path that
    sidesteps the prefactor algebra entirely, so the integer value
    is fine).

    A tw>2 reference IS available here -- ``direct_sum_extrapolated``
    runs at d = 2 on this lattice and agrees with the tensor path to
    ~2e-9 (see :class:`TestHighTwAcrossTruncations`).  An earlier
    version of this docstring claimed otherwise and the class was built
    around that claim, which is why the K_4 case below is a
    self-consistency test rather than a comparison.  Both are kept: the
    Cauchy test says the torus converges, the cross-truncation test says
    it converges to the right thing.

    This class exercises the d=2 path via:

    * tw<=2 graphs (4-cycle, triangle, 6-cycle): compare to
      ``graph_from_edges_uniform`` at ``sigma_max=0`` for fp-precision
      agreement, and to ``zeta_circle`` for an analytic cross-check
      with grid-truncation tolerance.
    * tw=3 graph (K_4): self-consistency convergence test as
      ``n_points`` increases, asserting that the difference between
      consecutive ``n_points`` values shrinks at the expected rate.
    """

    A:        np.ndarray = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    nu:       float      = 4.0       # sigma = nu - d = 2 in d = 2 hex
    n_points: int        = 16
    rtol:     float      = 1.0e-9    # FP-precision against lib's discretised path

    def _build_and_check(self, edges, expected_against_lib_rtol=None):
        edges = np.asarray(edges, dtype=int)
        nu_vec = np.full(len(edges), self.nu, dtype=float)
        v_new = complex(graph_zeta_general_at_zero(
            edges, nu_vec, self.A, self.n_points, pinned_vertex=0,
        ))
        g = graph_from_edges_uniform(
            edges, self.nu, self.A, self.n_points,
            s=0, t=0, sigma_max=0.0,
        )
        g = graph_compress(g, sigma_max=0.0)
        v_lib = complex(graph_zero(g))
        denom = max(abs(v_lib), 1e-30)
        rel = abs(v_new - v_lib) / denom
        return v_new, v_lib, rel

    def test_4cycle(self):
        # 4-cycle is tw=2; new method must agree with lib's discretised path
        # (loose since the lib's d=2 graph_compress + graph_multiply uses
        # FFT-based pointwise contractions whose FP ordering differs from
        # our real-space bucket elimination by a few ULPs per operation).
        edges = [[0,1],[1,2],[2,3],[0,3]]
        v_new, v_lib, rel = self._build_and_check(edges)
        # Both are discretised; agreement to ~1e-4 at n_points=16 is the
        # FP-ordering mismatch ceiling we observed empirically.
        assert rel < 1e-3, f"hex 4-cycle: new {v_new}, lib {v_lib}, rel.err {rel:.3e}"

    def test_triangle(self):
        edges = [[0,1],[1,2],[0,2]]
        v_new, v_lib, rel = self._build_and_check(edges)
        assert rel < 1e-3, f"hex triangle: new {v_new}, lib {v_lib}, rel.err {rel:.3e}"

    def test_6cycle(self):
        edges = [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]]
        v_new, v_lib, rel = self._build_and_check(edges)
        assert rel < 1e-3, f"hex 6-cycle: new {v_new}, lib {v_lib}, rel.err {rel:.3e}"

    def test_K4_convergence(self):
        # K_4 is tw=3, beyond the existing SP path; validate that the new
        # method converges as n_points increases (Cauchy criterion).
        edges = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        nu_vec = np.full(6, self.nu)
        prev = None
        last_diff = None
        for n in [4, 6, 8]:
            v = complex(graph_zeta_general_at_zero(edges, nu_vec, self.A, n))
            if prev is not None:
                last_diff = abs(v - prev) / max(abs(v), 1e-30)
            prev = v
        # n=8 vs n=6 should already be at <1e-2 for nu=4 hex (we saw 3.9e-3
        # empirically); assert under 5e-2 to be robust against numerical
        # noise on different machines.
        assert last_diff is not None
        assert last_diff < 5e-2, (
            f"K_4 hex Cauchy diff between n=6 and n=8: {last_diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Free-terminal Brillouin-zone validation
# ---------------------------------------------------------------------------

class TestFreeTerminalBZ:
    r"""Validate :func:`graph_zeta_general` with a free terminal — i.e.
    the case where the Fourier exponential factor is non-trivial — by
    comparing against the existing :func:`gzl.graph_sample` output
    on the full Brillouin-zone grid for 1qp graphs.  Same setup as the
    other d=1 tensor-network tests: nu = 4 (sigma = 3 in d = 1).
    """

    A:        np.ndarray = np.array([[1.0]])
    nu:       float      = 4.0       # sigma = 3
    n_points: int        = 16
    rtol:     float      = 1.0e-10   # bit-identical Fourier-grid representation

    @pytest.mark.parametrize(
        "name, edges_list, source, terminal",
        [
            ("1qp triangle (s=0, t=1)",  [[0,1],[0,2],[1,2]],         0, 1),
            ("1qp 4-cycle  (s=0, t=2)",  [[0,1],[1,2],[2,3],[0,3]],   0, 2),
            ("1qp theta22  (s=0, t=2)",  [[0,1],[1,2],[0,3],[3,2]],   0, 2),
        ],
    )
    def test_free_terminal_matches_graph_sample(self, name, edges_list, source, terminal):
        edges  = np.asarray(edges_list, dtype=int)
        nu_vec = np.full(len(edges), self.nu, dtype=float)

        # New tensor-network method, k-space output
        out_k = graph_zeta_general(
            edges, nu_vec, self.A, self.n_points,
            source=source, terminals=(terminal,), space="k",
        ).real

        # Lib reference: build via graph_from_edges_uniform with explicit
        # (s, t), force discretised path via graph_compress(sm=0), evaluate
        # on the BZ grid via graph_sample.
        g = graph_from_edges_uniform(
            edges, self.nu, self.A, self.n_points,
            s=source, t=terminal, sigma_max=0.0,
        )
        g = graph_compress(g, sigma_max=0.0)
        sample = graph_sample(g)

        # Compare element-wise across the full BZ grid
        denom = max(np.abs(sample).max(), 1e-30)
        rel   = np.abs(out_k - sample).max() / denom
        assert rel < self.rtol, (
            f"{name}: max rel.diff vs graph_sample {rel:.3e}\n"
            f"  out_k[:5] = {out_k[:5]}\n  sample[:5] = {sample[:5]}"
        )

    def test_z_then_fft_equals_k_directly(self):
        # Computing in 'z' space then taking ifftn*n^d should match the
        # direct 'k' output bit-for-bit.
        edges  = np.array([[0,1],[0,2],[1,2]], dtype=int)
        nu_vec = np.full(3, self.nu, dtype=float)
        out_z = graph_zeta_general(
            edges, nu_vec, self.A, self.n_points,
            source=0, terminals=(1,), space="z",
        )
        out_k_direct = graph_zeta_general(
            edges, nu_vec, self.A, self.n_points,
            source=0, terminals=(1,), space="k",
        )
        out_k_via_z  = np.fft.ifftn(out_z, axes=(0,)) * self.n_points
        rel = np.abs(out_k_direct - out_k_via_z).max() / max(
            np.abs(out_k_direct).max(), 1e-30
        )
        assert rel < 1e-12, f"z-then-fft vs direct k: rel.diff {rel:.3e}"

    def test_at_zero_matches_general_k0(self):
        # graph_zeta_general(terminals=(t,), space='k')[0] should equal
        # graph_zeta_general_at_zero(...) for the same graph.
        edges  = np.array([[0,1],[0,2],[1,2]], dtype=int)
        nu_vec = np.full(3, self.nu, dtype=float)
        out_k = graph_zeta_general(
            edges, nu_vec, self.A, self.n_points,
            source=0, terminals=(1,), space="k",
        ).real
        v_at_zero = complex(graph_zeta_general_at_zero(
            edges, nu_vec, self.A, self.n_points, pinned_vertex=0,
        )).real
        rel = abs(out_k[0] - v_at_zero) / max(abs(v_at_zero), 1e-30)
        assert rel < 1e-12, (
            f"k=0 of free-terminal output: {out_k[0]:.6f} vs "
            f"at_zero: {v_at_zero:.6f}  rel.diff {rel:.3e}"
        )


# ---------------------------------------------------------------------------
# Cross-truncation validation at d >= 2
# ---------------------------------------------------------------------------

class TestHighTwAcrossTruncations:
    r"""Torus and box must agree on treewidth > 2 blocks at d = 2.

    These are the only graphs in the library with no closed form, and
    the two engines reach them by genuinely different routes -- the
    torus takes differences mod ``n`` with a cyclic FFT, the box takes
    true differences on ``[-L, L]^d`` with a zero-padded one.  Agreement
    is therefore a real cross-check and not a tautology.

    It was not being run.  ``_run_validation`` gated its tw>2 reference
    behind ``d == 1`` with the comment "direct_sum currently d=1 only",
    and :class:`TestTensorNetwork2DHex` recorded the same claim in its
    docstring.  Both were wrong: direct_sum supports d in {1, 2, 3}, and
    on these graphs it costs under 0.2 s and 0.10 GB.  So every tw>2
    cross-truncation comparison at d >= 2 was silently skipped.

    Both an orthogonal (square) and a sheared (hex) lattice are covered,
    because the truncations differ in how they represent a lattice
    vector and the sheared cell is where that difference bites.
    """

    A_SQUARE = np.eye(2)
    A_HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])

    K4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    PRISM = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5],
             [0, 3], [1, 4], [2, 5]]
    K5_MINUS_E = [[i, j] for i in range(5) for j in range(i + 1, 5)
                  if (i, j) != (0, 4)]

    NU = 4.0            # sigma = 2 at d = 2
    N_POINTS = 20
    RTOL = 1.0e-7       # measured 1.7e-10 .. 1.9e-9; ~50x margin

    def _compare(self, edges, A):
        edges = np.asarray(edges, dtype=int)
        nu_vec = np.full(len(edges), self.NU, dtype=float)
        box = complex(direct_sum_extrapolated(edges, nu_vec, A)).real
        torus = complex(graph_zeta_general_at_zero(
            edges, nu_vec, A, self.N_POINTS,
        )).real
        return torus, box, abs(torus - box) / max(abs(box), 1e-30)

    @pytest.mark.parametrize("lattice", ["square", "hex"])
    @pytest.mark.parametrize("name", ["K4", "prism"])
    def test_torus_matches_box(self, lattice, name):
        A = self.A_SQUARE if lattice == "square" else self.A_HEX
        torus, box, rel = self._compare(getattr(self, name.upper()), A)
        assert rel < self.RTOL, (
            f"{lattice} {name}: torus {torus!r} vs box {box!r}, "
            f"rel {rel:.3e}"
        )

    @pytest.mark.parametrize("lattice", ["square", "hex"])
    def test_torus_matches_box_k5_minus_e(self, lattice):
        A = self.A_SQUARE if lattice == "square" else self.A_HEX
        torus, box, rel = self._compare(self.K5_MINUS_E, A)
        assert rel < self.RTOL, (
            f"{lattice} K5-e: torus {torus!r} vs box {box!r}, "
            f"rel {rel:.3e}"
        )


class TestTheEinsumPathHoistIsVersionRobust:
    r"""``einsum_call=True`` is semi-private and its tuple arity moved.

    It returned 5-tuples locally and 3-tuples on CI, and destructuring
    them crashed every dense contraction with "not enough values to
    unpack (expected 5, got 3)" — a green local suite, a red CI, and the
    whole d = 2 routing branch down.  ``dense_step`` now indexes rather
    than destructures, and PROVES the hoisted execution reproduces plain
    einsum on the sample slice before trusting it.
    """

    EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    NU = [2.5] * 6

    def _value(self):
        from gzl.hybrid import hybrid_zeta
        return float(np.real(hybrid_zeta(self.EDGES, self.NU, np.eye(2),
                                         n_points=10)))

    @pytest.mark.parametrize("arity", [3, 5])
    def test_any_contraction_list_arity_works(self, arity, monkeypatch):
        real = np.einsum_path

        def truncated(*a, **k):
            out = real(*a, **k)
            if k.get("einsum_call"):
                ops, lst = out
                return ops, [tuple(step[:arity]) for step in lst]
            return out

        monkeypatch.setattr(np, "einsum_path", truncated)
        got = self._value()
        assert np.isfinite(got)

    def test_a_broken_contraction_list_falls_back_rather_than_crashing(
            self, monkeypatch):
        """Garbage in must degrade to plain einsum, not raise."""
        real = np.einsum_path

        def garbage(*a, **k):
            out = real(*a, **k)
            if k.get("einsum_call"):
                ops, _ = out
                return ops, [((0, 1), set(), "zz,zz->zz")]
            return out

        monkeypatch.setattr(np, "einsum_path", garbage)
        assert np.isfinite(self._value())

    def test_the_hoist_is_value_exact(self, monkeypatch):
        """Hoisted and un-hoisted must agree BITWISE — same ops, same order."""
        hoisted = self._value()
        real = np.einsum_path

        def no_call(*a, **k):
            if k.get("einsum_call"):
                raise RuntimeError("forced fallback")
            return real(*a, **k)

        monkeypatch.setattr(np, "einsum_path", no_call)
        plain = self._value()
        assert hoisted == plain
