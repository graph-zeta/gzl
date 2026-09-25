# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""
Construction of graph zeta functions from multigraphs.

This module provides two public entry points:

* :func:`graph_from_sp` — strict 2-connected series-parallel inputs,
  implemented by iterated parallel-merge + series-collapse on the
  primitives :func:`~gzl.core.graph_multiply` (serial
  composition) and :func:`~gzl.core.graph_convolve` (parallel
  composition).
* :func:`graph_from_tw2` — arbitrary treewidth-2 multigraphs.  Adds a
  block-cut decomposition layer on top of :func:`graph_from_sp`: the
  spine blocks (on the s-t block-cut-tree path) compose serially via
  :func:`~gzl.core.graph_multiply` at shared articulation
  points, while off-spine subtrees contribute as scalar factors
  computed recursively and multiplied in via
  :func:`~gzl.core.graph_attach`.

A two-terminal multigraph has treewidth ≤ 2 iff every biconnected
block is series-parallel iff it contains no K_4 minor between the
terminals.  :func:`graph_from_tw2` detects treewidth > 2 as the
failure of SP reduction on some block and raises
:class:`NotTreewidthTwoError` (a subclass of
:class:`NotSeriesParallelError`).

Every edge carries a float ``nu``; an edge may additionally carry a
general interaction kernel ``V(x) = a(x) + Σ_j b_j K_{ν_j}(x)`` (an
:class:`~gzl.interaction.Interaction`, or a lazy product of them
standing for parallel edges) in the ``kern`` attribute — the
``kernels=`` argument of :func:`graph_from_edges`.  ``nu`` then carries
the kernel's TAIL exponent (``min_j ν_j``, ``+inf`` for a purely compact
kernel): the one float the divergence check reads.  The kernel enters at
the single construction site, :func:`_sp_reduce_block`, as an elementary
:class:`GraphZeta` whose compact table sits in ``aMat`` and whose
power-law terms sit in ``(bVec, nuVec)``; everything downstream is the
unchanged algebra.  Without kernels the power-law path is executed as it
always was.

NetworkX, a dependency of gzl, is imported lazily, by the functions
that take a graph.
"""

from __future__ import annotations

import math

import numpy as np

from gzl._errors import GraphZetaError, UnsupportedLatticeSumError
from gzl._labels import relabel_to_support as _relabel_to_support
from gzl._lattices import _resolve_lattice
from gzl.core import (
    GraphZeta,
    graph_convolve,
    graph_multiply,
    graph_zero,
    make_epstein_graph,
    make_graph_obj,
)
from gzl.interaction import (
    InteractionSupportError,
    _KernelProduct,
    is_interaction,
)


__all__ = [
    "NotSeriesParallelError",
    "NotTreewidthTwoError",
    "graph_from_sp",
    "graph_from_sp_uniform",
    "graph_from_tw2",
    "graph_from_tw2_uniform",
    "graph_from_edges",
    "graph_from_edges_uniform",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NotSeriesParallelError(GraphZetaError):
    """Raised when the input multigraph is not series-parallel between
    the supplied terminals.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    The SP-reduction algorithm (iterated parallel-merge +
    series-collapse) is both a decision procedure and a decomposer:
    if it reduces the input to a single s-t edge, the graph is
    series-parallel; otherwise this error is raised.
    """


class NotTreewidthTwoError(NotSeriesParallelError):
    """Raised when the input multigraph has treewidth > 2.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    :func:`graph_from_tw2` detects treewidth > 2 by finding a
    biconnected block that fails SP reduction (a K_4 minor).  Because
    this error subclasses :class:`NotSeriesParallelError`, code that
    catches the latter will also catch this.
    """


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover — exercised only without nx
        raise ImportError(
            "This function requires NetworkX, a dependency of gzl, and it "
            "cannot be imported.  Reinstall gzl, or `pip install networkx`."
        ) from e
    return nx


# ---------------------------------------------------------------------------
# Common validation / preparation
# ---------------------------------------------------------------------------

def _validate_and_prepare(multigraph, s, t, func_name: str):
    """Shared validation.  Returns a fresh MultiGraph with ``nu`` edge
    attributes (copied from the input; an optional ``kern`` interaction
    kernel is copied beside it, see :func:`_check_edge_kernel`) and the
    connectivity dropped to the s-t component."""
    nx = _require_networkx()

    if not isinstance(multigraph, nx.MultiGraph):
        raise ValueError(
            f"{func_name}: multigraph must be a networkx.MultiGraph."
        )

    if s == t:
        raise ValueError(
            f"{func_name}: terminals s and t must be distinct."
        )

    if s not in multigraph or t not in multigraph:
        raise ValueError(
            f"{func_name}: both terminals must be in multigraph "
            f"(got s={s!r}, t={t!r})."
        )

    for u, v in multigraph.edges():
        if u == v:
            raise ValueError(
                f"{func_name}: self-loops are not allowed (vertex {u!r})."
            )

    G = nx.MultiGraph()
    G.add_nodes_from(multigraph.nodes())
    for u, v, data in multigraph.edges(data=True):
        if "nu" not in data:
            raise ValueError(
                f"{func_name}: edge ({u!r}, {v!r}) is missing the "
                f"required 'nu' attribute."
            )
        if is_interaction(data["nu"]):
            raise ValueError(
                f"{func_name}: edge ({u!r}, {v!r}) carries an Interaction "
                f"in 'nu'; put it in the 'kern' attribute and its tail "
                f"exponent (kern.tail_exponent) in 'nu'."
            )
        kern = data.get("kern")
        if kern is None:
            G.add_edge(u, v, nu=float(data["nu"]))
        else:
            _check_edge_kernel(
                kern, float(data["nu"]), f"{func_name}: edge ({u!r}, {v!r})"
            )
            G.add_edge(u, v, nu=float(data["nu"]), kern=kern)

    # A NetworkX input can express an isolated vertex, whose
    # infinite-lattice zeta diverges (escaping cluster cut 0) — refuse
    # it explicitly instead of silently dropping it with the components
    # outside the s-t component below (same convention as
    # evaluate_graph; see gzl/_labels.py).
    isolated = [v for v in G.nodes() if G.degree(v) == 0]
    if isolated:
        raise ValueError(
            f"{func_name}: multigraph has isolated node(s) "
            f"{isolated[:3]!r}{'...' if len(isolated) > 3 else ''}: an "
            f"isolated vertex's lattice-sum factor diverges.  Remove it."
        )

    if not nx.has_path(G, s, t):
        raise ValueError(
            f"{func_name}: terminals {s!r} and {t!r} are not connected."
        )

    st_component = nx.node_connected_component(G, s)
    G.remove_nodes_from(set(G.nodes()) - st_component)
    return G


# ---------------------------------------------------------------------------
# Interaction leaves (the kernel path of the single construction site)
# ---------------------------------------------------------------------------

def _balanced_label_grid(n: int, d: int) -> np.ndarray:
    """Integer lattice labels of the ``n^d`` balanced window in the layout
    ``aMat`` uses, shape ``(n,)*d + (d,)``.

    Per axis ``[0, 1, ..., floor(n/2), -ceil(n/2) + 1, ..., -1]`` — the
    ``z_axis`` of :func:`~gzl.core.graph_compress` and
    :func:`~gzl.core.graph_convolve`, spelled the same way, so a
    table scattered here lands exactly where those operations put a
    window-truncated power law.
    """
    pos = np.arange(0, int(np.floor(n / 2)) + 1, dtype=int)
    neg = np.arange(-int(np.ceil(n / 2)) + 1, 0, dtype=int)
    z_axis = np.concatenate([pos, neg])
    z_grids = np.meshgrid(*([z_axis] * d), indexing="ij")
    return np.stack(z_grids, axis=-1)


def _kernel_factors(kern) -> tuple:
    """The Interaction factors of an edge kernel: a lazy Hadamard product
    stands for parallel edges and contributes one factor each; a plain
    Interaction is its own single factor."""
    return kern.factors if isinstance(kern, _KernelProduct) else (kern,)


def _interaction_leaf(kern, A, n_points: int) -> GraphZeta:
    r"""The elementary :class:`GraphZeta` of ONE Interaction
    ``V(x) = a(x) + Σ_j b_j K_{ν_j}(x)``.

    The power-law terms go where :func:`~gzl.core.make_epstein_graph`
    puts its single exponent — ``(bVec, nuVec) = (b, ν)``, analytic until
    :func:`~gzl.core.graph_compress` decides otherwise — and the
    compact table goes into ``aMat``, scattered at its labels on the
    balanced window.  ``aMat[z]`` is the coefficient of ``e^{+2πi z·k}``
    in ``ζ_reg(k)``, i.e. ``V(-A z)`` for a bridge (the definition's phase
    is ``e^{-2πi (x_t - x_s)·k}``; see :class:`~gzl.core.GraphZeta`);
    ``a`` is even by construction, so ``V(-A z) = V(A z)`` and the table
    is scattered at its own labels ``m``.

    A table that does not fit the window raises
    :class:`~gzl.interaction.InteractionSupportError` (from
    :meth:`~gzl.interaction.Interaction.compact_array`) instead of
    being clipped: a clipped table makes the algebra return a plausible
    number that is simply wrong (measured 42 % off on a radius-6 table
    at ``n_points = 9``).
    """
    A = np.asarray(A, dtype=float)
    if int(n_points) < 1:
        raise InteractionSupportError(
            f"n_points = {int(n_points)}: a compact interaction table needs "
            f"a grid to be scattered on (the power-law path builds an empty "
            f"object here); use n_points >= 2 R + 1."
        )
    # The same lattice / dimension guard Interaction.sample applies: a
    # table sampled on one lattice must not be scattered onto another.
    kern._check_lattice(A)
    labels = _balanced_label_grid(int(n_points), A.shape[0])
    aMat = kern.compact_array(labels)
    # The leaf's table magnitude: Σ_m |a(m)| of the scattered table, the
    # ℓ¹ norm graph_zero_conditioned needs in order to see a table whose
    # entries cancel inside its np.sum over aMat (|Σ a| reports 1 for
    # it).  An empty table (a power-law kernel) gives 0.0, which keeps
    # its kappa bit-identical to the float path's; see GraphZeta.
    return make_graph_obj(aMat, bVec=kern.b, nuVec=kern.nu, A=A,
                          table_magnitude=float(np.sum(np.abs(aMat))))


def _check_edge_kernel(kern, nu: float, where: str) -> None:
    """Validate one ``(nu, kern)`` edge pair of the kernel path.

    ``nu`` keeps carrying the edge's TAIL exponent (``min_j ν_j``, ``+inf``
    for a purely compact kernel) — the one float every planner, cut and
    divergence check reads — so it must agree with the kernel's own.  A
    NaN never trips the ``nu <= d`` divergence check (every comparison
    with NaN is False), so it is refused here rather than passed through.
    """
    if not is_interaction(kern):
        raise ValueError(
            f"{where}: 'kern' must be an Interaction or a product of "
            f"Interactions; got {type(kern).__name__}."
        )
    tail = float(kern.tail_exponent)
    if math.isnan(nu) or not (
        nu == tail or math.isclose(nu, tail, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise ValueError(
            f"{where}: nu = {nu!r} is not the tail exponent {tail!r} of "
            f"its kernel.  With a kernel present, nu carries the per-edge "
            f"tail (min nu_j; +inf for a purely compact kernel)."
        )


# ---------------------------------------------------------------------------
# SP reduction primitive (used by both graph_from_sp and graph_from_tw2)
# ---------------------------------------------------------------------------

def _sp_reduce_block(
    block_mg,
    s,
    t,
    A,
    n_points: int,
    sigma_max: float,
    func_name: str = "graph_from_sp",
) -> GraphZeta:
    """Core SP-reduction loop on a single multigraph block.

    The input ``block_mg`` must be:
    * a ``networkx.MultiGraph`` whose edges carry a float ``nu``
      attribute, and optionally a ``kern`` interaction kernel (an
      :class:`~gzl.interaction.Interaction` or a lazy product of
      them; ``nu`` is then its tail exponent);
    * connected, containing ``s`` and ``t`` as distinct vertices, and
      free of self-loops.

    This is the single construction site: an edge without ``kern`` is
    the elementary Epstein graph of its ``nu``; an edge with ``kern``
    is one :func:`_interaction_leaf` per factor of the kernel, inserted
    as parallel edges so that the parallel-merge loop below composes
    them with :func:`~gzl.core.graph_convolve` — which IS the
    exact real-space Hadamard product (pointwise product of the Fourier
    parts, summed exponents), so no product is ever expanded here.

    Stops when a single edge between ``s`` and ``t`` remains; otherwise
    raises :class:`NotSeriesParallelError`.

    No top-level validation is performed — callers are expected to do
    that via :func:`_validate_and_prepare`.
    """
    nx = _require_networkx()

    # THE COMPOSED-SUPPORT GUARD.  A leaf's table fits the window once
    # n_points >= 2 R + 1, but the series collapse below is a CYCLIC
    # convolution on the same n-grid (graph_multiply), so a chain of
    # tables composes to the SUM of their radii and wraps around the
    # window long before any single leaf does — measured on the
    # nearest-neighbour indicator (R = 1) at n = 4: C4 33 % high, C6
    # 60 % high, no exception.  Every table-carrying edge of the block
    # contributes its radius (one per factor of a product bundle), and
    # the block is refused below that total rather than aliased; the
    # front-end lifts the block's grid to n_v * R + 2, which clears it.
    #
    # Σ_e R_e is an UPPER bound on the composed support, not the support
    # itself: a series collapse adds radii, but a parallel merge is a
    # POINTWISE product whose support is the MIN of its legs, so a block
    # with a parallel bundle in series is refused a little early.
    # Measured (d = 1, tables a(m) = 0.7^|m|, brute-force reference):
    # cycles C3-C6, theta-with-legs and the diamond are tight — the first
    # accepted n IS the first exact n — while the series-parallel
    # (2)-(3) bundle block is refused up to n = 5 at R = 1 (exact from
    # n = 4) and up to n = 10 at R = 2 (exact from n = 7); never too
    # loose: no accepted n returned a wrong number on 7 shapes x R in
    # {1, 2}.  An early exception is the cheap side to err on.
    _total_R = 0
    for _u, _v, _data in block_mg.edges(data=True):
        _kern = _data.get("kern")
        if _kern is not None:
            _total_R += sum(int(f.support_radius) for f in _kernel_factors(_kern))
    if _total_R and int(n_points) <= _total_R:
        raise InteractionSupportError(
            f"{func_name}: the block's compact parts compose to a support "
            f"of Chebyshev radius {_total_R} under the series collapse, "
            f"which wraps around an n_points = {int(n_points)} window; use "
            f"n_points > {_total_R} (the front-end lifts the block grid to "
            f"n_v * R + 2)."
        )

    # Working copy with GraphZeta objects on each edge.
    W = nx.MultiGraph()
    W.add_nodes_from(block_mg.nodes())
    for u, v, data in block_mg.edges(data=True):
        kern = data.get("kern")
        if kern is None:
            gz = make_epstein_graph(float(data["nu"]), A, n_points)
            W.add_edge(u, v, gz=gz)
        else:
            for factor in _kernel_factors(kern):
                W.add_edge(u, v, gz=_interaction_leaf(factor, A, n_points))

    while True:
        changed = False

        # -- parallel merge --------------------------------------------
        pairs = {tuple(sorted((u, v), key=repr)) for u, v in W.edges()}
        for (a, b) in pairs:
            if W.number_of_edges(a, b) >= 2:
                edge_keys = list(W.get_edge_data(a, b).keys())
                gzs = [W[a][b][k]["gz"] for k in edge_keys]
                merged = gzs[0]
                for gz in gzs[1:]:
                    merged = graph_convolve(merged, gz, sigma_max)
                for k in edge_keys:
                    W.remove_edge(a, b, key=k)
                W.add_edge(a, b, gz=merged)
                changed = True

        # -- series collapse ------------------------------------------
        for w in list(W.nodes()):
            if w == s or w == t or w not in W:
                continue
            if W.degree(w) != 2:
                continue
            edges = list(W.edges(w, keys=True))
            if len(edges) != 2:
                continue
            (_, n1, k1), (_, n2, k2) = edges
            if n1 == n2:
                continue
            gz1 = W[w][n1][k1]["gz"]
            gz2 = W[w][n2][k2]["gz"]
            merged = graph_multiply(gz1, gz2, sigma_max)
            W.remove_node(w)
            W.add_edge(n1, n2, gz=merged)
            changed = True

        if not changed:
            break

    remaining_nodes = set(W.nodes())
    if remaining_nodes != {s, t}:
        extra = sorted(remaining_nodes - {s, t}, key=repr)
        raise NotSeriesParallelError(
            f"{func_name}: multigraph is not series-parallel between "
            f"{s!r} and {t!r}. Irreducible vertices after SP reduction: "
            f"{extra}. This indicates a K_4 minor (treewidth > 2) or "
            f"a dangling-tree / articulation-point structure not "
            f"handled here."
        )
    n_edges = W.number_of_edges(s, t)
    if n_edges != 1:
        raise NotSeriesParallelError(
            f"{func_name}: expected a single edge between terminals "
            f"after SP reduction (got {n_edges})."
        )

    (key,) = list(W.get_edge_data(s, t).keys())
    return W[s][t][key]["gz"]


# ---------------------------------------------------------------------------
# Public: graph_from_sp (strict 2-connected SP)
# ---------------------------------------------------------------------------

def graph_from_sp(
    multigraph,
    s,
    t,
    A,
    n_points: int,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Construct a :class:`GraphZeta` from a 2-connected series-parallel multigraph.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Strict entry point: the input must be 2-connected between ``s``
    and ``t`` (no articulation points, no dangling trees).  For graphs
    with decorations / articulation points, use :func:`graph_from_tw2`
    instead.

    See :func:`graph_from_tw2` for parameter semantics — the interface
    is identical.

    Raises
    ------
    NotSeriesParallelError
        If the multigraph is not 2-connected SP between the terminals.
    ValueError, ImportError
        As in :func:`graph_from_tw2`.
    """
    G = _validate_and_prepare(multigraph, s, t, "graph_from_sp")
    # Nothing below normalises A: the raw value reaches every leaf builder.
    A = _resolve_lattice(A)
    return _sp_reduce_block(G, s, t, A, n_points, sigma_max, "graph_from_sp")


def graph_from_sp_uniform(
    multigraph,
    s,
    t,
    nu: float,
    A,
    n_points: int,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Uniform-nu convenience wrapper around :func:`graph_from_sp`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    nx = _require_networkx()

    if not isinstance(multigraph, nx.MultiGraph):
        raise ValueError(
            "graph_from_sp_uniform: multigraph must be a "
            "networkx.MultiGraph."
        )

    G = nx.MultiGraph()
    G.add_nodes_from(multigraph.nodes())
    for u, v in multigraph.edges():
        G.add_edge(u, v, nu=float(nu))

    return graph_from_sp(G, s, t, A, n_points, sigma_max=sigma_max)


# ---------------------------------------------------------------------------
# Block-cut tree helpers (private)
# ---------------------------------------------------------------------------

def _build_block_cut_tree(G):
    """Return (bc_tree, block_nodes, cuts).

    * ``bc_tree`` — an ``nx.Graph`` whose nodes are tagged
      ``("block", i)`` for biconnected blocks and ``("cut", v)`` for
      articulation vertices, with an edge between every block and
      each articulation vertex it contains.
    * ``block_nodes`` — dict ``{("block", i): frozenset(block_vertices)}``.
    * ``cuts`` — set of articulation-vertex labels.
    """
    nx = _require_networkx()

    blocks = [frozenset(c) for c in nx.biconnected_components(G)]
    cuts = set(nx.articulation_points(G))

    bc = nx.Graph()
    block_nodes: dict = {}
    for i, block in enumerate(blocks):
        bid = ("block", i)
        block_nodes[bid] = block
        bc.add_node(bid)
        for v in block:
            if v in cuts:
                cid = ("cut", v)
                bc.add_node(cid)
                bc.add_edge(bid, cid)

    return bc, block_nodes, cuts


def _bc_node_of(v, cuts, block_nodes):
    """Return the bc-tree node that should represent vertex ``v`` when
    it serves as a terminal: the cut-node if ``v`` is an articulation
    point, else the unique block containing ``v``."""
    if v in cuts:
        return ("cut", v)
    for bid, block in block_nodes.items():
        if v in block:
            return bid
    raise ValueError(f"vertex {v!r} not found in any block.")


def _spine_local_terminals(spine, s, t):
    """For each block on the spine, compute its (entry, exit) vertex pair."""
    result = []
    block_positions = [i for i, n in enumerate(spine) if n[0] == "block"]
    for i in block_positions:
        entry = s if i == 0 else spine[i - 1][1]
        exit_ = t if i == len(spine) - 1 else spine[i + 1][1]
        result.append((spine[i], entry, exit_))
    return result


def _scale_graph(g: GraphZeta, scalar) -> GraphZeta:
    """Multiply a GraphZeta by a scalar (same semantics as the scalar
    part of :func:`graph_attach`)."""
    c = complex(scalar)
    return make_graph_obj(
        aMat=g.aMat * c,
        bVec=g.bVec * c,
        nuVec=g.nuVec,
        A=g.A,
        table_magnitude=(None if g.table_magnitude is None
                         else g.table_magnitude * abs(c)),
    )


def _decoration_subgraph(G, component_nodes, attach_vertex, block_nodes):
    """Extract the multigraph of a decoration rooted at an articulation
    vertex.  Includes ``attach_vertex`` plus all graph vertices in the
    off-spine blocks of the bc-tree component."""
    nx = _require_networkx()

    dec_vertices = {attach_vertex}
    for node in component_nodes:
        if node[0] == "block":
            dec_vertices.update(block_nodes[node])
        else:
            dec_vertices.add(node[1])

    return _induced_subgraph(G, dec_vertices)


def _induced_subgraph(G, vertices):
    """Induced subgraph of ``G`` on ``vertices``, in ``G``'s order.

    ``G.subgraph(vertices)`` iterates in the order of the set
    ``vertices`` once it holds fewer than half of ``G``'s nodes.  For
    string labels that order follows Python's randomised hash, and the
    series collapse of :func:`_sp_reduce_block` and the terminal chosen
    by :func:`_zeta_at_zero` follow it, so the value changed at
    truncation level from one process to the next.  Parallel edges and
    edge keys are kept.
    """
    H = G.__class__()
    H.add_nodes_from((v, dict(G.nodes[v])) for v in G if v in vertices)
    H.add_edges_from(
        (u, v, k, dict(data))
        for u, v, k, data in G.edges(keys=True, data=True)
        if u in vertices and v in vertices
    )
    return H


def _attachment_vertex(bc_tree, spine_set, component_nodes):
    """Identify the articulation vertex where an off-spine bc-tree
    component joins the spine."""
    for node in component_nodes:
        for nbr in bc_tree.neighbors(node):
            if nbr in spine_set:
                # nbr is either a spine cut or a spine block.
                if nbr[0] == "cut":
                    return nbr[1]
                # nbr is a spine block; then this 'node' is a cut
                # (since bc-tree edges alternate block↔cut).
                assert node[0] == "cut"
                return node[1]
    return None


# ---------------------------------------------------------------------------
# Public: graph_from_tw2 (arbitrary treewidth-2)
# ---------------------------------------------------------------------------

def graph_from_tw2(
    multigraph,
    s,
    t,
    A,
    n_points: int,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Construct a :class:`GraphZeta` from an arbitrary treewidth-2 multigraph.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Performs a block-cut decomposition, reduces each biconnected block
    on the s-t path via :func:`graph_from_sp`, composes them serially
    through the articulation points via
    :func:`~gzl.core.graph_multiply`, and multiplies in each
    off-spine decoration's value at ``k = 0`` (computed recursively).

    Parameters
    ----------
    multigraph : networkx.MultiGraph
        Two-terminal multigraph.  Each edge must carry a ``nu``
        attribute (float).  An edge may also carry a ``kern`` attribute
        — an :class:`~gzl.interaction.Interaction`
        ``V(x) = a(x) + Σ_j b_j K_{ν_j}(x)``, or a lazy product of them
        standing for parallel edges — in which case ``nu`` must be that
        kernel's tail exponent (``kern.tail_exponent``: ``min_j ν_j``,
        ``+inf`` for a purely compact kernel).  Edges with and without
        ``kern`` may mix.
    s, t : hashable
        Terminal vertices.  Must both belong to the graph and be
        distinct.
    A : array-like (d x d) or str
        Lattice matrix, shared by every elementary graph, or the name of
        a lattice -- ``"chain"``, ``"square"``, ``"triangular"``,
        ``"cubic"``.
    n_points : int
        Discretisation points per dimension — used uniformly for the
        spine and every decoration.  Each compact table of Chebyshev
        radius ``R`` needs ``n_points >= 2 R + 1`` (the balanced window
        must hold both ``+R`` and ``-R``), and a block's tables COMPOSE
        under the series collapse (a cyclic convolution), so the block
        needs ``n_points > Σ_e R_e`` over its table-carrying edges; below
        either bound :class:`~gzl.interaction.InteractionSupportError`
        is raised rather than an aliased number returned (the front-end
        lifts a block's grid to ``n_v R + 2``, which clears both).
        ``Σ_e R_e`` is an UPPER bound on the composed support — a
        parallel bundle's pointwise product has the support of its
        smallest leg, not the sum — measured tight on cycles and
        theta-with-legs, conservative for a bundle in series, and never
        too loose (see the guard in :func:`_sp_reduce_block`).
    sigma_max : float
        Singularity threshold forwarded to :func:`graph_multiply` /
        :func:`graph_convolve`.

    Returns
    -------
    GraphZeta

    Raises
    ------
    NotTreewidthTwoError
        If some biconnected block has a K_4 minor (treewidth > 2).
    ValueError
        On degenerate input (self-loops, missing terminals, terminals
        in different components, missing ``nu`` attribute, non-MultiGraph,
        a ``kern`` that is not an Interaction or whose tail exponent
        disagrees with ``nu``).
    InteractionSupportError
        If a ``kern`` compact table does not fit the ``n_points`` window.
    ImportError
        If NetworkX is not installed.

    Notes
    -----
    At ``k = 0`` the zeta function of a decoration is independent of
    the choice of terminals inside the decoration (the exponential
    phase collapses).  The recursive call therefore picks the
    attachment vertex as one terminal and its first neighbour as the
    second, so that each decoration is a product of block values.
    The blocks on the s-t path are composed with
    :func:`~gzl.core.graph_multiply`, whose Notes describe the loss of
    accuracy when such a product carries an exponent on
    ``nu = d + 2n``.  Nodes lying outside the s-t connected component
    are silently dropped.
    """
    G = _validate_and_prepare(multigraph, s, t, "graph_from_tw2")
    # Nothing below normalises A: the raw value reaches every leaf builder.
    A = _resolve_lattice(A)
    try:
        return _build_tw2(G, s, t, A, n_points, sigma_max)
    except NotSeriesParallelError as e:
        raise NotTreewidthTwoError(
            f"graph_from_tw2: multigraph has treewidth > 2 — some "
            f"biconnected block is not series-parallel.\n  {e}"
        ) from e


def graph_from_tw2_uniform(
    multigraph,
    s,
    t,
    nu: float,
    A,
    n_points: int,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Uniform-nu convenience wrapper around :func:`graph_from_tw2`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.
    """
    nx = _require_networkx()

    if not isinstance(multigraph, nx.MultiGraph):
        raise ValueError(
            "graph_from_tw2_uniform: multigraph must be a "
            "networkx.MultiGraph."
        )

    G = nx.MultiGraph()
    G.add_nodes_from(multigraph.nodes())
    for u, v in multigraph.edges():
        G.add_edge(u, v, nu=float(nu))

    return graph_from_tw2(G, s, t, A, n_points, sigma_max=sigma_max)


# ---------------------------------------------------------------------------
# Internal: tw-2 construction on an already validated graph
# ---------------------------------------------------------------------------

def _build_tw2(G, s, t, A, n_points, sigma_max) -> GraphZeta:
    """Block-cut decomposition + spine + decorations.  Assumes ``G`` is
    connected, contains ``s`` and ``t``, has ``nu`` edge attributes
    (plus validated optional ``kern`` kernels), and has no self-loops."""
    nx = _require_networkx()

    bc_tree, block_nodes, cuts = _build_block_cut_tree(G)

    if not block_nodes:
        # Possible only if G has no edges; caller should have ensured
        # connectivity of s and t with at least one edge.
        raise ValueError("graph_from_tw2: no edges in s-t component.")

    s_node = _bc_node_of(s, cuts, block_nodes)
    t_node = _bc_node_of(t, cuts, block_nodes)

    if s_node == t_node:
        # s, t lie in the same block (non-articulation terminals).
        spine = [s_node]
    else:
        spine = nx.shortest_path(bc_tree, s_node, t_node)
    spine_set = set(spine)

    # --- spine blocks -------------------------------------------------
    spine_pieces = []
    for block_node, entry, exit_ in _spine_local_terminals(spine, s, t):
        block_sub = _induced_subgraph(G, block_nodes[block_node])
        gz = _sp_reduce_block(
            block_sub, entry, exit_, A, n_points, sigma_max,
            func_name="graph_from_tw2",
        )
        spine_pieces.append(gz)

    result = spine_pieces[0]
    for gz in spine_pieces[1:]:
        result = graph_multiply(result, gz, sigma_max)

    # --- off-spine decorations ---------------------------------------
    bc_minus = bc_tree.copy()
    bc_minus.remove_nodes_from(spine_set)

    for component in nx.connected_components(bc_minus):
        component_set = set(component)
        attach = _attachment_vertex(bc_tree, spine_set, component_set)
        if attach is None:
            continue  # unreachable component (defensive)

        dec_mg = _decoration_subgraph(G, component_set, attach, block_nodes)
        if dec_mg.number_of_edges() == 0:
            continue

        scalar = _zeta_at_zero(dec_mg, attach, A, n_points, sigma_max)
        result = _scale_graph(result, scalar)

    return result


def _zeta_at_zero(dec_mg, attach_vertex, A, n_points, sigma_max) -> complex:
    """Compute ζ_dec(k=0) by recursing into :func:`graph_from_tw2`.

    At ``k = 0`` the choice of terminals inside a graph is irrelevant
    for the exact value of the zeta function, so we pick the attachment
    vertex as one terminal and its first neighbour as the other.  A
    neighbour keeps the spine of the recursion to the single block that
    holds the edge between them, and every further block becomes a
    decoration of its own.  The decoration is then a product of block
    values at ``k = 0``, and no serial product of blocks is sampled
    directly.  Such a product loses accuracy when an operand carries an
    exponent on ``nu = d + 2n`` (see the Notes of
    :func:`~gzl.core.graph_multiply`): with an arbitrary vertex as the
    second terminal, a triangle with a pendant path at ``nu = 1.5`` on
    the chain went through triangle times edge and lost three orders.
    """
    other = next(
        (v for v in dec_mg.neighbors(attach_vertex) if v != attach_vertex),
        None,
    )
    if other is None:
        return 1.0 + 0j  # pure vertex, no contribution

    gz = graph_from_tw2(
        dec_mg, attach_vertex, other, A, n_points, sigma_max=sigma_max
    )
    return complex(graph_zero(gz))


# ---------------------------------------------------------------------------
# Public: graph_from_edges (low-level array constructor)
# ---------------------------------------------------------------------------

def _nu_carries_an_interaction(nu) -> bool:
    """True when ``nu`` is an Interaction or holds one — a list, a tuple,
    or an object-dtype ndarray of them (numpy's own message for the
    latter is an opaque ``float() argument must be ...`` TypeError)."""
    if is_interaction(nu):
        return True
    if isinstance(nu, np.ndarray):
        return nu.dtype == object and any(is_interaction(x) for x in nu.flat)
    if np.isscalar(nu):
        return False
    try:
        return any(is_interaction(x) for x in nu)
    except TypeError:
        return False


def graph_from_edges(
    edges,
    nu,
    A,
    n_points: int,
    s: int = 0,
    t: int = 0,
    sigma_max: float = 4.0,
    *,
    kernels=None,
) -> GraphZeta:
    r"""Construct a :class:`GraphZeta` directly from an indexed edge list.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Implements the formal definition

    .. math::

       \zeta_G(\boldsymbol{k}) \;=\;
       {\sum_{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(N-1)}}}^{\!\!\!\prime}
       e^{-2\pi i (\boldsymbol{x}^{(t)} - \boldsymbol{x}^{(s)}) \cdot \boldsymbol{k}}
       \prod_{e \in E} \lvert \boldsymbol{x}^{(v_e)} - \boldsymbol{x}^{(u_e)} \rvert^{-\nu_e}

    with vertex 0 pinned at the origin by translational invariance and the
    primed sum excluding configurations where any edge would collapse.
    With ``kernels`` the edge factor ``|x|^{-\nu_e}`` becomes the general
    kernel ``V_e(x) = a_e(x) + \sum_j b_j |x|^{-\nu_j}``, whose compact
    part ``a_e`` may weight coincident endpoints (``a_e(0)``).

    This is the array-based counterpart to :func:`graph_from_tw2`. Internally
    it builds a :class:`networkx.MultiGraph` with per-edge ``nu`` attributes
    (and ``kern`` attributes when ``kernels`` is given) and dispatches to
    :func:`graph_from_tw2`.

    Parameters
    ----------
    edges : array-like of shape (n, 2), int
        Indexed list of undirected edges. ``edges[i] = (u_i, v_i)``.
        Multigraphs are encoded by repeating rows; e.g. a double bond
        between vertices 0 and 1 is ``[[0, 1], [0, 1]]``. Vertex labels
        are names: the vertex set is the labels present in ``edges``
        (at least two are required), compressed order-preservingly to
        ``0..V-1`` on entry; self-loops (``u == v``) are not allowed.
    nu : array-like of shape (n,), float
        Per-edge exponents.  Must satisfy ``nu[i] > d`` (with
        ``d = A.shape[0]``).  Complex exponents are not yet supported
        because the EpsteinLib backend is real-only.  With ``kernels``
        given, ``nu[i]`` is the TAIL exponent of ``kernels[i]``
        (``kernels[i].tail_exponent``: ``min_j nu_j``, ``+inf`` for a
        purely compact kernel) — the divergence check reads it, the
        kernel supplies the terms.
    A : array-like (d, d), float, or str
        Lattice basis, or the name of a lattice -- ``"chain"``,
        ``"square"``, ``"triangular"``, ``"cubic"``.
    n_points : int
        Number of discretisation points per spatial dimension.
    s, t : int, optional
        Distinguished terminals.  The Fourier factor couples to
        :math:`\boldsymbol{x}^{(t)} - \boldsymbol{x}^{(s)}`.  When
        ``s == t`` (the default ``(0, 0)``) the Fourier factor is
        identically 1, so the value at :math:`\boldsymbol{k} = 0` is the
        vacuum lattice sum and is independent of the choice of
        terminals.  In that case the function picks an auxiliary
        terminal internally (the smallest vertex ``!= s``) so that the
        underlying tw-2 reduction can run; the returned object is then
        only meaningful at :math:`\boldsymbol{k} = 0` (use
        :func:`graph_zero`), not at non-zero momentum.
    sigma_max : float, optional
        Singularity threshold forwarded to the underlying constructors.
        The ``sigma_max > 0`` Gamma-function algebra is meant for inputs
        with ``sigma = min nu - d < 1.49`` — the band the front-end
        routes here (:func:`gzl.evaluate_graph`); ``sigma_max =
        0`` is the exact-Fourier method for anything else.  Exponents
        within ~1e-5 of the Gamma-pole family ``d + 2k`` (``k >= 1``)
        but NOT on it are a known pre-existing flank of the pole
        absorption (``singularity_tol = 1e-4`` in
        :func:`~gzl.core.graph_compress`): measured on the chain
        at ``nu = d + 2 + 1e-6`` the triangle is 65 % off and the C4
        sign-flipped, while the pole itself, a 1e-7 detuning and every
        detuning ``>= 1e-5`` are fine — identical on the power-law and
        the kernel path, and never reached through the front-end.
    kernels : sequence of Interaction, optional
        Keyword-only.  Per-edge interaction kernels
        ``V_e(x) = a_e(x) + Σ_j b_j K_{ν_j}(x)`` — an
        :class:`~gzl.interaction.Interaction` each, or a lazy
        product ``I1 * I2`` standing for parallel edges — aligned with
        the rows of ``edges``; ``nu`` then carries each kernel's tail
        exponent.  ``None`` (default) is the power-law path, which runs
        exactly as before.  A compact table of Chebyshev radius ``R``
        needs ``n_points >= 2 R + 1``, and a block's tables compose under
        the series collapse, so the block needs ``n_points > Σ_e R_e``
        (see :func:`graph_from_tw2`); the front-end lifts the grid.
        That sum is an UPPER bound on the composed support (a parallel
        bundle's support is the min of its legs), measured never too
        loose.  The built object carries
        :attr:`~gzl.core.GraphZeta.table_magnitude`, so
        :func:`~gzl.core.graph_zero_conditioned` sees a table
        whose entries cancel.

    Returns
    -------
    GraphZeta
        Two-terminal graph zeta function.  Use :func:`graph_zero` for
        the value at :math:`\boldsymbol{k} = \boldsymbol{0}` (vacuum /
        0qp series), or :func:`graph_sample` for the full momentum grid
        (1qp series), provided ``s != t``.

    Raises
    ------
    NotTreewidthTwoError
        If the underlying simple graph contains a :math:`K_4` minor.
    ValueError
        On shape mismatches, self-loops, ``nu[i] <= d``, terminals out
        of range, fewer than two vertices, a ``kernels`` entry that is
        not an Interaction, or a ``nu[i]`` that is not its kernel's tail
        exponent (NaN included).
    InteractionSupportError
        If a kernel's compact table does not fit the ``n_points`` window.
    ImportError
        If NetworkX is not installed.

    See Also
    --------
    graph_from_edges_uniform : convenience wrapper for a single scalar nu.
    graph_from_tw2           : multigraph-input counterpart.
    """
    nx = _require_networkx()

    edges_arr = np.asarray(edges, dtype=int)
    if _nu_carries_an_interaction(nu):
        raise ValueError(
            "graph_from_edges: nu carries an Interaction; pass the kernels "
            "through kernels= and their tail exponents (kern.tail_exponent) "
            "in nu (evaluate_graph does this for you)."
        )
    nu_arr = np.asarray(nu, dtype=float)
    A_arr = np.asarray(_resolve_lattice(A), dtype=float)

    if edges_arr.ndim != 2 or edges_arr.shape[1] != 2:
        raise ValueError(
            f"graph_from_edges: edges must have shape (n, 2); got "
            f"{edges_arr.shape}."
        )
    n_edges = edges_arr.shape[0]
    if n_edges == 0:
        raise ValueError(
            "graph_from_edges: edges must contain at least one edge."
        )
    if nu_arr.shape != (n_edges,):
        raise ValueError(
            f"graph_from_edges: nu must have shape ({n_edges},); got "
            f"{nu_arr.shape}."
        )
    if kernels is not None:
        kernels = list(kernels)
        if len(kernels) != n_edges:
            raise ValueError(
                f"graph_from_edges: kernels has length {len(kernels)} but "
                f"edges has {n_edges} rows."
            )
        for i, (kern, nu_e) in enumerate(zip(kernels, nu_arr.tolist())):
            _check_edge_kernel(kern, float(nu_e), f"graph_from_edges: edge {i}")
    if (edges_arr < 0).any():
        raise ValueError(
            "graph_from_edges: edges contain negative vertex indices."
        )

    self_loops = edges_arr[:, 0] == edges_arr[:, 1]
    if self_loops.any():
        bad = int(np.where(self_loops)[0][0])
        raise ValueError(
            f"graph_from_edges: self-loop at edge index {bad} "
            f"(vertices {edges_arr[bad].tolist()}); not allowed."
        )

    # The vertex set is the edge support (see gzl/_labels.py):
    # sparse labels compress order-preservingly, a contiguous input
    # passes through unchanged, and a terminal label that appears in no
    # edge is not a vertex and raises.
    edges_arr, _refs = _relabel_to_support(edges_arr,
                                           {"s": int(s), "t": int(t)})
    s, t = _refs["s"], _refs["t"]

    n_vertices = int(edges_arr.max()) + 1
    if n_vertices < 2:
        raise ValueError(
            "graph_from_edges: need at least two distinct vertices."
        )

    d = A_arr.shape[0]
    if np.isnan(nu_arr).any():
        raise ValueError(f"graph_from_edges: nu contains NaN: {nu_arr!r}")
    if not (nu_arr > d).all():
        bad = int(np.where(~(nu_arr > d))[0][0])
        raise UnsupportedLatticeSumError(
            f"graph_from_edges: nu[{bad}] = {nu_arr[bad]} is not strictly "
            f"greater than d = {d}, which the algebra does not support. "
            "Every exponent must satisfy nu_e > d (a sufficient condition "
            "for convergence, not a necessary one; see "
            "UnsupportedLatticeSumError)."
        )

    # Resolve s == t to a distinct auxiliary terminal for the underlying
    # tw-2 reducer.  At k = 0 the result is independent of this choice.
    if s == t:
        aux = next(v for v in range(n_vertices) if v != s)
        t_eff = aux
    else:
        t_eff = t

    G = nx.MultiGraph()
    G.add_nodes_from(range(n_vertices))
    if kernels is None:
        for (u, v), nu_e in zip(edges_arr.tolist(), nu_arr.tolist()):
            G.add_edge(int(u), int(v), nu=float(nu_e))
    else:
        # _relabel_to_support keeps the row order, so kernels stay aligned.
        for (u, v), nu_e, kern in zip(edges_arr.tolist(), nu_arr.tolist(), kernels):
            G.add_edge(int(u), int(v), nu=float(nu_e), kern=kern)

    return graph_from_tw2(G, s, t_eff, A_arr, int(n_points), sigma_max=sigma_max)


def graph_from_edges_uniform(
    edges,
    nu: float,
    A,
    n_points: int,
    s: int = 0,
    t: int = 0,
    sigma_max: float = 4.0,
) -> GraphZeta:
    """Uniform-nu convenience wrapper around :func:`graph_from_edges`.

    Provisional API, which may change in a minor release.  See "API
    stability" in DOCUMENTATION.md.

    Equivalent to::

        graph_from_edges(edges, np.full(len(edges), nu), A, n_points, ...)

    Useful for transverse-field Ising-model series and any other case in
    which all edges carry the same exponent.
    """
    edges_arr = np.asarray(edges, dtype=int)
    if edges_arr.ndim != 2 or edges_arr.shape[1] != 2:
        raise ValueError(
            f"graph_from_edges_uniform: edges must have shape (n, 2); got "
            f"{edges_arr.shape}."
        )
    if _nu_carries_an_interaction(nu):
        raise ValueError(
            "graph_from_edges_uniform: nu carries an Interaction; pass the "
            "kernels through graph_from_edges(..., kernels=) and their tail "
            "exponents (kern.tail_exponent) in nu (evaluate_graph does this "
            "for you)."
        )
    nu_vec = np.full(edges_arr.shape[0], float(nu), dtype=float)
    return graph_from_edges(
        edges_arr, nu_vec, A, n_points, s=s, t=t, sigma_max=sigma_max
    )
