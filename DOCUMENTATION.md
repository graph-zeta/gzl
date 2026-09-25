# GZL documentation

The [README](README.md) covers installation, the definition of the graph
zeta function, and evaluating a single graph or a corpus. This document
carries the rest. It describes the front-ends in full, general
interaction kernels, finite momentum, the worked examples, the engines,
the shipped reference data, which parts of the API are stable, and the
complete API reference.

## Single-graph front-end: `evaluate_graph`

For most use cases the simplest entry point is `evaluate_graph`, a
single function that takes either a flat edge list or a
`networkx.MultiGraph` and evaluates $\zeta_G(\boldsymbol{k})$ via a
topology-first router. External momentum is zero by default, and an
arbitrary $\boldsymbol{k}$, a batch, or a full Brillouin-zone grid is
available via the `momentum` keyword, described in [Finite external
momentum](#finite-external-momentum) below. Internally it runs the
three-step reduction of [Evaluating a single graph
zeta](README.md#evaluating-a-single-graph-zeta) in the README, a
Hadamard merge, a block-cut decomposition and then per-block routing,
picking the cheapest correct evaluator for each biconnected block:

| Biconnected block of the block-cut decomposition | Evaluator |
|---|---|
| Bridge ($n_v = 2$, $n_e = 1$) | closed-form `epstein_zeta(ν_B, A, 0, 0)` |
| Simple cycle ($n_v \ge 3$, $n_e = n_v$) | closed-form `zeta_circle(ν_vec, A)` |
| SP-reducible at the chosen terminals | `graph_from_edges → graph_zero`, singular tails as Epstein zeta, regular Fourier on the `n_points`-grid |
| Otherwise (treewidth $\ge 3$, or SP reduction fails) | torus tensor network, optional 3-point Richardson |

The algebra-vs-tensor decision among SP-reducible blocks is made
per-block on the smallest $\sigma \equiv \nu - d$: for
$\sigma < 1.49$ the block takes the semi-analytical algebra
path (the slow $1/\lvert x\rvert^\nu$ tail is retained as an Epstein
zeta, and only the regular Fourier part sees the grid). For
$\sigma \ge 1.49$ it falls through to the torus tensor network
(improved-Fourier, exponentially convergent, no Γ-prefactor
cancellation). Graphs that decompose entirely into bridges plus
simple cycles evaluate fully analytically, and no `n_points`-grid
is touched.

The lattice sum converges absolutely if $\nu_e > d$ on every edge, and
for other exponents $\zeta_G$ is defined by meromorphic continuation.
`evaluate_graph` returns this continuation for bridges, where it is
given by the Epstein zeta function, and raises
`UnsupportedLatticeSumError` at its pole $\nu = d$, which a bridge
reaches for a momentum in the reciprocal lattice. Every other block
with an exponent $\nu \le d$ after the Hadamard merge raises
`UnsupportedLatticeSumError` as well. This is a limitation of the
implementation and not a statement about convergence. Several of these
sums converge, such as the triangle with $\nu = 0.9$ on the chain, but
no evaluator of GZL is validated there.

The algebra path handles the Γ((d−ν)/2) pole family ν = d + 2n
(n = 1, 2, …) of its `graph_multiply` prefactor in two
complementary ways, both automatic. An *input* exponent on or near a
pole (an edge at σ = 2, a Hadamard bundle at σ = k − d/2, or the
exponent 2ν = 3 of a triangle with ν = 3/2 on the chain) is absorbed
into the regular Fourier part before the prefactor is formed. A
*cross term* whose exponent lands on the pole family is a finite
0 × ∞ limit, a structurally vanishing prefactor times the next
multiplication's Γ-pole. The resonance condition is m·σ ∈ 2ℕ for a
chain-reachable m, which is hit at σ = 0.2, 0.25, 0.4, 0.5 and 1.0. The
library emits it at a canonically offset exponent with the exact
residue-limit prefactor, so the exponent tower it generates survives.
In `evaluate_graph`, resonant σ therefore evaluate at the same
background accuracy as generic σ, and no irrational offset of ν is
needed. This relies on the block structure. An absorbed input loses
the tail of its kernel beyond the grid window, and the loss stays
below the truncation error of the algebra only if a parallel
composition follows the product. Inside a block every serial
composition is closed by a parallel one. Products of the algebra that
are evaluated directly are not, see
[The `sigma_max` parameter](#the-sigma_max-parameter).
`PrefactorSingularityError` is raised only for an exponent at or below
d, which `evaluate_graph` refuses beforehand. The Γ-prefactor algebra
is bypassed entirely on the tensor path and, equivalently, at
σ_max = 0 (the improved Fourier method, with exactly known Fourier
coefficients).

**Nearest-neighbour limit.** Passing `nu = np.inf` selects the exact
nearest-neighbour kernel, the indicator of the *minimal nonzero
lattice shell* instead of $1/\lvert x\rvert^\nu$, so $\zeta_G$ counts
nearest-neighbour graph *homomorphisms* into the lattice exactly
(integer-valued and independent of `n_points`). A bridge then returns
the lattice coordination number (square 4, triangular 6, FCC 12).
Because the kernel keys off the *minimal* nonzero distance rather than
$\lvert x\rvert = 1$, it is scale-invariant. It returns the same
combinatorial count at any lattice scale, including sub-unit
nearest-neighbour distances. The literal $1/\lvert x\rvert^\nu$ has no
such limit unless the nearest-neighbour distance is exactly 1. For a
*smaller* distance every nearest-neighbour weight diverges as
$\nu \to \infty$, while for a *larger* distance the whole kernel
vanishes. Only the indicator gives the coordination count in every
case, which is why `nu = np.inf` is a dedicated kernel rather than a
large finite $\nu$. At finite
momentum the nearest-neighbour value is an exact finite trigonometric
polynomial: each block's real-space form factor is computed exactly and
`n_points` sets only the output Brillouin-zone resolution. A
square-lattice nearest-neighbour bridge, for instance, reproduces

$$\zeta_G(\boldsymbol{k}) = 2\bigl(\cos 2\pi k_x + \cos 2\pi k_y\bigr)$$

to machine precision at any `n_points`. (Relatedly, a simple cycle
whose smallest $\sigma = \nu - d$ exceeds 10 is sent to the tensor
rather than the closed-form `zeta_circle`, which is slow and
overflow-prone at large exponent.)

The example below builds a composite graph on the **2D triangular
lattice** that exercises all three of bridge / simple-cycle /
algebra in a single call: a diamond 0-1-2-3 with chord 0-2 (a
tw-2 spine block, SP-reducible at the chord endpoints 0 and 2),
a 3-cycle pendant attached at vertex 0, and an Epstein bridge
attached at vertex 2.

<p align="center">
  <img src="docs/figures/evaluate_graph_example.svg" alt="Composite graph topology used in the evaluate_graph example" />
</p>

The figure is generated as TikZ via the small reusable
[`docs/figures/plot_multigraph.py`](docs/figures/plot_multigraph.py)
tool (run [`docs/figures/evaluate_graph_example.py`](docs/figures/evaluate_graph_example.py) to re-render).
A vector PDF is committed alongside the SVG at
[`docs/figures/evaluate_graph_example.pdf`](docs/figures/evaluate_graph_example.pdf).

```python
import numpy as np
import networkx as nx
from gzl import evaluate_graph

# 2D triangular lattice
A = np.array([
    [1.0, 0.5             ],
    [0.0, np.sqrt(3) / 2.0],
])
nu = 3.0

# Composite graph (see figure above): diamond 0-1-2-3 with chord 0-2,
# triangle pendant 0-4-5, Epstein bridge 2-6.
edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [0, 2],
    [0, 4], [4, 5], [5, 0],
    [2, 6],
], dtype=int)

val, info = evaluate_graph(edges, nu, A, n_points=128, return_diagnostics=True)
print(info["n_bridges"], info["n_simple_cycles"], info["n_block_algebra"])
# 1 1 1   (one bridge + one simple cycle + one algebra-evaluated block)

# Equivalent NetworkX form. `nu` is read from each edge's attribute.
G = nx.MultiGraph()
for u, v in edges:
    G.add_edge(int(u), int(v), nu=nu)
assert evaluate_graph(G, A, n_points=128) == val

# Reference: product of per-block contributions, precomputed once on
# the same triangular lattice with this nu:
#   evaluate_graph(diamond, nu=3.0, A=A, n_points=256)   # algebra
#   zeta_circle(np.full(3, 3.0), A).real                 # closed-form
#   epstein_zeta(3.0, A, np.zeros(2), np.zeros(2)).real  # closed-form
REF = 7.6997399866714275e+01 * 2.5582967615946501e+01 * 1.1034175734914811e+01
print(f"val = {val:.10e}, ref = {REF:.10e}, "
      f"rel err = {abs(val - REF) / REF:.3e}")
# val = 2.1735361919e+04, ref = 2.1735361974e+04, rel err = 2.5e-9
```

`nu` may be a scalar (uniform across edges) or a 1-D array of
length $E$.  In the MultiGraph form it is read per-edge from the `nu`
attribute.  Hadamard merging produces one bundle per simple-graph
edge with the summed exponent. Non-uniform per-edge ν is handled
natively by both the algebra and the tensor path.

Errors are reported as structured exceptions, all subclassing
`GraphZetaError(ValueError)`:

* `SelfLoopError`, raised when the input edge list contains a self-loop $(v, v)$.
* `DisconnectedGraphError`, raised when the simple graph (after Hadamard-merging)
  is disconnected.
* `VertexOutOfRangeError`, raised when `source` or a member of `terminals` is not
  a vertex of the graph (the vertex set is the labels present in the
  edge list, and a label in no edge is not a vertex).
* `NPointsRequiredError`, raised when at least one block needs the σ-routed
  evaluator and `n_points <= 0`, or when the `n_points ** d` sites of the
  torus cannot hold such a block, that is, when its vertices cannot be
  placed on them with the two ends of every edge on distinct sites. There
  every term of the torus sum vanishes.
* `UnsupportedLatticeSumError`, raised for an exponent $\nu \le d$ as
  described above.
* `TopologyEvaluatorUnavailableError`, raised when `networkx` cannot be
  imported. `networkx` and `epsteinlib` are dependencies of gzl, so this
  means a broken installation, and without `epsteinlib` `import gzl`
  already fails.

A request that no evaluator serves raises `UnsupportedRequestError`,
which is also a `NotImplementedError`. Such requests are two or more
free terminals, and a 1qp graph whose path from `source` to `terminal`
runs through a block that mixes $\nu = \infty$ with finite exponents, at
any momentum. The vacuum value of that graph is evaluated, and so is
every momentum once its nearest-neighbour edges are written as
`Interaction.nearest_neighbour(A)`.

A malformed argument raises a plain `ValueError` or `TypeError`, which
is not a `GraphZetaError`. Examples are an edge list whose rows are not
pairs, a vertex label, `source`, `terminal` or `n_points` that is not a
whole number, a momentum that is not finite, and an `A` that is not a
finite, non-singular square matrix or a lattice name.

### General interactions

Every front-end that accepts an exponent `nu` also accepts an
`Interaction` (the engines take one through their keyword-only
`kernels=` list and refuse one in `nu_vec` with a message naming it).
This is the general kernel
$K(\boldsymbol{x}) = a(\boldsymbol{x}) + \sum_j b_j \mathcal{K_{\nu_j}}(\boldsymbol{x})$
of the [definition](README.md#definition-of-the-graph-zeta-function). Several
power-law terms, plus a real, even, compactly supported short-range
part $a$ stored as a table over integer lattice labels
($\boldsymbol{x} = A\boldsymbol{m}$). A plain
`Interaction(b=[1], nu=[ν])` is demoted to the float `ν` at the boundary,
so its values are byte-identical to those of the float `ν`.

```python
import numpy as np
from gzl import Interaction, evaluate_graph, zeta_circle

A = "triangular"
J1, J2, Jd = 0.5, 0.2, 0.1

# total couplings on the nearest (|x| = 1) and next-nearest (|x| = sqrt 3)
# shells, and a dipolar tail Jd / |x|^3 beyond them
K = Interaction.from_shells(A, {1.0: J1, np.sqrt(3): J2},
                            b=[Jd], nu=[3.0], label="J1J2dip")
# a short-range function of the displacement VECTOR (may be anisotropic;
# must be even), sampled on |A m| <= radius, the origin included
W = Interaction.from_function(lambda x: 0.4 * np.exp(-np.linalg.norm(x, axis=-1)),
                              A, radius=3.0, b=[1.0], nu=[3.0])
T = Interaction.from_table({(1, 0): 0.3, (-1, 0): 0.3, (0, 1): 0.3, (0, -1): 0.3},
                           b=[1.0, -0.2], nu=[3.0, 5.0])       # explicit table + two terms
NN = Interaction.nearest_neighbour(A)                          # identical numbers to nu=np.inf

edges = [[0, 1], [1, 2], [0, 2]]                      # a triangle
evaluate_graph([[0, 1]], K, A)                        # bridge: K.lattice_sum(A), exact
evaluate_graph([[0, 1]], K, A, terminal=1, n_points=48)       # its dispersion, exact at every k
evaluate_graph([[0, 1], [1, 2], [0, 2]], K, A)        # cycle: the generalised closed form
evaluate_graph(edges, [K, 3.0, T], A, n_points=32)            # per edge, floats and Interactions mixed
zeta_circle([K, K, K], A)
```

The routing is based on the tail exponent $\min_j \nu_j$ of each edge,
which is $+\infty$ for a purely compact interaction. A block whose
interactions are all purely compact is therefore treated like a block
at `nu = np.inf`. It is evaluated exactly by the torus tensor network
on the smallest torus that reproduces the lattice sum, so that its value
does not depend on `n_points`. Any other block with a compact part
takes the routes of a power-law block, but every torus grid it runs on
holds at least $n_v R + 2$ points per dimension, with $n_v$ the number
of vertices of the block and $R$ the largest `support_radius` of its
interactions. On a smaller torus, the tables along a cycle can wind
around it and add a spurious term. If `n_points` is below this size,
it is raised for that block alone. Bridges and simple cycles with a
power-law tail keep their closed forms. A bridge evaluates to the exact
lattice sum of its kernel at any momentum, given by Epstein zeta
functions for the power-law terms plus a finite sum over the table.
For a simple cycle, the Fourier transform of the compact part is a
trigonometric polynomial, so that `zeta_circle` still applies. Its
quadrature rule is refined until two consecutive rules agree to
$10^{-9}$ relative to the magnitude of the integrand, and a cycle for
which the refinement does not converge is evaluated like any other
series-parallel block instead.

Parallel edges multiply pointwise, so that an edge of multiplicity $m$
carries the kernel $K^m$. When an engine is called directly on a grid
that cannot hold the support of a table, it raises
`InteractionSupportError` instead of returning an aliased value. The
block cache keys on the fingerprint of each interaction, so that
kernels with the same tail exponent but different tables never share
an entry.

A worked script, `examples/interaction_j1_j2_dipolar_triangular.py`,
builds $J_1 + J_2$ + dipolar on the triangular lattice, checks a single
edge against the Fourier transform of the coupling, and computes the
gap series at Γ and K.

The values are validated as described above. The spelling of this
interface is still settling, so treat the `Interaction` constructor set,
the 12-decimal quantisation of its cache fingerprint, and the routing
thresholds for mixed kernels as provisional, see
[API stability](#api-stability). The power-law path is unaffected either
way.

### Finite external momentum

Setting a free terminal turns the graph into a one-quasiparticle
(1qp) object whose value depends on an external momentum
$\boldsymbol{k}$, the lattice Fourier transform of the two-point
correlator defined at the top of this section.  Bravais lattices
are closed under $\boldsymbol{x} \mapsto -\boldsymbol{x}$, so
$\zeta_G(\boldsymbol{k}) = \zeta_G(-\boldsymbol{k}) \in \mathbb{R}$
and the return type stays real.

Block-cut at finite $\boldsymbol{k}$ identifies the unique block-tree
path between $s$ and $t$, the spine.  Off-spine decorations
attach via 1-sums and contribute $\zeta_B(\boldsymbol{0})$ by the
attachment theorem, and only on-spine blocks see $\boldsymbol{k}$.
On the typical 1qp corpus the spine is one to three blocks, so the
finite-**k** pipeline reuses the zero-momentum
evaluators on every off-spine block and only routes the spine
through finite-**k** paths
(`epstein_zeta(ν, A, 0, k_lat)` for bridges, `graph_sample` /
`graph_sample_at` for SP-reducible blocks, and the torus tensor
network otherwise).

The `momentum` keyword takes one of four shapes, and the result type
follows:

| `momentum` | shape rule | returns |
|---|---|---|
| `None` (vacuum, `terminal is None` or `terminal == source`) | k-independent | `float` $\zeta_G(\boldsymbol{0})$ |
| `None` (1qp, `terminal != source`) | full Brillouin zone, requires `n_points > 0` | `ndarray` of shape $(n_\text{points})^d$ |
| `scalar` (in $d = 1$) or `(d,)` ndarray | one $\boldsymbol{k}$-vector | `float` $\zeta_G(\boldsymbol{k})$ |
| 1-D array (in $d = 1$) or `(N, d)` ndarray | $N$ $\boldsymbol{k}$-vectors | `ndarray` of shape $(N,)$ |

`momentum` is in fractional Brillouin-zone coordinates, the
same convention as `graph_sample`'s grid, with $i$-th component
$\beta_i \in [0, 1)$ along the $i$-th dual basis vector, so that
the lattice Fourier kernel is
$e^{-2\pi i\boldsymbol{\beta}\cdot\boldsymbol{n}}$ for the
integer lattice index $\boldsymbol{n} \in \mathbb{Z}^d$.  This is
$\boldsymbol{\beta} = A^T\boldsymbol{k}$ where
$\boldsymbol{k}$ is the wavevector in the convention of the
definition at the top of this section (the Cartesian wavevector
divided by $2\pi$).  For an orthonormal basis (1D chain, 2D square)
the two coincide, and on the 2D triangular lattice they differ by
$A^T$.  Internally `evaluate_graph` recovers the Cartesian
wavevector via `k_lat = (A.T)^-1 @ momentum` before passing it to
`epstein_zeta`.

In particular, if you have a physics wavevector
$\boldsymbol{k_{\text{phys}}}$ (radians per length, so that
$e^{-i\boldsymbol{k_{\text{phys}}}\cdot\boldsymbol{x}}$ is the
Fourier kernel against Cartesian position $\boldsymbol{x}$), the
corresponding `momentum` is `A.T @ k_phys / (2 * np.pi)`.  The
full-BZ grid then samples
`k_phys = 2π · (A.T)^-1 @ (i/n, j/n, ...)` for
$i, j, \ldots \in [0, n_\text{points})$, the parallelogram image of
the equidistant $\boldsymbol{\beta}$-grid in physical
$\boldsymbol{k}$-space.

The full-BZ-grid path exploits an FFT redundancy in the on-spine
evaluators. When the terminal is a path-end vertex of the block-cut
tree (the typical 1qp case), producing the full
$(n_\text{points})^d$-grid is essentially the same compute cost as
producing one single $\boldsymbol{k}$.  Single-**k** and
batch modes are therefore *memory* optimisations, returning a
`float` or `(N,)`-array instead of an
$(n_\text{points})^d$-array, rather than compute optimisations.

The example below reuses the same composite triangular-lattice graph
as above, now with explicit spine $(s, t) = (0, 2)$ (the diamond's
chord endpoints).  The diamond is on-spine and sees
$\boldsymbol{k}$ through the algebra path. The triangle pendant at
vertex 0 and the Epstein bridge at vertex 2 are off-spine and
contribute $\zeta_B(\boldsymbol{0})$.

```python
import numpy as np
from gzl import evaluate_graph

A = np.array([
    [1.0, 0.5             ],
    [0.0, np.sqrt(3) / 2.0],
])
nu = 3.0
edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [0, 2],   # diamond  (on-spine, sees k)
    [0, 4], [4, 5], [5, 0],                   # triangle (off-spine, k = 0)
    [2, 6],                                   # bridge   (off-spine, k = 0)
], dtype=int)

# (1) single-k value, in fractional reciprocal-lattice coordinates.
k = np.array([0.25, 0.10])
v = evaluate_graph(edges, nu, A, source=0, terminal=2,
                   momentum=k, n_points=128)
# v ≈ 7.4242800207e+03

# (2) full Brillouin-zone grid in one call, shape (n_points,) ** d.
n = 128
grid = evaluate_graph(edges, nu, A, source=0, terminal=2, n_points=n)
assert grid.shape == (n, n)
# grid[0, 0] is ζ_G(0), equal to the zero-momentum value from the
# previous example to FFT-discretisation precision:
#     grid[0, 0] ≈ 2.1735361919e+04

# (3) batch of k-vectors, one call, shape (N,) result.
ks = np.array([[0.0, 0.0], [0.25, 0.10], [0.5, 0.5]])
batch = evaluate_graph(edges, nu, A, source=0, terminal=2,
                       momentum=ks, n_points=128)
# batch ≈ [21735.36,  7424.28, -5867.93]
```

The 1qp gap series, the corpus coefficient $c_r(\boldsymbol{k})$
of [Evaluating a corpus](README.md#evaluating-a-corpus), at
finite $\boldsymbol{k}$ is the natural application. The shipped
TFIM-softcore Monte-Carlo data validates this path on the 1D chain
across $\sigma \in \lbrace 1, 2, 5 \rbrace$ and on 2D square / triangular at
$\sigma = 2$ (see [`tests/test_finite_k_validation.py`](tests/test_finite_k_validation.py)).

`evaluate_graph` is the recommended entry point for one-off
single-graph evaluation. The next section lifts it to a whole corpus.
The lower-level primitives further below (`graph_multiply`,
`graph_convolve`, `graph_attach`, `graph_from_tw2`,
`graph_zeta_general`) remain available for users assembling graph
zetas explicitly through the algebra or building higher-level pipelines
on top of them.

## Corpus front-ends: `evaluate_corpus` and `compute_series_coefficients`

The `gzl.series` module evaluates the corpus coefficient
$c_r(\boldsymbol{k})$, the sum over a graph corpus
$\mathcal{C_r}$ at perturbative order $r$ defined in
[Evaluating a corpus](README.md#evaluating-a-corpus) in the README, through
`compute_series_coefficients`, with `evaluate_corpus` exposing the
per-graph values it sums, plus a CLI driver. With `momentum=None` a
0qp corpus returns the $\boldsymbol{k} = \boldsymbol{0}$ scalar per
order, and a 1qp corpus returns the full Brillouin-zone dispersion in a
single pass. An explicit `momentum` selects one $\boldsymbol{k}$, a
batch of them, or, via `np.zeros(d)`, the $\boldsymbol{k} = \boldsymbol{0}$
scalar. Per-graph evaluation is delegated to `evaluate_graph` (the
three-step router above). A graph that `evaluate_graph` refuses stops
the pass with the same exception, and a note on the exception names the
graph and its order.

### Library API

```python
import numpy as np
from gzl import data_path
from gzl.series import compute_series_coefficients

CORPUS = data_path("tfim_softcore_corpus_1qp.npz")   # ships with the package
A  = "chain"                    # 1D integer chain
nu = 3.0                        # σ = 2 in d = 1 (ν = d + σ)

# 1qp corpus with momentum=None → the full Brillouin-zone dispersion in one
# pass: each order maps to a (n_points,)^d array, c_r(k) at every grid k.
# One ν was asked for, so there is no sweep axis; nu=[3.0, 3.5] prepends one.
disp = compute_series_coefficients(CORPUS, nu=nu, A=A, n_points=64, order_max=11)
print(disp[11].shape)           # (64,)   64 k-points
print(disp[11][ 0])             # k = 0 : -62.6689826604
print(disp[11][32])             # k = π : 186.1767942461

# A single k (fractional BZ coord 0.5 = k = π in d = 1) → one number per order,
# reproducing the matching grid cell to ~1e-11:
at_pi = compute_series_coefficients(CORPUS, nu=nu, A=A, n_points=64,
                                    order_max=11, momentum=0.5)
print(at_pi[11])                # 186.1767942463
# momentum=np.zeros(1) → the k = 0 scalar; momentum=(N, d) ndarray → a batch
# of N k-vectors (each order then has trailing shape (N,)).
```

`compute_series_coefficients` accepts a scalar or 1-D array `nu` (one
coefficient array per ν), or an `Interaction` / a list of them (a sweep
of couplings, one coefficient array per interaction, described in
[General interactions](#general-interactions)), the `momentum`
selector above, an `order_max` cap, `progress`, and an optional
`return_diagnostics`. `fast_cycles` and the remaining method settings
are passed on to `evaluate_graph` and are provisional, see
[API stability](#api-stability). The settings `nu_tensor_threshold`,
`tw_threshold`, `sigma_max`, `high_tw_fallback`, `direct_sum_L_list`
and `direct_sum_K` belonged to a whole-graph
$\boldsymbol{k} = \boldsymbol{0}$ fallback for the graphs
`evaluate_graph` refuses. The fallback has been removed, and they are
deprecated, have no effect and emit a `DeprecationWarning` when given.
The corpus path can be a consolidated `.npz` (flat / CSR layout) or `.h5`
(per-graph subgroup layout). See `tools/consolidate_corpus.py` for
producing one from a per-order H5 set. The two TFIM corpora ship with the package as
`data_path("tfim_softcore_corpus_0qp.npz")` and
`data_path("tfim_softcore_corpus_1qp.npz")`.

### Per-graph values: `evaluate_corpus`

For the per-graph values themselves, the lattice embedding factors
$\zeta_{(G,\mathcal{K_\nu})}$ before the prefactor-weighted sum,
`evaluate_corpus` runs the same pass and returns one record per
$(\text{order}, G, \nu)$:

```python
import numpy as np
from gzl import data_path, evaluate_corpus

CORPUS_0QP = data_path("tfim_softcore_corpus_0qp.npz")

# At nu=np.inf the values are the exact nearest-neighbour embedding
# factors (integer homomorphism counts) on the square lattice.
records = evaluate_corpus(CORPUS_0QP, nu=np.inf, A="square", n_points=0)
# records[0] == {'order': 2, 'graph_id': 'graph_0', 'nu': inf, 'nu_index': 0,
#                'value': 4.0,          # raw ζ_G = this graph's NN embedding number (here, coordination 4)
#                'prefactor': -0.25, 'contribution': -1.0,   # a_Γ · ζ_G
#                'route': 'topology_analytic', 's': 0, 't': 0}
```

Each record carries the raw `value` ($\zeta_G$) and the weighted
`contribution` ($a_r(G)\zeta_G$). Grouping the records by
$(\text{order}, \nu)$ and summing `contribution` reproduces
`compute_series_coefficients` exactly, and the two share one iteration
and one block cache. For an `Interaction` sweep `nu` is the tail exponent
and each record additionally carries `interaction` (the object) and
`nu_label`. A plain `Interaction(b=[1], nu=[ν])` keeps its identity and
label in the records and CSVs but is routed as the float ν,
bit-identical to a pass with the float ν. The route counts of
`return_diagnostics` (and the CLI's diagnostics CSV) are recorded for
the first sweep point only. In a mixed sweep other points may take
other routes, since a compact part lifts grids and changes arms, and a
purely compact kernel runs in nearest-neighbour mode. Those are not
counted. Per-point routes are in the records `evaluate_corpus` returns.
CLI: `gzl series --per-graph` (add `--nu inf` for the
embedding-factor dump).

### CLI driver

```bash
gzl series \
    --config examples/series_1qp_sigma2_d1.toml \
    --progress
```

writes a long-format CSV (`order, nu, coefficient`) plus a per-order
diagnostics CSV. An `[interaction]` table (keys `label`, `b`, `nu`, and
one of `shells = [[distance, coupling], ...]`, `compact = [[m_1, ..., m_d,
value], ...]` or `compact_npz`), or an `[[interactions]]` array of such
tables for a sweep, replaces the `nu*` keys. The CSV then carries the
tail exponent in `nu` and adds an `interaction` (label) column after
it, so numeric readers keep working. Finite k is selected with `--momentum` (JSON in
fractional BZ coordinates: a scalar `0.5`, a vector `[0.25, 0.5]`, or a
batch `[[0.0], [0.5]]`) or the equivalent `[evaluation] momentum` TOML
key. For a momentum grid or batch the CSV gains
`k_index, k_frac_0[, k_frac_1, …]` columns before `coefficient` (the
scalar / k = 0 case keeps the three-column form). Two example configs,
`series_0qp_d1.toml` and `series_1qp_sigma2_d1.toml`, are in `examples/`.
Flags override the TOML. The full set is in `gzl series --help`.
`python -m gzl.series` remains an alias for the same command.

The banner and `--progress` go to standard error and the CSV to the file
named by `--output`, so `--output -` puts the coefficients on standard
output and the run can be piped. `-q` silences the banner, the closing
summary and `--progress`. `--dry-run` prints every setting the pass would
be given, after flags have overridden the config and the config the
defaults, and exits without computing. The exit code is 0 when the pass
ran, 2 when the command was used wrongly, and 1 when the pass could not
finish. A failure prints one line rather than a stack trace, followed
by the corpus graph the pass stopped at when `evaluate_graph` refused
one, and `--traceback` (or `GZL_TRACEBACK=1`) restores the stack trace.

## From the algebraic primitives, a 4-cycle on the 2D triangular lattice

The library composes graph zetas through three primitive operations
on graph zeta objects, each with a clean identity in
$\boldsymbol{k}$-space.

`graph_multiply(g_1, g_2)` is the pointwise product in
$\boldsymbol{k}$-space:

```math
\zeta_{g_1 \cdot g_2}(\boldsymbol{k}) \;=\; \zeta_{g_1}(\boldsymbol{k})\,\zeta_{g_2}(\boldsymbol{k}).
```

In real space this is the serial composition: identify $t_1$
with $s_2$ and sum over the shared internal vertex, producing a
chain. For a single Epstein-zeta edge,
`graph_multiply_power(g, n)` returns an $n$-edge serial chain (a
path with $n + 1$ vertices and $n$ edges, terminals at the ends).

`graph_convolve(g_1, g_2)` is the convolution over the Brillouin
zone $\mathrm{BZ} = A^{-T}\mathbb{T}^d$:

```math
\zeta_{g_1 \star g_2}(\boldsymbol{k}) \;=\; \bigl(\zeta_{g_1} \ast \zeta_{g_2}\bigr)(\boldsymbol{k}) \;=\; V_\Lambda\, \int_{\mathrm{BZ}} \zeta_{g_1}(\boldsymbol{q})\, \zeta_{g_2}(\boldsymbol{k} - \boldsymbol{q})\,\mathrm{d}\boldsymbol{q}, \qquad V_\Lambda = \lvert\det A\rvert.
```

In real space this is the parallel composition: identify both
terminal pairs $(s_1, t_1) = (s_2, t_2)$ and let the two subgraphs
run between them in parallel. For a single edge,
`graph_convolve_power(g, n)` returns a bundle of $n$ parallel
edges.

`graph_attach(g, d)` is the 1-sum decoration:

```math
\zeta_{g \rtimes d}(\boldsymbol{k}) \;=\; \zeta_{g}(\boldsymbol{k})\,\zeta_{d}(\boldsymbol{0}).
```

A single vertex of the decoration $d$ is glued to a single vertex
of the host $g$. Only $\zeta_d$ at zero momentum survives, so the
decoration enters as a scalar prefactor that preserves the
$\boldsymbol{k}$-dependence of the host. Useful for treewidth-2
off-spine pieces or any vertex-glued addendum.

A closed $n$-cycle is therefore an $(n - 1)$-edge chain closed by one
parallel edge. The example below builds a 4-cycle on the 2D
triangular lattice, attaches a 3-cycle decoration to it through
`graph_attach`, and checks both stages against `zeta_circle`, the
high-precision adaptive-integration reference for cycle graphs:

```python
import numpy as np
from gzl import (
    make_epstein_graph,
    graph_multiply_power,
    graph_convolve,
    graph_attach,
    graph_zero,
    zeta_circle,
)

# --- 2D triangular lattice --------------------------------------------------
# The columns of A are the primitive vectors:
#   a_1 = (1, 0)
#   a_2 = (1/2, sqrt(3)/2)
A = np.array([
    [1.0, 0.5             ],
    [0.0, np.sqrt(3) / 2.0],
])

# Long-range exponent.  An irrational shift keeps us off the discrete
# Gamma-poles of the multiplication prefactor at nu = d + 2n.
nu       = 3.0 + np.pi / 30
n_points = 64                   # 2D grid: n_points^d = 64**2 = 4096 points

# --- 4-cycle: 3-edge serial chain closed by one parallel edge --------------
g        = make_epstein_graph(nu, A, n_points)   # single edge 1/|x|^nu
g_chain  = graph_multiply_power(g, 3)            # 3-edge serial chain s - a - b - t
g_4cyc   = graph_convolve(g_chain, g)            # close with one parallel edge s - t

val_4    = float(np.real(graph_zero(g_4cyc)))
ref_4    = float(zeta_circle(np.full(4, nu), A).real)
print(f"graph_zero(4-cycle) = {val_4:.10e}")    # 1.9190560...e+02
print(f"zeta_circle ref     = {ref_4:.10e}")
print(f"relative error      = {abs(val_4 - ref_4) / abs(ref_4):.3e}")  # ~1.7e-07

# --- 3-cycle decoration: 2-edge serial chain closed by one parallel edge ---
g_chain2 = graph_multiply_power(g, 2)            # 2-edge serial chain
g_3cyc   = graph_convolve(g_chain2, g)           # 3-cycle: 2-chain ∥ 1-edge

# --- attach the 3-cycle as a 1-sum decoration to the 4-cycle ---------------
g_dec    = graph_attach(g_4cyc, g_3cyc)          # ζ_dec(k) = ζ_4cyc(k) · ζ_3cyc(0)

val_dec  = float(np.real(graph_zero(g_dec)))
ref_dec  = ref_4 * float(zeta_circle(np.full(3, nu), A).real)
print(f"graph_zero(decorated 4-cycle) = {val_dec:.10e}")  # 4.6670225...e+03
print(f"ζ_4cyc · ζ_3cyc reference     = {ref_dec:.10e}")
print(f"relative error                = {abs(val_dec - ref_dec) / abs(ref_dec):.3e}")  # ~2.1e-07
```

The 4-cycle result converges as $\mathcal{O}(n_\text{points}^{d - \nu})$:
`n_points = 16` gives ~2e-4, `n_points = 64` gives ~2e-7,
`n_points = 128` gives ~5e-9. Pick `n_points` so that the truncation
residual sits comfortably below the precision your application needs.

The same 4-cycle can be assembled by other routes, for instance as
`graph_convolve_power(graph_multiply_power(g, 2), 2)`, two parallel
copies of a 2-edge chain, and the algebra is associative and
order-independent at the level of the exact graph zeta. Differences
between routes (well below `n_points` truncation here) come from the
analytic-Epstein-vs-Fourier split controlled by `sigma_max`, described
below.

> **Tip.** For graphs whose topology is naturally written as a
> `networkx.MultiGraph` (terminals + per-edge `nu` attributes), reach
> for the high-level constructors `graph_from_sp_uniform` /
> `graph_from_tw2_uniform` instead. They accept the multigraph
> directly and decompose it into the same primitives, with
> `graph_attach` taking care of every off-spine decoration. The next
> section shows that idiom.

## From a NetworkX MultiGraph (with exact-reference comparison)

For series-parallel and treewidth-2 topologies you can hand the graph to
`graph_from_tw2_uniform` (or `graph_from_tw2` for non-uniform exponents) as a
`networkx.MultiGraph` with two terminals and per-edge `nu` attributes. The
constructor block-cut-decomposes the input, reduces the spine block by
iterated series-parallel reduction, and folds in every off-spine decoration
through its value at $\boldsymbol{k} = \boldsymbol{0}$.

The example below builds a composite topology that exercises all three
ingredients of a treewidth-2 graph on the 1D integer lattice:

* a main 9-node cycle between the two terminals $s$ and $t$ (the spine
  block),
* a 3-node cycle decoration attached at one interior vertex of the main
  cycle, and
* a 3-node tree decoration (a 2-edge path) attached at one vertex of
  the small cycle.

For a graph of this shape the value at $\boldsymbol{k} = \boldsymbol{0}$
factorises into a product of three independent contributions: the circle
zeta of the main 9-cycle, the circle zeta of the 3-cycle decoration, and an Epstein zeta function raised to the power of the number of edges in
the tree decoration (here squared, since the tree has two edges). Each
off-spine decoration is summed over its own free vertices once the cut
vertex is fixed, so the small cycle contributes its own circle zeta and the
2-edge tree contributes one Epstein-zeta factor per edge. We compute the
three high-precision constants once (via `zeta_circle` and `epsteinlib`) and
hard-code them so the example is self-contained.

```python
import numpy as np
import networkx as nx
from gzl import graph_from_tw2_uniform, graph_zero

# --- problem setup ---------------------------------------------------------
nu       = 1.5 + np.pi / 30        # irrational shift dodges Gamma poles
A        = np.array([[1.0]])       # 1D integer lattice
n_points = 500

# Reference constants obtained once on this lattice with the same nu:
#   zeta_circle(np.full(9, nu), A).real
#   zeta_circle(np.full(3, nu), A).real
#   epstein_zeta(nu, A, np.zeros(1), np.zeros(1)).real
ZETA_CIRCLE_9 = 9.052310003079856e+03
ZETA_CIRCLE_3 = 3.375583050373807e+00
ZETA_EPSTEIN  = 4.546144606627310e+00
ZETA_REF      = ZETA_CIRCLE_9 * ZETA_CIRCLE_3 * ZETA_EPSTEIN ** 2  # ~ 6.315e5

# --- build the composite multigraph ----------------------------------------
G = nx.MultiGraph()

# 9-cycle on the spine: s - m1 - m2 - ... - m7 - t, plus the closing edge s-t
spine = ["s", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "t"]
for u, v in zip(spine, spine[1:]):
    G.add_edge(u, v, nu=nu)
G.add_edge("s", "t", nu=nu)        # closes the 9-cycle

# 3-cycle decoration attached at the interior vertex m4
G.add_edge("m4", "c1", nu=nu)
G.add_edge("c1", "c2", nu=nu)
G.add_edge("c2", "m4", nu=nu)

# 3-node tree decoration (path c1 - t1 - t2) attached at c1
G.add_edge("c1", "t1", nu=nu)
G.add_edge("t1", "t2", nu=nu)

# --- convert and evaluate at k=0 ------------------------------------------
g = graph_from_tw2_uniform(G, "s", "t", nu, A, n_points)
val = float(np.real(graph_zero(g)))

print(f"graph_zero(g)  = {val:.9e}")
print(f"reference      = {ZETA_REF:.9e}")
print(f"relative error = {abs(val - ZETA_REF) / ZETA_REF:.3e}")
# graph_zero(g)  = 6.315309919e+05
# reference      = 6.315310494e+05
# relative error = ~9.1e-08
```

`zeta_circle` itself is exposed in the public API and is useful for sanity-
checking new graphs in $d = 1, 2, 3$:

```python
from gzl import zeta_circle
ref_9 = zeta_circle(np.full(9, nu), A)
```

## Treewidth-agnostic tensor-network evaluation

The series-parallel and treewidth-2 constructors above succeed on every
graph whose underlying simple graph has treewidth $\le 2$.  For
graphs with $K_4$ minors (treewidth $\ge 3$), such as $K_4$ itself,
prisms, $K_{3,3}$ and dense expansions at high perturbative order, they
raise `NotTreewidthTwoError`.  GZL provides a complementary
treewidth-agnostic evaluator `graph_zeta_general` that walks any
multigraph by bucket elimination on a torus of
$n_\text{points}^d$ sites per vertex axis, and supports arbitrary
external-momentum dependence via a list of free terminal vertices.

### Scalar at zero momentum

The simplest entry point computes $\zeta_g(\boldsymbol{0}, \dots, \boldsymbol{0})$
, the graph zeta at all-zero external momenta, and is itself a
thin wrapper:

```python
from gzl import graph_zeta_general_at_zero

# K_4 on the 1D integer chain, treewidth 3
edges  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
nu_vec = np.full(6, 4.0)                 # nu = 4, sigma = 3
A      = "chain"
val    = graph_zeta_general_at_zero(edges, nu_vec, A, n_points=32)
```

### Free terminals on the Brillouin-zone grid

For non-zero external momenta, the 1qp single-particle propagator
and beyond, pass the free terminal vertices as `terminals=...`.
The algorithm pins the `source` vertex at the origin and keeps the
terminal axes open through the elimination, so the resulting
tensor depends on the terminal positions (`space='z'`) or, after an
inverse-FFT, on the terminal momenta (`space='k'`):

```python
from gzl import graph_zeta_general

# 1qp triangle: source = 0, free terminal = 1, internal = 2
edges  = np.array([[0,1],[0,2],[1,2]])
nu_vec = np.full(3, 4.0)
A      = "chain"

# zeta_g on the full Brillouin-zone grid for the t = 1 terminal:
zeta_k = graph_zeta_general(
    edges, nu_vec, A, n_points=32,
    source=0, terminals=(1,), space='k',
)              # shape (32,) complex; agrees with graph_sample to round-off
```

For two free terminals (a 3-terminal graph) the output is a 2-axis
tensor, and so on.  The cost grows by an extra factor of
$n_\text{points}^d$ per free terminal, which is manageable in $d = 1$ and
fine for one or two terminals in $d = 2$.

### Algorithm and cost

Each edge $\lbrace u, v \rbrace$ with exponent $\nu_e$ contributes a 2-axis
analytic kernel

```math
T_{uv}[z_u, z_v] = \frac{1}{\bigl\lvert A\,(z_u - z_v)_{\text{periodic}}\bigr\rvert^{\,\nu_e}}
```

evaluated at integer lattice points, with no FFT and no aliasing introduced
by Fourier-extracting the kernel.  Bucket elimination contracts away
each non-source, non-terminal vertex in the order the planner chooses,
at a cost of at most

```math
\mathcal{O}\!\bigl(V \cdot n_\text{points}^{(\tau + 1 + N_t)\,d}\bigr),
```

where $\tau$ is the treewidth, $V$ the vertex count, and $N_t$ the
number of free terminals (the source is pinned, so doesn't count).

**FFT elimination fast path.**  An elimination step whose bucket
isolates one original edge kernel toward a partner vertex (the
partner appears in the scope of exactly one bucket factor) is a
cyclic convolution along the eliminated axis and is contracted by
real FFT.  That step costs

```math
\mathcal{O}\!\bigl(n_\text{points}^{(\lvert C\rvert - 1) d} \log n_\text{points}\bigr)
\quad\text{instead of}\quad
\mathcal{O}\!\bigl(n_\text{points}^{\lvert C\rvert d}\bigr)
```

for a bag $C$, one full power of $n_\text{points}^d$ less, exactly
(identical to the dense contraction up to round-off, and steps that fail
the condition run dense).  On $K_5$-type cores this removes one
power from the *peak* step, e.g. measured 29x at $d = 1$,
$n_\text{points} = 128$ and growing with $n_\text{points}$.  The
same fast path accelerates the real-space `direct_sum` engines in
its zero-padded Toeplitz form, where the padding costs a factor
$2^d$ in transform length, so that step is gated on a measured cost
model and small boxes stay on the dense branch.

### What this gives you

* Treewidth-agnostic evaluation that works on any topology including
  the $K_4$-minor graphs the SP / treewidth-2 constructors can't
  reach.
* Drop-in replacement for `graph_sample(graph_compress(...))` on
  treewidth $\le 2$ inputs, in agreement to round-off in our validation
  suite.
* External-momentum dependence on every free terminal, with the
  result delivered either in real-space (`space='z'`) or
  momentum-space (`space='k'`) representation.

Currently supports $d \ge 1$ and runs at $\sigma_\text{max} = 0$
(pure Fourier-coefficient representation, and at $d = 3$ the edge-difference
table caps practical use at $n_\text{points} \le 16$).  Validation against the
existing `graph_from_edges_uniform` at `sigma_max=0.0` agrees to
round-off for treewidth $\le 2$, and against `direct_sum_extrapolated` for
treewidth $> 2$ the agreement is at the level of the truncation
residuals of either method (~1e-5 at $n_\text{points} = 32$ in $d=1$
on the TFIM softcore graph corpus, with details in
`tests/test_tensor_network.py`).

The full Γ-prefactor multiplication / convolution algebra used by
`graph_multiply` and `graph_convolve` at $\sigma_\text{max} > 0$
extends to N-terminal tensors and would give analytically-exact
contractions on this path. That extension is the natural follow-up
work.

## Hybrid engine (default): FFT reduction + dense core

`graph_zeta_general` contracts *every* vertex densely, which is more than
most blocks need.  A block is usually mostly series and parallel
structure, chains and parallel bundles, wrapped around a small
irreducible core.  By translation invariance every edge kernel is
circulant, so the convolution theorem collapses each series step (a
degree-2 vertex) and each parallel bundle with a single FFT, leaving only
the 3-connected core to contract densely.  The whole Brillouin-zone grid
is then read off with one final `fftn`.

This is the `hybrid_zeta` engine, and it is the form of the torus
tensor network used by `evaluate_graph` and
`compute_series_coefficients`. Pass `engine="tensor"` to fall back to
the bucket-elimination tensor:

```python
from gzl import evaluate_graph

edges  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])   # K_4
A      = "chain"
grid_h = evaluate_graph(edges, 2.5, A, source=0, terminal=2, n_points=64)
grid_t = evaluate_graph(edges, 2.5, A, source=0, terminal=2,
                        n_points=64, engine="tensor")        # opt out
# grid_h == grid_t to floating-point round-off
```

Called directly with the same arguments, `hybrid_zeta` and
`graph_zeta_general` agree to round-off, since both evaluate the same
torus truncation. Inside `evaluate_graph`, the hybrid engine can in
addition run the core of a block on a different grid than its series
and parallel parts, which the tensor cannot. Where this applies,
`engine="tensor"` evaluates the whole block on one grid and does not
reproduce the default. On the 1qp TFIM series on the chain at
$\nu = 3$ and $n_\text{points} = 64$, the two agree to round-off up to
order 6 and differ by up to $2 \cdot 10^{-4}$ of the coefficient scale
at order 11.

The hybrid engine falls back to the tensor for $\nu = \infty$ and for
purely compact interactions, for two or more free terminals, and for a
core it cannot contract for structural reasons. A core above the memory
budget raises `HybridCoreTooLargeError` instead, and the router then
evaluates a dense block with the box.

## Comparison with Monte Carlo

`benchmarks/mc_comparison.py` compares the series coefficients of both
shipped corpora with the Monte Carlo series in `gzl/data` on
the chain, square, triangular and cubic lattices. It reproduces the
comparison in Sec. 8.3 of
[arXiv:2609.18918](https://arxiv.org/abs/2609.18918). Its options and
run time are listed in [`benchmarks/README.md`](benchmarks/README.md).

## Shipped reference data

The package carries, in `gzl/data/`, everything needed to evaluate
the TFIM corpora and to compare against Monte Carlo series
without a source checkout:

| file | contents |
|---|---|
| `tfim_softcore_corpus_0qp.npz`, `tfim_softcore_corpus_1qp.npz` | the softcore TFIM corpora (8 403 and 22 677 combinatorial graphs) |
| `full_graph_topologies.npz` | topology snapshot of both corpora, 31 080 graphs |
| `MC_patched/TFIM/<lattice>/` | Monte Carlo series coefficients for chain, square, triangular and cubic lattices |

```python
from gzl import compute_series_coefficients, data_path

corpus = data_path("tfim_softcore_corpus_1qp.npz")
c = compute_series_coefficients(corpus, 2.0, "chain", 64, order_max=6)
mc = data_path("MC_patched/TFIM/chain/1qp_gap_series_1d_chain_tfim_k0_sigma1.0_order11.csv")
```

The corpora and the topology snapshot are part of gzl. The Monte
Carlo series are third-party research data,
redistributed with attribution on their sources' terms. They are not
covered by the AGPL. Every file is listed with its source, the evidence for
that source and its SHA-256 in `gzl/data/PROVENANCE.csv`, and the
sources, which are Fey, Kapfer and Schmidt (2019), Fey (2020), Langheld
et al. (2022) with its Zenodo data record, and P. Adelhardt (private
communication), are named in [`NOTICE`](./NOTICE). Please cite the source of the files you
use.

## API stability

GZL follows [Semantic Versioning](https://semver.org/) for the stable
part of its API, which is listed below. A program that uses only this
part and works with version 1.0.0 runs unchanged with every later
version 1.x, and changes that would break it are reserved for a new
major version. The provisional part is documented as well, but it may
change in a minor release, and every such change is recorded in the
[changelog](CHANGELOG.md). All other names are internal. A name belongs
to the API only as it is imported from `gzl` itself, as in
`from gzl import evaluate_graph`, and module paths such as
`gzl.frontend` are internal.

### Stable

- The front-ends `evaluate_graph`, `evaluate_corpus` and
  `compute_series_coefficients`, the closed form `zeta_circle`, the
  helpers `data_path` and `nu_grid`, and the version string
  `gzl.__version__`.
- The base exception `GraphZetaError`, which is a `ValueError`, and its
  subclasses `SelfLoopError`, `DisconnectedGraphError`,
  `VertexOutOfRangeError`, `NPointsRequiredError`,
  `TopologyEvaluatorUnavailableError`, `UnsupportedLatticeSumError` and
  `UnsupportedRequestError`. The last one is raised for a request that
  no evaluator implements, such as two free terminals, and is also a
  `NotImplementedError`. Any other subclass these functions raise is
  internal and is caught as a `GraphZetaError`. A malformed argument,
  such as a NaN momentum, a singular lattice matrix or a fractional
  vertex label, raises a plain `ValueError` or `TypeError`.
- The lattice names `"chain"`, `"square"`, `"triangular"` and
  `"cubic"`, the corpus names `"tfim0qp"` and `"tfim1qp"`, and the
  corpus file format.
- The commands `gzl series`, `gzl selftest` and `gzl info`, the options
  and configuration keys of `gzl series`, the columns of the coefficient
  and per-graph CSV files it writes, and its exit codes. The text that
  `gzl selftest` and `gzl info` print is meant for reading.

Among the parameters of the stable functions, those describing the
problem are stable, namely the graph or corpus, `nu`, `A`, `source`,
`terminal`, `momentum`, `n_points` and `order_max`, together with
`return_diagnostics`, `progress` and every parameter of `zeta_circle`,
`data_path` and `nu_grid`. The parameters that select or tune
the numerical method are provisional. These are `richardson`,
`fast_cycles`, `accuracy`, `block_cache`, `engine`, `dense_engine`,
`sp_n_points`, `core_grading`, `nu_tensor_threshold`, `tw_threshold`,
`sigma_max`, `high_tw_fallback`, `direct_sum_L_list` and
`direct_sum_K`, together with the corresponding options and
configuration keys of `gzl series`. Of these, `nu_tensor_threshold`,
`tw_threshold`, `sigma_max`, `high_tw_fallback`, `direct_sum_L_list`
and `direct_sum_K` are deprecated and have no effect. The same holds
for the content of the diagnostics, including the diagnostics file of
`gzl series`, and for the value of `route` in the records of
`evaluate_corpus`, which serve inspection only.

### Provisional

- The class `Interaction` for general interaction kernels, and
  `InteractionSupportError`.
- The semi-analytical algebra, consisting of the class `GraphZeta`, the
  constructors `make_epstein_graph` and `make_graph_obj`, the operations
  `graph_multiply`, `graph_multiply_power`, `graph_convolve`,
  `graph_convolve_power`, `graph_attach` and `graph_compress`, the
  evaluations `graph_sample`, `graph_sample_at` and `graph_zero`, the
  helpers `cNu` and `periodic_convolve_nd`, and
  `PrefactorSingularityError`. The operations keep their mathematical
  meaning, but the representation of a `GraphZeta` may change when
  anisotropic Epstein zeta functions are included.
- The graph constructors `graph_from_sp`, `graph_from_sp_uniform`,
  `graph_from_tw2`, `graph_from_tw2_uniform`, `graph_from_edges` and
  `graph_from_edges_uniform`, and the exceptions
  `NotSeriesParallelError` and `NotTreewidthTwoError`.
- The engines `graph_zeta_general`, `graph_zeta_general_at_zero`,
  `hybrid_zeta`, `slab_zeta`, `direct_sum_zero_momentum` and
  `direct_sum_extrapolated`, and the exceptions
  `HybridCoreTooLargeError` and `SlabTooLargeError`.

A provisional name may become stable in a minor release. A stable name
is removed only in a new major version, after a minor release in which
it issues a deprecation warning. The package alias `graph_zeta` follows
this rule and will be removed in version 2.0.

### Numerical values

GZL evaluates graph zeta functions on finite grids, and most values
carry a truncation error that decreases with `n_points`. The meaning of
a value is stable. It is the graph zeta function defined in the
[README](README.md#definition-of-the-graph-zeta-function), with momenta
in fractional coordinates of the reciprocal lattice and `n_points` grid
points per dimension. The digits within the truncation error are not
stable. A new version may improve the evaluation and thereby move
values within the truncation error, and the changelog records every
such change together with its size. Closed forms, such as those of
bridges and cycles, and the values checked by `gzl selftest` change only
when an error is corrected. A published number is reproduced by
installing the version it was computed with, as in
`pip install gzl==1.0.0`. A value can also depend on the form of the
request. A momentum requested on its own and the same momentum read
from the full Brillouin-zone grid, zero momentum included, agree within
the truncation error, but not necessarily in the last digits. Within
one version, results on different machines agree up to the last few
binary digits, which depend on the builds of NumPy and of the
linear-algebra libraries installed there.

## Core API

Wherever a lattice matrix `A` is expected, it can be replaced by one of
the names `"chain"`, `"square"`, `"triangular"`, and `"cubic"`. The
chain, square, and cubic lattices are ℤ, ℤ², and ℤ³, and the triangular
lattice is spanned by the primitive vectors (1, 0) and (1/2, √3/2).

| Function | Signature and description |
|---|---|
| `make_epstein_graph` | `(nu, A, n_samples)`, elementary graph with a single Epstein-zeta singularity of exponent $\nu$ on lattice $A$, discretised on `n_samples` points per dimension. |
| `graph_multiply` | `(g1, g2, sigma_max=4.0)`, pointwise product. `sigma_max` sets the compression threshold (see below). |
| `graph_multiply_power` | `(g, n_exp, sigma_max=4.0)`, $n$-th pointwise power under `graph_multiply`, with $n =$ `n_exp`. For a single-edge Epstein graph this is a serial chain of $n$ edges. $n$ must be a positive integer. |
| `graph_convolve` | `(g1, g2, sigma_max=4.0)`, periodic convolution. `sigma_max` sets the compression threshold (see below). |
| `graph_convolve_power` | `(g, n_exp, sigma_max=4.0)`, $n$-fold convolution power, with $n =$ `n_exp`. For a single-edge Epstein graph this is a parallel graph of $n$ edges. |
| `graph_attach` | `(g, decoration)`, 1-sum attachment: returns $\zeta_g(\boldsymbol{k})\zeta_\text{decoration}(\boldsymbol{0})$. The `decoration` is glued to `g` at a single vertex and contributes only its value at $\boldsymbol{k} = \boldsymbol{0}$ (its own terminals are forgotten). Host and decoration must share $A$ but may use different grids. |
| `graph_from_sp` | `(multigraph, s, t, A, n_points, sigma_max=4.0)`, strict constructor: builds a `GraphZeta` from a 2-connected series-parallel `networkx.MultiGraph` with two terminals and per-edge `nu` attributes, via iterated series-parallel reduction. Raises `NotSeriesParallelError` for $K_4$ minors or articulation points. |
| `graph_from_sp_uniform` | `(multigraph, s, t, nu, A, n_points, sigma_max=4.0)`, uniform-ν convenience wrapper around `graph_from_sp`. |
| `graph_from_tw2` | `(multigraph, s, t, A, n_points, sigma_max=4.0)`, general constructor for arbitrary treewidth-2 multigraphs. Block-cut-decomposes the input, reduces each spine block with `graph_from_sp`, and multiplies in every off-spine decoration's value at $\boldsymbol{k} = \boldsymbol{0}$ via `graph_attach` (recursively handling nested decorations). Raises `NotTreewidthTwoError` (a subclass of `NotSeriesParallelError`) on $K_4$ minors. |
| `graph_from_tw2_uniform` | `(multigraph, s, t, nu, A, n_points, sigma_max=4.0)`, uniform-ν convenience wrapper around `graph_from_tw2`. |
| `graph_compress` | `(g, sigma_max)`, absorb exponents with $\nu > d + \sigma_\text{max}$ into the Fourier part. |
| `graph_sample` | `(g)`, evaluate on an equidistant grid over $[0, 1)^d$. |
| `graph_zero` | `(g)`, value at $\boldsymbol{k} = \boldsymbol{0}$ of a built `GraphZeta` algebra object (a low-level operation, and for end-to-end evaluation from a raw edge list use `evaluate_graph`). |
| `graph_sample_at` | `(g, k_frac)`, single-k evaluation of a built `GraphZeta` at fractional momentum `k_frac` $\in [0, 1)^d$.  Same value convention as `graph_sample`, so at on-grid k the two agree, and at off-grid k the balanced-z layout preserves lattice inversion symmetry.  Cost $\mathcal{O}(N_\text{singular} + n^d)$ vs $\mathcal{O}(N_\text{singular} \cdot n^d)$ for the full-grid `graph_sample`. |
| `evaluate_graph` | `(graph, nu=None, A=None, *, source=None, terminal=None, momentum=None, n_points=0, return_diagnostics=False, ...)`, **single-graph topology-first front-end.**  Accepts either a flat edge list (`evaluate_graph(edges_flat, nu, A, ...)`) or a `networkx.MultiGraph` with `nu` edge attributes (`evaluate_graph(multigraph, A, ...)`). `nu` may be a float, a per-edge array, an `Interaction`, or a per-edge sequence mixing floats and `Interaction`s (a `MultiGraph` edge attribute likewise).  Hadamard-merges parallels, block-cut decomposes, and routes each block to the cheapest correct evaluator: `epstein_zeta` for bridges, `zeta_circle` for simple cycles, the semi-analytical algebra path for SP-reducible blocks at small σ, the torus tensor network (with optional 3-point Richardson) otherwise.  At finite external momentum the spine of the block-cut tree is identified and only on-spine blocks see the routed momentum, and off-spine 1-sum decorations contribute their k = 0 scalar.  `terminal` accepts an `int` or a length-≤ 1 array (length ≥ 2 raises `UnsupportedRequestError`, a `NotImplementedError`).  `momentum`: `None` ⇒ ζ_G(0) (vacuum) or full BZ grid `(n_points,)^d` (1qp), scalar / `(d,)` ndarray ⇒ ζ_G(k) at a single k, and 1-D / `(N, d)` ndarray ⇒ a batch over N k-vectors.  Returns are real `float` (vacuum / single-k) or real `ndarray` (grid / batch).  `graph_sample_at(g, k)` and `graph_zeta_general(..., momentum=k)` are the corresponding single-k helpers at the lower levels.  Vertex labels are names: the vertex set is the labels present in the edge list (a label in no edge is not a vertex, and a `source` or `terminal` that is no vertex raises). `source=None` pins the smallest present label, and a `MultiGraph` with an isolated node is refused, because its infinite-lattice sum diverges.  Structured exceptions (`SelfLoopError`, `DisconnectedGraphError`, `VertexOutOfRangeError`, `NPointsRequiredError`, `TopologyEvaluatorUnavailableError`, `UnsupportedLatticeSumError` and `UnsupportedRequestError`, all subclassing `GraphZetaError`) are raised for a request it cannot evaluate. |
| `zeta_circle` | `(nu_vec, A)`, reference of circle graph at $\boldsymbol{k} = \boldsymbol{0}$ via adaptive integration for $d = 1, 2, 3$. Another $d$ or an exponent `inf` raises `UnsupportedRequestError`, and `evaluate_graph` sends a cycle in $d \ge 4$ to the algebra or the tensor instead. Entries of `nu_vec` may be `Interaction`s (general per-edge kernels): the per-edge transforms then carry the compact parts as exact trigonometric polynomials and the fixed rule is refined and self-checked (see [General interactions](#general-interactions)). |
| `Interaction` | `Interaction(b=(), nu=(), compact=None, *, label=None)`, a general edge kernel $K(\boldsymbol{x}) = a(\boldsymbol{x}) + \sum_j b_j \mathcal{K_{\nu_j}}(\boldsymbol{x})$: power-law terms `b`, `nu` plus an even, compactly supported table `compact` over integer lattice labels. Constructors `power_law(nu, b=1.0)`, `from_table(table, *, b, nu)`, `from_function(K, A, radius, *, b, nu)` (a function of the displacement vector, sampled on $\lvert A\boldsymbol{m}\rvert \le$ `radius`, origin included, evenness checked), `from_total(K, A, radius, b, nu)`, `from_shells(A, {distance: coupling}, *, b, nu, total=True)`, `nearest_neighbour(A, J=1.0)`, `from_config(dict, A)`. Products `K1 * K2`, `K ** m` (the parallel-edge / multiplicity kernel), scalar multiples and sums. Also `tail_exponent`, `support_radius`, `key()`, `sample(labels, A, power=...)`, `lattice_sum(A, k_frac=None)` (the exact bridge value / dispersion). Accepted as `nu` by `evaluate_graph`, `zeta_circle`, `compute_series_coefficients`, `evaluate_corpus` and the CLI, and through the keyword `kernels=` (one per edge, beside the tail exponents in `nu_vec`) by `graph_zeta_general`, `hybrid_zeta`, `slab_zeta`, `direct_sum_*` and `graph_from_edges`. An `Interaction` placed in an engine's `nu_vec` raises a `ValueError` naming `kernels=`. A plain `Interaction(b=[1], nu=[ν])` is the float `ν`. `InteractionSupportError` (a `GraphZetaError`) is raised by a standalone engine call whose window or box rung cannot hold the compact support (the front-end lifts a mixed block's grid to $n_v R + 2$ instead, and runs a purely compact block on its smallest exact torus). |
| `direct_sum_zero_momentum` | `(edges, nu, A, L, ...)`, direct lattice-sum evaluation of $\zeta_g(\boldsymbol{0})$ via real-space bucket elimination over $[-L, L]^d$, treewidth-agnostic, supporting arbitrary multigraph topology and any $d \ge 1$. `evaluate_graph` uses the same box for the dense blocks (treewidth $> 2$) that neither the torus nor the slab evaluates. Elimination steps that isolate one original edge kernel run as zero-padded real-FFT Toeplitz contractions, one power of $(2L+1)^d$ less per eligible step, identical to the dense sum up to round-off. |
| `direct_sum_extrapolated` | `(edges, nu, A, ...)`, Richardson-extrapolated `direct_sum_zero_momentum` over a list of `L` values, converging at the leading-order Euler–Maclaurin rate set by `K` correction terms. |
| `slab_zeta` | `(edges, nu_vec, A, n, *, source=0, slab=None, use_symmetry=True, max_bytes=None)`, **vacuum evaluator with a second pin.**  Returns `(value, n_cells, exponent_inner)`.  Fixing one free vertex at an explicit lattice site turns its incident edges into one-axis potentials, so the inner post-peel exponent drops by one and an outer sum over that vertex's sites restores the exact value: total work is unchanged to leading order, peak memory is not.  This is the same torus truncation `hybrid_zeta` and `graph_zeta_general` compute at the same `n`, merely associated differently, so it is not an independent truncation and gives reach rather than a second opinion.  Since the fold it is a thin wrapper over the one dense core (`hybrid._dense_core` with a second pin, `pins=((source, 0), (slab, z))`). The frontend rides it for the measured split-no-op `(3, 2)` dense sub-class (evaluated at two rungs, shipped only on self-agreement) and for dense blocks its accuracy gate declines.  The outer loop folds onto the orbits of the signed permutations that preserve both the cell metric and the coordinate truncation window (48-fold on a cubic cell, 16 tetragonal, 8 orthorhombic, 2 generic), exact on any lattice, with the group derived by testing candidates against the kernel's own distance table rather than from a symmetry argument.  `max_bytes` refuses with `SlabTooLargeError` (a `MemoryError` subclass) before any table is allocated. |
| `graph_zeta_general` | `(edges_flat, nu_vec, A, n_points, *, source=None, terminals=(), space='z')`, **treewidth-agnostic Fourier-grid evaluator** for $\zeta_g$ at arbitrary external momenta on the listed terminal vertices. Bucket elimination on a torus of $n_\text{points}^d$ sites per axis, at a cost of at most $\mathcal{O}(V \cdot n_\text{points}^{(\tau+1+N_t)d})$ where $N_t = \mathrm{len(terminals)}$. With `terminals=()` it returns the scalar $\zeta_g(\boldsymbol{0}, \dots, \boldsymbol{0})$, and otherwise a tensor over terminal positions (`space='z'`) or momenta (`space='k'`, applies `ifftn * n^d` along terminal axes). Supports $d \ge 1$ at $\sigma_\text{max} = 0$, where the edge difference table is $(n^d) \times (n^d)$ `float64` at $\boldsymbol{k} = \boldsymbol{0}$, so at $d = 3$ the practical ceiling is $n_\text{points} \le 16$ (and $\text{tw} \le 2$ blocks). Agrees to round-off with `graph_sample(graph_compress(graph_from_edges_uniform(..., sigma_max=0.0), 0.0))` for treewidth $\le 2$ inputs, and works on any topology. Elimination steps that isolate one original circulant edge kernel run as cyclic real-FFT convolutions, one power of $n_\text{points}^d$ less per eligible step (see *Algorithm and cost* above). |
| `graph_zeta_general_at_zero` | `(edges_flat, nu_vec, A, n_points, *, pinned_vertex=None)`, convenience wrapper around `graph_zeta_general` with `terminals=()`, returning the zero-external-momentum scalar as a Python `float`. |
| `compute_series_coefficients` | `(corpus_path, nu, A, n_points, *, momentum=None, order_max=None, return_diagnostics=False, progress=False, ...)`, **per-order series-coefficient evaluator** $c_r(\boldsymbol{k}) = \sum_{G \in \mathcal{C_r}} a_r(G) \zeta_{(G,\mathcal{K_\nu})}(\boldsymbol{k})$ over a graph corpus.  `nu` is a float, a 1-D array (a sweep), an `Interaction`, or a list mixing the two (a sweep of couplings).  `momentum` (fractional BZ coordinates) selects the evaluation: `None` ⇒ $\boldsymbol{k} = \boldsymbol{0}$ scalar (0qp) or the full BZ grid `(n_points,)^d` (1qp), scalar / `(d,)` ⇒ one $\boldsymbol{k}$, and `(N, d)` ⇒ a batch. Finite $\boldsymbol{k}$ runs only through the topology evaluator.  Per-graph evaluation is delegated to `evaluate_graph` (see above for the exact per-block routing rule), and a graph it refuses (a self-loop, a disconnected graph, `n_points = 0` for a block that needs a grid) stops the pass with the same exception.  Returns `dict[int, ndarray]`, each of shape `trailing` for a single ν (a float, a NumPy scalar or one `Interaction`) and `(n_nu,) + trailing` for a sweep (any list or 1-D array, even of length one). The rule is the argument, not the resulting length, so `nu=3.0` and `nu=[3.0]` give the same numbers and differ only in shape.  CLI: `gzl series --config <toml>` (`--momentum` for finite k). |
| `evaluate_corpus` | `(corpus_path, nu, A, n_points, *, momentum=None, order_max=None, progress=False, ...)`, **per-graph corpus evaluator**: returns a `list` of records `{order, graph_id, nu, nu_index, value, prefactor, contribution, route, s, t}`, one per $(\text{order}, G, \nu)$, where `value` is the raw $\zeta_{(G,\mathcal{K_\nu})}$ (the lattice embedding factor, at `nu=np.inf` the exact nearest-neighbour homomorphism count) and `contribution` $= a_r(G)\zeta_G$.  This is the per-graph pass `compute_series_coefficients` sums: grouping by $(\text{order}, \nu)$ and summing `contribution` reproduces it bit-for-bit.  CLI: `gzl series --per-graph` (`--nu inf` for the embedding-factor dump). |
| `data_path` | `(name="")`, filesystem `Path` of a file or directory in the package's data directory (`"tfim_softcore_corpus_1qp.npz"`, `"MC_patched/TFIM/chain"`, and empty for the directory itself). Raises `ValueError` for a path that leaves the data directory and `FileNotFoundError` for one that does not ship. See [Shipped reference data](#shipped-reference-data). |
| `nu_grid` | `(start, end, step)`, `np.arange`-style helper to build a $\nu$-grid for `compute_series_coefficients`, inclusive of `end` when it lands on the step modulo float64 noise. |

### The `sigma_max` parameter

`sigma_max` controls how `graph_compress` splits a graph into its analytical
(Epstein-zeta) and numerical (Fourier) parts:

- Exponents with $\nu \le d + \sigma_\text{max}$ are kept analytically as
  Epstein-zeta contributions, which are essential for strong singularities that would
  otherwise alias on the discrete grid.
- Exponents with $\nu > d + \sigma_\text{max}$ are **absorbed into the Fourier
  coefficients `aMat`**, because $1/\lvert\boldsymbol{x}\rvert^\nu$ is mild enough to be
  represented faithfully on the sample grid.

Both `graph_multiply` and `graph_convolve` call `graph_compress` on their
inputs (and, in the case of `graph_multiply`, also on the result) before
performing the algebraic operation. The default value $\sigma_\text{max} = 4$
is a conservative choice that works well for typical lattice sums in
$d \le 3$.

**When to increase $\sigma_\text{max}$.** if you observe loss of precision in
the Fourier part (e.g. when `n_samples` is small or the graph contains
several mild singularities that together degrade grid accuracy). Keeping more
terms analytically is slower but in general more accurate. Values above $4$
are not recommended as they can lead to significant cancellation errors for
large numbers of operations.

**When to decrease $\sigma_\text{max}$.** First, to speed up repeated
operations on graphs with many exponents, at the cost of representing milder
singularities numerically. Second, to reduce cancellation error for very
long operation chains. Third, setting $\sigma_\text{max} = 0$ selects the
improved Fourier method (exactly known Fourier coefficients), which
sidesteps the `PrefactorSingularityError` entirely and is bit-equivalent
under associative reordering of the operations.

Note that `graph_multiply` handles the Γ-prefactor pole family
ν = d + 2n (n = 1, 2, …) automatically. Near-pole inputs are absorbed
into the Fourier part, and degenerate cross terms are emitted at a
canonically offset exponent carrying the exact residue limit. The
cross terms need no offset of ν. The absorbed inputs are cut off at
the grid window, which costs accuracy when the product is evaluated
directly. On the chain, triangle times edge at ν = 1.5 has a relative
error of 1.7e-4 at `n_points = 250`, against 1.4e-8 at ν = 1.5 + π/30,
and four edges in series at ν = 3 have 1.1e-4, against 1.9e-7 at
ν = 3 + π/30. On the square lattice, triangle times edge at ν = 4 has
1.7e-3 at `n_points = 32`, against 1.3e-6 at ν = 4 + π/30. A
convolution after the product brings the error back to the
background, which is why `evaluate_graph` needs no offset. For
products evaluated directly, shift ν by an irrational offset of the
size used in the examples above. Much smaller offsets are no remedy.
At ν = 1.5 + π/300 the open product recovers, but the same product
closed by a further edge loses a factor 23 against ν = 1.5, because an
exponent close to a pole costs accuracy of its own. Details are given
where the algebra path is introduced, under
[Single-graph front-end: `evaluate_graph`](#single-graph-front-end-evaluate_graph).

