<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/graph-zeta/gzl/main/docs/logo/gzl-logo-wide-dark.svg" />
    <img src="https://raw.githubusercontent.com/graph-zeta/gzl/main/docs/logo/gzl-logo-wide-light.svg" width="440" alt="Graph Zeta Library (GZL)" />
  </picture>
</h1>

[![CI](https://github.com/graph-zeta/gzl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/graph-zeta/gzl/actions/workflows/ci.yml?query=branch%3Amain)
[![math](https://img.shields.io/badge/math-arXiv%3A2609.18918-blue.svg)](https://arxiv.org/abs/2609.18918)
[![physics](https://img.shields.io/badge/physics-arXiv%3A2609.18761-blue.svg)](https://arxiv.org/abs/2609.18761)
[![license](https://img.shields.io/badge/license-AGPL--3.0--or--later-lightgrey.svg)](https://github.com/graph-zeta/gzl/blob/main/LICENSE)

Author: Andreas A. Buchheit

Contact: andreas.buchheit@uni-saarland.de, [buchheit-research.org](https://buchheit-research.org)

Dear Visitor, welcome to the Graph Zeta Library!

This library permits the efficient and precise evaluation of the high-dimensional oscillatory lattice sums appearing in high-order linked-cluster expansions of quantum lattice models based on graph zeta functions. For state-of-the-art series expansions, it replaces cluster-scale Monte Carlo runs with precise evaluations within minutes on standard desktop hardware. In addition, it allows one to resolve the full Brillouin zone grid at the cost of a single momentum evaluation, providing access to finely resolved dispersion relations. Supported are general 1D, 2D, and 3D lattices and power-law interactions accompanied by short-ranged terms. Regarding models, we currently provide the graphs and prefactors for the long-range transverse-field Ising model (LRTFIM, order 13 for 0qp and order 11 for 1qp). The method is general and additional default models, such as the Heisenberg model, will be included as time progresses.

GZL evaluates each graph lattice sum, called a graph zeta function if power-law kernels are involved, by first factorizing it over blocks. Each block is then evaluated by the cheapest available strategy depending on its treewidth tw, while exploiting redundancies through caching. Basic blocks, such as bridges and cycles, admit analytic forms in terms of generalized zeta functions. Series-parallel blocks with tw ≤ 2 are computed at linear cost in the number of graph nodes and log-linear cost in the size of the momentum grid using a semi-analytical algebra based on Epstein zeta functions and rapidly decaying Fourier series. This makes accurate evaluation possible even for exponents close to the spatial dimension *d*. Finally, blocks with tw > 2 are evaluated by tensor-network bucket elimination, yielding polynomial scaling of numerical work and memory in momentum grid size with exponents only growing with tw rather than with the number of vertices. Through use of FFT, the full momentum-dependent single-particle excitation is recovered on a momentum grid at the cost of a single momentum evaluation, where Monte Carlo methods would require one run per momentum.

For all details on the numerical method, including the proofs, as well as numerical benchmarks, see [arXiv:2609.18918](https://arxiv.org/abs/2609.18918). The method is integrated into linked-cluster perturbation theory in [arXiv:2609.18761](https://arxiv.org/abs/2609.18761), including a detailed study of the long-range transverse-field Ising model (LRTFIM), with dispersion relations for different microscopic interaction models for the 3D quantum magnet KTmSe₂.

## Installation

GZL requires Python 3.11 or newer and is available on PyPI via

```bash
pip install gzl
```

From a clone of this repository, the library can be installed in editable mode:

```bash
pip install -e .
```

Either installation puts the `gzl` command on the path, whose subcommands are listed by `gzl --help`. `gzl selftest` checks an installation against the shipped data and against values whose answers are known independently, and `gzl info` prints the versions and paths a bug report needs.

Both installations include the full runtime stack consisting of [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [EpsteinLib](https://pypi.org/project/epsteinlib/), [NetworkX](https://networkx.org/), and [h5py](https://www.h5py.org/), which permits the direct use of the `evaluate_graph` front-end and of the corpus loaders. As EpsteinLib is built from source during installation, a C compiler is required on Linux and macOS (e.g. via the Command Line Tools).

The `[dev]` extra adds `pytest`, `pytest-xdist` and `mpmath` for running the test suite:

```bash
pip install -e ".[dev]"
```

The tests are then run from the root of the repository via

```bash
pytest -v
```

Long-running stability sweeps are skipped by default and can be included via `pytest --runslow`.

The TFIM graph corpora and the Monte Carlo reference series are shipped within the package. Details are given in [Shipped reference data](https://github.com/graph-zeta/gzl/blob/main/DOCUMENTATION.md#shipped-reference-data).

## Basic example

The series coefficients of the one-quasiparticle gap of the LRTFIM on the cubic lattice, for ν = 4 up to order 9, are obtained over the whole Brillouin zone by

```python
from gzl import compute_series_coefficients

corpus   = "tfim1qp"   # or "tfim0qp" for the ground-state series
A        = "cubic"     # or "chain", "square", "triangular", or an np.array
nu       = 4.0         # a float, an array (a sweep), or an Interaction
n_points = 8           # Brillouin-zone grid points per dimension

res = compute_series_coefficients(corpus, nu, A, n_points, order_max=9)
for order, value in res.items():
    print(order, value[0, 0, 0])       # k = 0
```

within seconds on a single core. Each coefficient is an array over the Brillouin-zone grid, of which the example prints the zero-momentum component. A single ν carries no further axis. A sweep, `nu = [3.5, 4.0]`, prepends one. The coefficients agree with the Monte Carlo values of S. Fey, [PhD thesis, FAU Erlangen-Nürnberg (2020)](https://open.fau.de/handle/openfau/13818), Table F.10, within their statistical errors.

The same coefficients are obtained from the command line by

```bash
gzl series --corpus tfim1qp --A cubic --nu 4 --n-points 8 \
           --order-max 9 --output coefficients.csv
```

which writes one row per order and Brillouin-zone grid point to `coefficients.csv`. The rows with `k_index` 0 hold the zero-momentum values printed above.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/graph-zeta/gzl/main/docs/figures/cutaway_afm_dark.svg" />
    <img src="https://raw.githubusercontent.com/graph-zeta/gzl/main/docs/figures/cutaway_afm.svg" width="563" alt="Dispersion relation of the LRTFIM over the cubic Brillouin zone, with one octant cut away" />
  </picture>
  <br />
  <em>Dispersion relation for an antiferromagnetic coupling λ = −0.03 at higher order r = 10 and finer resolution with 16³ = 4096 momentum points, requiring approximately 10 minutes on a single core (taken from <a href="https://arxiv.org/abs/2609.18918">arXiv:2609.18918</a>).</em>
</p>

The definition of the graph zeta function and the quick start continue in
the [README on GitHub](https://github.com/graph-zeta/gzl#quick-start). For an extensive discussion of
all front-ends, general interaction kernels, the worked examples, the
engines and the full API reference, see
[DOCUMENTATION.md](https://github.com/graph-zeta/gzl/blob/main/DOCUMENTATION.md).
