# Changelog

All notable changes to `gzl` are recorded here.  This project follows
[Semantic Versioning](https://semver.org/), and the format of this file
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-09-25

First public release of the Graph Zeta Library.  It implements the method
of [arXiv:2609.18918](https://arxiv.org/abs/2609.18918), which
[arXiv:2609.18761](https://arxiv.org/abs/2609.18761) applies to
linked-cluster expansions of quantum lattice models.

* `evaluate_graph` evaluates a graph zeta function at zero momentum, at
  given momenta or on the whole Brillouin-zone grid, on 1D, 2D and 3D
  Bravais lattices.  It factorizes the graph over its blocks and evaluates
  bridges and cycles in closed form, series-parallel blocks with the
  semi-analytical algebra, and blocks of higher treewidth by tensor-network
  bucket elimination.
* `compute_series_coefficients` and `evaluate_corpus` evaluate the series
  coefficients of a graph corpus and the value of each of its graphs.
* `Interaction` describes couplings that combine power laws with a
  compactly supported short-range part, and is accepted wherever an
  exponent is.
* The corpora of the long-range transverse-field Ising model and the
  Monte Carlo reference series ship with the package.
* The `gzl` command runs a corpus pass from the command line
  (`gzl series`) and reports and checks the installation (`gzl info`,
  `gzl selftest`).
* DOCUMENTATION.md, "API stability", lists the stable part of the API.

GZL requires Python 3.11 or newer and is licensed under the GNU Affero
General Public License, version 3 or later.

[1.0.0]: https://github.com/graph-zeta/gzl/releases/tag/v1.0.0
