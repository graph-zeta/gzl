# AGENTS.md

Guidance for coding agents working in this repository. The user
documentation is in README.md and DOCUMENTATION.md, the contribution
workflow in CONTRIBUTING.md.

## Install, test, run

```bash
pip install -e ".[dev]"            # numpy, scipy, epsteinlib, networkx, h5py + pytest, pytest-xdist, mpmath
pytest -n auto --dist loadfile     # what CI runs, skips @pytest.mark.slow
pytest --runslow                   # also the slow tests (pytest -m slow: only those)
pytest tests/test_series.py::TestCorpusNPZ::test_scalar_nu_has_no_sweep_axis
gzl series --config examples/series_1qp_sigma2_d1.toml --progress
```

`pythonpath = ["."]` in `[tool.pytest.ini_options]` puts the checkout on
`sys.path` for the suite. A script run by path needs `PYTHONPATH=$PWD`, and
`python -m gzl` is the `gzl` command. Importing gzl from one checkout while
standing in another raises `ProvenanceError` (`gzl/_provenance.py`).

## The graph zeta function

A graph zeta is `G = (V, E, ν, s, t)`, a finite multigraph with two
terminals `s, t` and per-edge exponents `ν: E → ℝ⁺`, on a Bravais lattice
`Λ = A·ℤ^d`. The columns of `A` are the primitive vectors, and every public
function taking `A` also accepts `"chain"`, `"square"`, `"triangular"` and
`"cubic"`. With `K_ν(x) = |x|^{-ν}` for `x ≠ 0` and `K_ν(0) = 0`

```
ζ_G(k) = Σ_{ {x_v ∈ Λ}, v ≠ p }  exp(−2πi (x_t − x_s)·k)  Π_{e∈E} K_{ν_e}(x_{e⁺} − x_{e⁻})
```

summed over all vertex positions with one pin `p` fixed at `x_p = 0`. The
value does not depend on the pin, and the zero of the kernel at the origin
removes every configuration with coincident endpoints. Here k is Cartesian
(over 2π), while every `momentum` argument is fractional, `β = Aᵀk`. The sum
converges absolutely when every `ν_e > d` (sufficient, not necessary) and
continues meromorphically in each `ν_e`. For `s = t` (vacuum) the Fourier
factor is 1. For real exponents ζ_G(k) is real. σ denotes `min ν − d`.

A corpus (NPZ in flat CSR layout, or HDF5) stores combinatorial graphs only:
topology, multiplicities, prefactor and optional terminals `hopping`, no
exponents. A scalar ν goes on every edge, and an edge of multiplicity m
becomes one edge of exponent mν. The series coefficient is
`c_r(k) = Σ_{G ∈ C_r} a_r(G) ζ_(G,ν)(k)` over the corpus graphs of order r.
The lattice enters only through `A`, so one corpus serves every lattice.

## Architecture

There is one torus truncation and one box truncation, both run by the
bucket-elimination executor in `_contract.py`. On the torus (a periodic
`n_points^d` grid per vertex) `graph_zeta_general` drives the executor
directly, `hybrid_zeta` through `_dense_core` in `hybrid.py` after an SP
reduction, and `slab_zeta` through `_dense_core` with a second pin. Called
directly, the three agree to round-off. The box (`direct_sum`, an open
`[-L, L]^d` window with Richardson extrapolation) is the independent second
reference family, which makes every routing threshold a measurement.
Anything that looks like a third engine is a pin choice or an association.

- `core.py`. `GraphZeta(aMat, bVec, nuVec, A)` splits a graph zeta into a
  regular part (Fourier coefficients on the balanced `n_points^d` grid) and
  singular Epstein-zeta terms. `graph_multiply` is serial composition,
  `graph_convolve` parallel composition, `graph_attach` a 1-sum decoration.
  `graph_compress(g, sigma_max)` keeps exponents `ν ≤ d + sigma_max`
  analytic and absorbs the rest into the Fourier part (`sigma_max=0`: exact
  Fourier coefficients, no Γ algebra). A fifth field, `table_magnitude`,
  carries a compact table's `Σ|a|` (None for power laws) so that
  `graph_zero_conditioned` sees cancellation inside a table, and every new
  operation propagates it.
- `construction.py`. `graph_from_sp`, `graph_from_tw2` (NetworkX, treewidth
  ≤ 2, `NotSeriesParallelError` on a K_4 minor), `graph_from_edges[_uniform]`.
- `_elimination.py`. `plan` chooses pin and elimination order,
  `min_free_cut_nu` gives the escaping cluster cut that sets the truncation
  error exponent.
- `tensor_network.py`. `graph_zeta_general`. A step that isolates one edge
  kernel runs as an FFT convolution. A purely compact kernel (ν = ∞, a table
  without power-law tail) takes an exact sparse peel, so integer tables stay
  integral (ν = ∞ counts homomorphisms), and the router runs such a block
  on its smallest exact torus (`_compact_exact_grid`).
- `slab.py`. The second pin lowers the inner exponent by one power of
  `n_points^d`, and with it peak memory, at the same total work. The outer
  sum folds onto orbits of `lattice_window_group`, found by testing
  candidates against `_label_distances`, never from the metric alone.
- `direct_sum.py`. The box. Its pin never moves (the box is not
  translation-closed), and its FFT peel runs only past a measured cost
  gate, since zero-padding costs `2^d`.
- `interaction.py`. `Interaction`, a finite table symmetric under x → −x
  plus power laws. Front ends take it as `nu`, engines through `kernels=`.
- `circle.py` has `zeta_circle` (the Epstein zeta is epsteinlib's),
  `frontend.py` `evaluate_graph`, `series.py` the corpus front ends,
  `cli.py` and `_cli_*.py` the `gzl` command.
- `_labels.py`, `_lattices.py`. Input normalisation. Vertex labels are
  names, an edge list is compressed order-preservingly, and a pin label in
  no edge raises. The NetworkX entry points refuse an isolated vertex,
  whose lattice sum diverges.

A torus evaluation pays the planner's exponent on its dominant step, at
most `n_points^(τ·d)` with τ the treewidth (`tests/test_tau_d_acceptance.py`).
The dense bound `O(V · n_points^((τ+1+N_t)·d))` is not the cost. Engines
price memory before they allocate, because numpy does not raise on a huge
`np.empty`, and refuse with a `MemoryError` subclass. A byte-budget refusal
(`HybridCoreBudgetError`) goes to the box, never to `graph_zeta_general`,
which has no byte cap.

### The front end

`evaluate_graph` takes an edge list with `nu`, or a NetworkX multigraph with
ν on its edges, and returns a real float, or a real array for a full
Brillouin-zone grid or a batch of k-vectors. It merges parallel edges into
bundles of summed exponent, factors the graph into biconnected blocks (at
finite k only the blocks on the s–t path carry momentum, the others give
ζ_B(0)), and routes each block to the cheapest method that meets the
accuracy standard. A bridge goes to the Epstein zeta, a cycle off the spine
at d ≤ 3 to `zeta_circle`, treewidth 2 at small σ to the algebra, the rest
to the torus through `hybrid_zeta`, except dense blocks at d ≥ 4, which go
to the box. The slab and the box take the classes where the torus measured
too inaccurate, and the box a dense block whose torus ran out of memory. The thresholds are the tables in
`frontend.py` (`_TORUS_MIN_N_BY_CLASS`, `_SLAB_N_BY_CLASS`,
`_SLAB_PREFERRED_RUNGS`, `_FK_PAIR_BY_P`), each beside its measurement.
Change one only with a new measurement against the box. `engine="tensor"`
drops the split and the graded core, so through `evaluate_graph` it is not
equivalent to the default.

`accuracy="strict"` (default for vacuum scalars and explicit k) runs the
accuracy gate and its detours. `accuracy="floor"` (default for a full-grid
request, always used by the corpus front ends) evaluates every dense block
at the pass grid, so a pass's cost grows about linearly with its graphs.

`block_cache` keeps one value per isomorphism class of blocks, computed on
the first labelling seen, so fill order can move results at truncation
level. Its key describes the evaluator too: a setting that changes a
block's value goes into `_flags` in `_evaluate_via_topology`.

A refused well-formed request raises a `GraphZetaError` (a `ValueError`).
A bundle exponent `ν ≤ d` in any block but a bridge, or a bridge at its
pole, raises `UnsupportedLatticeSumError`, a support limit rather than a
divergence verdict. Malformed arguments raise a plain `ValueError` or
`TypeError`. The corpus pass has no fallback: a refused graph raises with a
note naming the graph and order (`tests/test_series_refusals.py` pins that
no shipped graph is refused), and the old routing settings of the corpus
front ends are accepted, ignored and warned about until removal.

### The pole resonance

`graph_multiply` carries a Γ prefactor with poles at `ν = d + 2n`, handled
automatically for `n ≥ 1` (`ν = d` raises `PrefactorSingularityError`).
Resonant means an input at `ν = d + 2n` or a cross term with `m·σ ∈ 2ℕ`
(σ = 0.25, 0.5, 1, ...). Everything through `evaluate_graph`, and any block
closed by a parallel composition, keeps background accuracy there. An open
product of the public algebra loses the window tail of the absorbed input,
so evaluate it at ν plus an irrational offset of order π/30, or with
`sigma_max=0`. Absorbing exact grid samples instead breaks convolutions
(see the comment above `_POLE_BIRTH_WINDOW` in `core.py`).

### Data and the public surface

`gzl.data_path(name)` resolves files under `gzl/data/`: the corpora
`tfim_softcore_corpus_{0,1}qp.npz` (corpus arguments, not `data_path`, also
take `"tfim0qp"` and `"tfim1qp"`), `full_graph_topologies.npz` and the Monte
Carlo references. Never anchor on `gzl.data`, a namespace package.
`PROVENANCE.csv` lists every file with SHA-256 and source, `NOTICE` every
third-party source, and `tests/test_reference_data.py` keeps the three in
step. A new file type there needs a glob in `[tool.setuptools.package-data]`,
since CI checks the wheel and sdist against `git ls-files gzl`.

`gzl/__init__.py` re-exports the public API, and DOCUMENTATION.md ("API
stability") marks each name stable or provisional. `tests/test_api_contract.py`
holds the tiers against the code: a new export or parameter needs an entry
there and a CHANGELOG line, and a stable one changes only in a major
version. New private helpers start with `_` or live in a private module.
`graph_zeta/` is the old import name, an alias of gzl until 2.0, never a
re-export. Tests patch module-level switches of the engines
(`tensor_network._USE_CONV` and similar), so engine code reads them at call
time.

`tools/` holds the scripts that built the corpora and the topology snapshot
(their pCUT inputs are not distributed) and `tools/pypi_description.py`,
which regenerates docs/pypi.md after a README edit. CI runs every
`examples/*.py`. `benchmarks/mc_comparison.py` reproduces the Monte Carlo
comparison of arXiv:2609.18918, Sec. 8.3.

## Conventions

- **The frozen reference harness gates every engine change.**
  `tests/test_engine_reference.py` replays the `float.hex()` values in
  `tests/data/engine_reference_values.json` over every engine path. A red
  entry means a change moved a shipped number. Fix the change, or, for a
  deliberate and documented reassociation, move the entry to `ROUNDOFF_OK`
  in the same commit with the reason. Never regenerate the JSON in a change
  that alters an engine: a value move gets its own commit naming old and
  new hex and the change that moved it (`_meta.regenerations`).
- **Zero tolerance holds on the freezing toolchain.** On Linux
  (`tests/_env_gate.FROZEN_ENV`) the comparison is bitwise except box keys
  (8 ULP, LAPACK `lstsq`), non-identity metric cells (24 ULP) and one key
  measured flaky there (`router_mix|K4|d2|n12`, 8 ULP). Elsewhere every key
  gets 64 ULP, which still refuses every defect measured so far (off Linux
  the variation comes from `np.fft`). `tests/test_executor_goldens.py` was
  frozen on macOS/arm64 and is bitwise only there, so an engine refactor
  needs both gates read in their own environments.
- **Reference data is pinned.** A file under `gzl/data/` changes, appears
  or disappears only together with its `PROVENANCE.csv` row. Third-party
  data ships only with a citable source or recorded permission and a
  `NOTICE` entry, and is never edited in place. Never guard a shipped file
  with skip-if-missing, which would hide a broken path.
- Python ≥ 3.11, type hints encouraged on new public functions. New `.py`
  files carry the SPDX header
  ```python
  # SPDX-License-Identifier: AGPL-3.0-or-later
  # Copyright (C) <year> <author>
  ```
- The licence is AGPL-3.0-or-later, like epsteinlib. Contributions come in
  under the MIT License in addition, and opening a PR is the agreement.
- **Every change goes through a PR**, docs and one-line fixes included:
  branch off `main` (in a fork unless you can push), `git commit -s`, push,
  `gh pr create`. Never push to `main`. CONTRIBUTING.md has the details.
- **Branch names describe the work** (`feature/`, `fix/`, `perf/`, `docs/`,
  `measure/`, `refactor/`, `repro/`, as in `perf/dense-core-byte-ceiling`),
  never an author, agent, model or tool (no `claude/`). Rename before
  pushing, since renaming closes an open PR.
- **No AI-attribution trailers** in commits or PRs (no `Co-Authored-By` for
  a model, no "Generated with ..."). The only trailer is `Signed-off-by:`.
- Write README.md and DOCUMENTATION.md in plain sentences, without em
  dashes, semicolons or "X: the Y that ..." appositions.
- **Verify before claiming.** Where an exact check exists (a reference
  value, a closed form, a gated test, a known limit), every quantitative
  claim is its output, and the cheapest disconfirming test runs first. A
  simplified model's result stays a model result until checked on the real
  object. Check extrapolations against their admissible range, look at every
  plot you generate, and never compare across regimes (hardcore or
  softcore, sector, normalisation) without an explicit conversion.
- **Never cite from memory.** Check the details and the attribution of
  every reference against the publisher, a DOI resolver or arXiv first.

## Releasing

`gzl.__version__` is the single source of the version (PEP 440). CHANGELOG.md
has one entry `## [X.Y.Z]` per version, shared by its pre-releases and
dated for the final release. CITATION.cff's `version` is updated by hand.
Merge, wait for CI to pass on that commit of `main`, and push the tag
`v<version>`. `release.yml` checks that tag, version, changelog and CI agree
(`.github/scripts/release_guard.py`), builds sdist and wheel once, checks
them on every supported Python (`package.yml`) and creates the GitHub
release. On graph-zeta/gzl a pre-release also goes to TestPyPI and a final
release to PyPI, each after approval in its environment. A merge with
`[skip ci]`, or touching only files ci.yml ignores (the docs, CHANGELOG),
has no CI run and cannot be tagged until `ci.yml` is started on it by hand,
so date the changelog in the version bump PR. Run `release.yml` by hand for
a dry run.

## LaTeX in README.md and DOCUMENTATION.md

GitHub runs CommonMark emphasis and escapes before it finds `$...$`:

1. `_` between two punctuation characters (`}_{`) can pair with a later `_`
   as emphasis. Write `$\boldsymbol{k_{\text{phys}}}$`, not `$\boldsymbol{k}_{\text{phys}}$`.
2. `\,`, `\!`, `\;`, `\:` lose their backslash in inline math.
3. Math attached to a word (`finite-$\boldsymbol{k}$`) is never detected.
   Write `finite-**k**`.
4. `$math$)` sometimes fails. Add a space or rephrase.

Round-trip the changed file (here README.md) through GitHub's renderer
before committing,

```bash
jq -Rs '{text: ., mode: "gfm", context: "graph-zeta/gzl"}' README.md \
  | gh api -X POST /markdown --input - > /tmp/readme_rendered.html
```

check that no LaTeX command or `$...$` pair appears outside
`<math-renderer>` and `<code>`, and no `<math-renderer>` holds a residue
like `i,\boldsymbol`, then view the file on GitHub, where the math is set.
