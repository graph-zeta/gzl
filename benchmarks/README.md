# Benchmarks

Standalone scripts, not part of the test suite.  From a source checkout, run
them with the checkout first on the path:
`PYTHONPATH=$(git rev-parse --show-toplevel) python benchmarks/<script>.py`.

| script | what it measures |
|---|---|
| `mc_comparison.py` | graph-zeta series coefficients against Monte Carlo series of the long-range TFIM (arXiv:2609.18918, Sec. 8.3, Table 2), with every printed cell checked against the published table; about 5 minutes on 6 workers at the published grids, each worker holding up to about 3 GB (the default `--jobs` stays within physical memory); `--quick` for a smoke run, `--help` for the options |
| `bench_power.py` | wall clock of `graph_multiply_power` / `graph_convolve_power`, binary vs linear exponentiation |
| `accuracy_circle.py` | cycle-graph zeta from the algebra against `zeta_circle`, binary vs linear power |
| `accuracy_circle_shift_free.py` | the same across a sweep of ν through the former pole at ν = d + 2 |
