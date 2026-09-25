# Monte Carlo reference data

Reference series of the long-range transverse-field Ising model, used to
validate the graph-zeta series against published Monte Carlo results.
They are external data, not output of the library, and are read with
`gzl.data_path("MC_patched/TFIM/<lattice>/<file>")`.

Only files with a citable published source or recorded permission are
included. Every file is listed in `gzl/data/PROVENANCE.csv` with its
source, the evidence for that source and its SHA-256, and the sources are
named in the distribution's `NOTICE`. Please cite the source of the files
you use.

## Sources

| lattice | series | files | source |
|---|---|---|---|
| chain | 0qp ground-state energy, σ = 0.5, 1, 2, 3, order 13 | `gs_energy_series_*.csv` | Langheld et al. (2022), data record on Zenodo |
| chain | 1qp gap at k = 0, σ = 0.5, 1, 2, 3, order 11 | `1qp_gap_series_*_k0_*.csv` | Langheld et al. (2022), data record on Zenodo |
| chain | 1qp gap at k = 0 | `*__k0__alpha_*.dat` | Fey (2020), Table F.2 |
| chain | 1qp gap at k = π | `*__kPi__alpha_*.dat` | Fey (2020), Table F.3 |
| chain | 1qp gap at k = π, ten values of σ, order 10 | `1qp_gap_series_*_kPi_*_order10.csv` | P. Adelhardt, private communication |
| square | 1qp gap at k = 0 and k = (π, π) | `*__k0__*.dat`, `*__kPi__*.dat` | Fey, Kapfer and Schmidt (2019), supplementary Tables I and III, also Fey (2020), Tables F.5 and F.6 |
| square | 0qp ground-state energy | `*__0qp__*.dat` | Fey (2020), Table F.4 |
| square | 0qp ground-state energy, order 13, and 1qp gap at k = 0, order 11 | `*.csv` | P. Adelhardt, private communication |
| triangular | 1qp gap at k = 0 and at the K point | `*__k0__*.dat`, `*__kx2.09_ky-2.09__*.dat` | Fey, Kapfer and Schmidt (2019), supplementary Tables II and IV, also Fey (2020), Tables F.8 and F.9 |
| triangular | 0qp ground-state energy | `*__0qp__*.dat` | Fey (2020), Table F.7 |
| cubic | 1qp gap at k = 0 and k = (π, π, π) | `*__k0__*.dat`, `*__kPi__*.dat` | Fey (2020), Tables F.10 and F.11 |

There is no cubic 0qp series. Remarks on single files are in the `note`
column of `PROVENANCE.csv`. The top order of the square and triangular
ground-state energy series of Fey (2020) is suspected to be
underconverged and is left out of the comparison in Sec. 8.3 of
[arXiv:2609.18918](https://arxiv.org/abs/2609.18918).

## File formats

* `*_series_*.csv` has the columns `order,prefactor,error,relative error`.
* `*_series_coeffs_*.csv` is split by cluster size, with the columns
  `order,Nsites,prefactor,error,relative error,Nseeds`. Summed over
  `Nsites` at each order it gives the series.
* `TFIM_lr_*__MC__list_of_prefactors__*.dat` has the whitespace-separated
  columns `order value error` and no header. These files carry the
  order-1 value, which the CSV files set to zero.

In a file name, `alpha` is the exponent ν of the interaction and
`sigma = alpha - d`. Momentum labels in file names are rounded, `2.09`
stands for 2π/3. A label is the fractional coordinate in the dual basis
times 2π, so the `momentum` argument of the library is the exact label
divided by 2π.

## References

* S. Fey, S. C. Kapfer and K. P. Schmidt, *Quantum criticality of
  two-dimensional quantum magnets with long-range interactions*,
  Phys. Rev. Lett. **122**, 017203 (2019).
  DOI [10.1103/PhysRevLett.122.017203](https://doi.org/10.1103/PhysRevLett.122.017203),
  arXiv:1802.06684.
* S. Fey, *Investigation of Zero-Temperature Transverse-Field Ising
  Models With Long-Range Interactions*, PhD thesis,
  Friedrich-Alexander-Universität Erlangen-Nürnberg (2020),
  [open.fau.de/handle/openfau/13818](https://open.fau.de/handle/openfau/13818).
  Appendix F holds the coefficient tables:

  | table | lattice | series |
  |---|---|---|
  | F.1 | chain | 0qp ground-state energy |
  | F.2 | chain | 1qp gap at k = 0 |
  | F.3 | chain | 1qp gap at k = π |
  | F.4 | square | 0qp ground-state energy |
  | F.5 | square | 1qp gap at k = 0 |
  | F.6 | square | 1qp gap at k = (π, π) |
  | F.7 | triangular | 0qp ground-state energy |
  | F.8 | triangular | 1qp gap at k = 0 |
  | F.9 | triangular | 1qp gap at the K point |
  | F.10 | cubic | 1qp gap at k = 0 |
  | F.11 | cubic | 1qp gap at k = (π, π, π) |

* A. Langheld, J. A. Koziol, P. Adelhardt, S. C. Kapfer and
  K. P. Schmidt, *Scaling at quantum phase transitions above the upper
  critical dimension*, SciPost Phys. **13**, 088 (2022).
  DOI [10.21468/SciPostPhys.13.4.088](https://doi.org/10.21468/SciPostPhys.13.4.088).
  Data record: *Raw data to "Scaling at quantum phase transitions above
  the upper critical dimension"*, Zenodo (2022),
  DOI [10.5281/zenodo.6645107](https://doi.org/10.5281/zenodo.6645107),
  CC BY 4.0.
* P. Adelhardt, private communication: the chain series at k = π to
  order 10, and the square-lattice ground-state energy to order 13 and
  gap at k = 0 to order 11. Unpublished coefficients, distributed with
  permission. Quantities derived from them are published in
  P. Adelhardt, J. A. Koziol, A. Schellenberger and K. P. Schmidt,
  Phys. Rev. B **102**, 174424 (2020),
  DOI [10.1103/PhysRevB.102.174424](https://doi.org/10.1103/PhysRevB.102.174424),
  and P. Adelhardt, J. A. Koziol, A. Langheld and K. P. Schmidt,
  Entropy **26**, 401 (2024),
  DOI [10.3390/e26050401](https://doi.org/10.3390/e26050401).
