# gzl reference data

Data files shipped with the package, for evaluating the TFIM corpora and
comparing against published Monte Carlo series without a source checkout.
Resolve them with `gzl.data_path(name)`:

```python
from gzl import compute_series_coefficients, data_path

corpus = data_path("tfim_softcore_corpus_1qp.npz")
c = compute_series_coefficients(corpus, 2.0, "chain", 64, order_max=6)
```

| file | what |
|---|---|
| `tfim_softcore_corpus_0qp.npz` | 0qp softcore TFIM corpus, 8 403 combinatorial graphs, orders 2–13 |
| `tfim_softcore_corpus_1qp.npz` | 1qp softcore TFIM corpus, 22 677 combinatorial graphs, orders 1–11 |
| `full_graph_topologies.npz` | topology snapshot of both corpora, 31 080 graphs |
| `MC_patched/TFIM/<lattice>/` | Monte Carlo series-coefficient references; see `MC_patched/README.md` |

The corpora and the topology snapshot are part of gzl.  Everything
else is **third-party data**: the project license does not cover it, and
its sources are named in the distribution's `NOTICE`.

`PROVENANCE.csv` lists every data file with its source key (a `NOTICE`
entry, or `gzl`), the table or record it comes from, the evidence for
that attribution, and its SHA-256.  Cite the source it names when you use a
file.
