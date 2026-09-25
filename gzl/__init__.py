# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Andreas A. Buchheit

"""
Graph Zeta Library (GZL).
"""

# A plain literal, and the single source of truth: pyproject.toml reads it
# with setuptools' ``attr:`` directive, which parses this file rather than
# importing it, so the build never runs the provenance guard below.
__version__ = "1.0.0rc2"

# Fail loudly when this import resolved to a DIFFERENT graph-zeta checkout
# than the one the caller is standing in.  With several worktrees and a
# single ``pip install -e`` target this is easy to hit, and it is silent:
# the foreign tree imports cleanly and every number the run produces
# describes code nobody is editing.  See ``gzl/_provenance.py`` for what
# is checked and for the ``GRAPH_ZETA_PROVENANCE`` escape hatches.
from gzl import _provenance as _gz_provenance
_gz_provenance.check(__file__)

from gzl._data import data_path  # noqa: F401
from gzl.core import (  # noqa: F401
    GraphZeta,
    PrefactorSingularityError,
    cNu,
    make_graph_obj,
    make_epstein_graph,
    graph_sample,
    graph_sample_at,
    graph_zero,
    graph_compress,
    graph_multiply,
    graph_multiply_power,
    graph_convolve,
    graph_convolve_power,
    graph_attach,
    periodic_convolve_nd,
)
from gzl.interaction import (  # noqa: F401
    Interaction,
    InteractionSupportError,
)
from gzl.circle import zeta_circle  # noqa: F401
from gzl.direct_sum import (  # noqa: F401
    direct_sum_zero_momentum,
    direct_sum_extrapolated,
)
from gzl.construction import (  # noqa: F401
    NotSeriesParallelError,
    NotTreewidthTwoError,
    graph_from_sp,
    graph_from_sp_uniform,
    graph_from_tw2,
    graph_from_tw2_uniform,
    graph_from_edges,
    graph_from_edges_uniform,
)
from gzl.tensor_network import (  # noqa: F401
    graph_zeta_general,
    graph_zeta_general_at_zero,
)
from gzl.slab import (  # noqa: F401
    slab_zeta,
    SlabTooLargeError,
)
from gzl.hybrid import (  # noqa: F401
    hybrid_zeta,
    HybridCoreTooLargeError,
)
from gzl.frontend import (  # noqa: F401
    evaluate_graph,
    GraphZetaError,
    SelfLoopError,
    DisconnectedGraphError,
    VertexOutOfRangeError,
    NPointsRequiredError,
    TopologyEvaluatorUnavailableError,
    UnsupportedLatticeSumError,
    UnsupportedRequestError,
)
# Corpus front-end.  Convention: ``evaluate_*`` return graph-zeta
# *values* (``evaluate_graph`` → one graph, ``evaluate_corpus`` → every
# graph in a corpus); ``compute_series_coefficients`` assembles the
# derived perturbative series ``c_r = Σ_G a_r(G) ζ_G`` from those values.
# Imported last: ``series`` imports back from ``gzl`` (the names
# above are all in place by this point), so this avoids a circular import.
from gzl.series import (  # noqa: F401
    evaluate_corpus,
    compute_series_coefficients,
    nu_grid,
)

__all__ = [
    "__version__",
    "GraphZeta",
    "PrefactorSingularityError",
    "cNu",
    "make_graph_obj",
    "make_epstein_graph",
    "graph_sample",
    "graph_sample_at",
    "graph_zero",
    "graph_compress",
    "graph_multiply",
    "graph_multiply_power",
    "graph_convolve",
    "graph_convolve_power",
    "graph_attach",
    "periodic_convolve_nd",
    "Interaction",
    "InteractionSupportError",
    "zeta_circle",
    "NotSeriesParallelError",
    "NotTreewidthTwoError",
    "graph_from_sp",
    "graph_from_sp_uniform",
    "graph_from_tw2",
    "graph_from_tw2_uniform",
    "graph_from_edges",
    "graph_from_edges_uniform",
    "direct_sum_zero_momentum",
    "direct_sum_extrapolated",
    "graph_zeta_general",
    "graph_zeta_general_at_zero",
    "hybrid_zeta",
    "HybridCoreTooLargeError",
    "slab_zeta",
    "SlabTooLargeError",
    "evaluate_graph",
    "evaluate_corpus",
    "compute_series_coefficients",
    "nu_grid",
    "GraphZetaError",
    "SelfLoopError",
    "DisconnectedGraphError",
    "VertexOutOfRangeError",
    "NPointsRequiredError",
    "TopologyEvaluatorUnavailableError",
    "UnsupportedLatticeSumError",
    "UnsupportedRequestError",
    "data_path",
]
