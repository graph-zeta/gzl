# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""``graph_zeta``, the old name of :mod:`gzl`.

The library is the Graph Zeta Library, GZL; ``graph-zeta`` is the method
it implements.  Until 1.0 the import package carried the method's name,
and this module keeps that spelling working: ``import graph_zeta`` and
``from graph_zeta.series import ...`` reach exactly the objects ``import
gzl`` does.

It is an ALIAS, not a copy.  ``sys.modules`` is pointed at the real
package and a finder maps every ``graph_zeta.X`` to ``gzl.X``, so the
two names share one module object, one set of caches and one set of
classes -- ``isinstance`` holds across both spellings, and a graph built
under one name is accepted by a function reached through the other.  A
re-export (``from gzl import *``) would instead build a SECOND library
in the same process, with its own caches and its own classes, which is a
correctness bug rather than a naming one.

The alias goes away in 2.0.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder

_NEW = "gzl"
_REMOVED_IN = "2.0"


class _AliasLoader(Loader):
    """Hands back the ``gzl`` module itself, rather than executing one."""

    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module) -> None:
        pass

    # ``python -m graph_zeta.series`` goes through runpy, which asks the
    # loader for the code object rather than for the module.  Delegate,
    # or the documented CLI invocation dies on a missing ``get_code``.
    def get_code(self, fullname):
        return self._target_loader().get_code(self.target)

    def get_source(self, fullname):
        return self._target_loader().get_source(self.target)

    def _target_loader(self):
        spec = importlib.util.find_spec(self.target)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot resolve {self.target}")
        return spec.loader


class _AliasFinder(MetaPathFinder):
    """``graph_zeta.anything`` resolves to ``gzl.anything``."""

    def find_spec(self, name, path=None, target=None):
        if name != __name__ and not name.startswith(__name__ + "."):
            return None
        new = _NEW + name[len(__name__):]
        origin = is_pkg = None
        if name != __name__:
            # Carry the real module's ``__file__`` and package-ness, so
            # tracebacks and runpy see the file they are running.
            target_spec = importlib.util.find_spec(new)
            if target_spec is not None:
                origin = target_spec.origin
                is_pkg = target_spec.submodule_search_locations is not None
        return importlib.util.spec_from_loader(
            name, _AliasLoader(new), origin=origin, is_package=is_pkg,
        )


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(
    f"graph_zeta is the old name of {_NEW}; import {_NEW} instead "
    f"(the alias is removed in {_REMOVED_IN}).",
    DeprecationWarning,
    stacklevel=2,
)

# Last: from here on ``graph_zeta`` IS ``gzl``, the same object.
sys.modules[__name__] = importlib.import_module(_NEW)
