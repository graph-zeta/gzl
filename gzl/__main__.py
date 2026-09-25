# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""``python -m gzl`` -- the same command the ``gzl`` console script runs."""

from __future__ import annotations

import sys

from gzl.cli import main

if __name__ == "__main__":
    sys.exit(main())
