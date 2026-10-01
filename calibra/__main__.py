"""Enable ``python -m calibra``.

Kept separate from ``cli.py`` so importing ``calibra`` does not pull in argparse
and the CLI surface -- the library is usable on its own.
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())