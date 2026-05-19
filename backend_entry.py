"""PyInstaller-friendly entry point for the bundled backend.

Lives OUTSIDE the `backend` package on purpose. When PyInstaller bundles a
script as the program entry, the script runs as `__main__` with no package
context, which breaks relative imports inside `backend/main.py` (`from .config
import CONFIG` blows up with "attempted relative import with no known parent
package"). This wrapper imports `backend` as a proper package first, so all
the relative imports inside it resolve correctly.
"""
from __future__ import annotations

import argparse
import uvicorn

# Importing the package establishes `backend` as the parent for everything
# inside it. The relative imports in backend/main.py now find their siblings.
from backend.main import app  # noqa: F401 — registers routes on `app`


def main() -> None:
    # `--port` is supplied by the Electron bridge, which picks a free port
    # at startup (prefers 8765, falls back to an OS-assigned one if a zombie
    # holds it). Default kept for the rare case the binary is launched
    # directly, e.g. for ad-hoc debugging.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
