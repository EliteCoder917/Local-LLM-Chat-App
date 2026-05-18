"""PyInstaller-friendly entry point for the bundled backend.

Lives OUTSIDE the `backend` package on purpose. When PyInstaller bundles a
script as the program entry, the script runs as `__main__` with no package
context, which breaks relative imports inside `backend/main.py` (`from .config
import CONFIG` blows up with "attempted relative import with no known parent
package"). This wrapper imports `backend` as a proper package first, so all
the relative imports inside it resolve correctly.
"""
from __future__ import annotations

import uvicorn

# Importing the package establishes `backend` as the parent for everything
# inside it. The relative imports in backend/main.py now find their siblings.
from backend.main import app  # noqa: F401 — registers routes on `app`


def main() -> None:
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
