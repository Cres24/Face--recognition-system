"""Download the pinned ONNX models (repo-root entry point).

The canonical implementation lives in the Django backend service so Vercel can
run it from that service root::

    face_recognition_system/scripts/download_models.py

This wrapper just delegates to it, keeping one source of truth for URLs/hashes.

    python scripts/download_models.py             # download anything missing
    python scripts/download_models.py --force     # re-download and verify everything
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "face_recognition_system"
    / "scripts"
    / "download_models.py"
)

_spec = importlib.util.spec_from_file_location("backend_download_models", _BACKEND_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load model downloader at {_BACKEND_SCRIPT}")

_module = importlib.util.module_from_spec(_spec)
# dataclass() looks up cls.__module__ in sys.modules, so register before exec.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

main = _module.main


if __name__ == "__main__":
    raise SystemExit(main())
