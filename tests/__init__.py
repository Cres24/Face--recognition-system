"""Test bootstrap: make the Django project and repo-root packages importable."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "face_recognition_system"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_recognition_system.settings")
os.environ.setdefault("SECRET_KEY", "test-only-secret-key")
