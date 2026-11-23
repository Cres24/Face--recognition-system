"""Face feature extraction, behind one small interface.

Alignment lives in ``shared.face_align`` and is identical for every backend. Only
feature extraction and comparison differ, so adding a new model (e.g. an exported
MobileFaceNet ONNX) means adding one class here and a settings switch.

Embeddings returned by :meth:`FaceEmbedder.embed` are L2-normalised 1-D float32
arrays; cosine similarity is a plain dot product.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


class FaceEmbedder:
    model_id = "unknown"
    embedding_size = 0

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


class SFaceEmbedder(FaceEmbedder):
    """OpenCV SFace: 128-d features from an already aligned 112x112 BGR crop."""

    model_id = "sface-2021dec"
    embedding_size = 128

    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SFace model not found: {path}. Run `python scripts/download_models.py`."
            )
        create = getattr(cv2, "FaceRecognizerSF_create", None) or cv2.FaceRecognizerSF.create
        self._recognizer = create(str(path), "")

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        # SFace's own ONNX graph normalises the raw RGB blob internally, so pass the
        # aligned crop straight through. Do NOT apply the MobileFaceNet (x-127.5)/127.5
        # normalisation here.
        feature = self._recognizer.feature(aligned_bgr)
        embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("SFace returned a zero embedding")
        return embedding / norm


@lru_cache(maxsize=1)
def get_embedder() -> FaceEmbedder:
    """Return the embedder selected by ``FACE_EMBEDDING_BACKEND``."""
    backend = settings.FACE_EMBEDDING_BACKEND.lower()
    if backend == "sface":
        return SFaceEmbedder(settings.FACE_SFACE_MODEL)
    raise ImproperlyConfigured(
        f"Unsupported FACE_EMBEDDING_BACKEND={backend!r}. Expected 'sface'."
    )
