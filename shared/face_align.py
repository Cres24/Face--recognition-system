"""Shared face detection + alignment used at enrolment, recognition and (later) training.

The destination landmark template is the standard ArcFace 112x112 reference. OpenCV's
``FaceRecognizerSF.alignCrop()`` uses the same template and the same closed-form
similarity transform implemented here, so swapping embedding backends does not change
how crops are produced.

Only ``cv2`` and ``numpy`` are imported so this module can be used by the Django app
and by standalone training/data-prep scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

# (right eye, left eye, nose, right mouth corner, left mouth corner) for a 112x112 crop.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


def make_detector(yunet_path, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
    """Create a YuNet detector. Use the same threshold everywhere (app + data prep)."""
    path = str(yunet_path)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"YuNet model not found: {path}. Run `python scripts/download_models.py`."
        )
    return cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold, nms_threshold, top_k)


def detect(detector, img_bgr):
    """Return YuNet detections as an (N, 15) float32 array (empty if no faces)."""
    height, width = img_bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(img_bgr)
    return faces if faces is not None else np.empty((0, 15), np.float32)


def largest_face(faces):
    """Pick the face with the largest bounding-box area."""
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]


def similarity_transform(landmarks, template=TEMPLATE):
    """Least-squares 2D similarity transform (Umeyama), matching OpenCV's alignCrop."""
    src = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    dst = np.asarray(template, dtype=np.float64).reshape(5, 2)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    a = dst_demean.T @ src_demean / len(src)

    u, s, vt = np.linalg.svd(a)
    d = np.array([1.0, -1.0] if np.linalg.det(a) < 0 else [1.0, 1.0])
    rotation = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / (np.sum(src_demean ** 2) / len(src)))

    matrix = rotation * scale
    translation = dst_mean - matrix @ src_mean
    return np.hstack([matrix, translation[:, None]]).astype(np.float32)


def align_face(img_bgr, face_row, size=112):
    """Warp ``face_row``'s 5 YuNet landmarks into a 112x112 BGR crop.

    ``face_row`` is one row returned by :func:`detect` (bbox, 5 landmarks, score).
    """
    landmarks = np.asarray(face_row, dtype=np.float64)[4:14].reshape(5, 2)
    template = TEMPLATE if size == 112 else TEMPLATE * (size / 112.0)
    matrix = similarity_transform(landmarks, template)
    return cv2.warpAffine(img_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
