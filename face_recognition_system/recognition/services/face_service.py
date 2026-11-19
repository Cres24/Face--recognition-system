"""Face detection and recognition built entirely on OpenCV models.

Detection: YuNet (default) or Haar cascade.
Embedding/matching: SFace (128-D features, cosine similarity).
Face records live in MongoDB via ``db.mongo``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

from db.mongo import get_people_collection

_detector = None
_recognizer = None

# Constant name changed across OpenCV releases; FR_COSINE == 0.
_COSINE = getattr(cv2, "FaceRecognizerSF_FR_COSINE",
                  getattr(cv2, "FaceRecognizerSF_COSINE", 0))


@dataclass
class EnrolResult:
    name: str
    image_path: str
    embedding_size: int


@dataclass
class FaceMatch:
    name: str
    score: float
    box: tuple[int, int, int, int]


def _detector_kind() -> str:
    return settings.FACE_DETECTOR.lower()


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    if _detector_kind() == "yunet":
        model = Path(settings.FACE_YUNET_MODEL)
        if not model.is_file():
            raise FileNotFoundError(f"YuNet model not found: {model}")
        _detector = cv2.FaceDetectorYN_create(
            str(model), "", (320, 320),
            score_threshold=0.9, nms_threshold=0.3, top_k=5000,
        )
    elif _detector_kind() == "haar":
        cascade = cv2.CascadeClassifier(str(settings.FACE_HAAR_CASCADE))
        if cascade.empty():
            raise FileNotFoundError(f"Haar cascade not found: {settings.FACE_HAAR_CASCADE}")
        _detector = cascade
    else:
        raise ValueError("FACE_DETECTOR must be either 'yunet' or 'haar'")
    return _detector


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        model = Path(settings.FACE_SFACE_MODEL)
        if not model.is_file():
            raise FileNotFoundError(f"SFace model not found: {model}")
        _recognizer = cv2.FaceRecognizerSF_create(str(model), "")
    return _recognizer


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data")
    return image


def read_image(path: str | Path) -> np.ndarray:
    """Read a BGR image from disk."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def detect_faces(image: np.ndarray) -> list[dict]:
    """Return detected faces with a bounding box and (when available) landmarks."""
    detector = _get_detector()
    faces: list[dict] = []

    if _detector_kind() == "yunet":
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        # OpenCV returns (retval, faces) where faces has shape (N, 15):
        # bbox, 5 landmarks, and a score.
        _, detections = detector.detect(image)
        if detections is None:
            return faces
        for detection in np.asarray(detections):
            x, y, w, h = (int(round(float(value))) for value in detection[:4])
            faces.append({
                "box": (x, y, w, h),
                "info": np.asarray(detection, dtype=np.float32),
            })
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        for box in boxes:
            x, y, w, h = (int(value) for value in box)
            faces.append({"box": (x, y, w, h), "info": None})

    return faces


def _embed(image: np.ndarray, face: dict) -> np.ndarray:
    """Align a face and return its SFace feature as a (1, 128) float32 array."""
    recognizer = _get_recognizer()
    if face["info"] is not None:
        aligned = recognizer.alignCrop(image, face["info"])
    else:
        x, y, w, h = face["box"]
        crop = image[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            raise ValueError("Empty face crop")
        aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    return np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(1, -1)


def enrol(name: str, image: np.ndarray) -> EnrolResult:
    """Register the largest face in an image under ``name`` in MongoDB."""
    name = name.strip()
    if not name:
        raise ValueError("name must not be empty")

    faces = detect_faces(image)
    if not faces:
        raise ValueError("No face detected in the supplied image")
    face = max(faces, key=lambda item: item["box"][2] * item["box"][3])
    embedding = _embed(image, face)

    directory = Path(settings.MEDIA_ROOT) / settings.FACE_UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    relative_path = Path(settings.FACE_UPLOAD_DIR) / filename
    if not cv2.imwrite(str(directory / filename), image):
        raise OSError("Failed to save the face image")

    get_people_collection().update_one(
        {"name": name},
        {"$set": {
            "name": name,
            "embedding": embedding.flatten().tolist(),
            "image_path": relative_path.as_posix(),
            "created_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    return EnrolResult(name, relative_path.as_posix(), int(embedding.shape[1]))


def recognize(image: np.ndarray) -> list[FaceMatch]:
    """Identify every face in an image against the enrolled embeddings."""
    faces = detect_faces(image)
    if not faces:
        return []

    stored_people = list(
        get_people_collection().find({}, {"name": 1, "embedding": 1})
    )
    if not stored_people:
        return []

    recognizer = _get_recognizer()
    threshold = settings.FACE_MATCH_THRESHOLD
    matches: list[FaceMatch] = []

    for face in faces:
        embedding = _embed(image, face)
        best_name = "Unknown"
        best_score = -1.0
        for person in stored_people:
            known = np.asarray(person.get("embedding", []), dtype=np.float32).reshape(1, -1)
            if known.size == 0 or known.shape != embedding.shape:
                continue
            score = float(recognizer.match(embedding, known, _COSINE))
            if score > best_score:
                best_score, best_name = score, person["name"]

        matches.append(FaceMatch(
            name=best_name if best_score >= threshold else "Unknown",
            score=round(best_score, 4),
            box=face["box"],
        ))

    return matches


def draw_matches(image: np.ndarray, matches: list[FaceMatch]) -> np.ndarray:
    """Draw boxes and labels on a copy of the image (used by the webcam CLI)."""
    output = image.copy()
    for match in matches:
        x, y, w, h = match.box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{match.name} ({match.score:.3f})"
        cv2.putText(
            output, label, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return output
