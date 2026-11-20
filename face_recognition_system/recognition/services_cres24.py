import os
import numpy as np
from pymongo import MongoClient
from deepface import DeepFace

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["face_recognition_db"]
users_col = db["users"]

# Cosine distance threshold for Facenet512 (lower = stricter)
THRESHOLD = 0.30


def cosine_distance(vec1, vec2) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def enroll_person(name: str, frame) -> dict:
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )
        embedding = results[0]["embedding"]
        users_col.insert_one({"name": name, "embedding": embedding})
        return {"status": "enrolled", "name": name}
    except ValueError:
        return {"status": "error", "message": "No face detected in frame."}


def recognize_person(frame) -> dict:
    stored_users = list(users_col.find({}, {"_id": 0, "name": 1, "embedding": 1}))
    if not stored_users:
        return {"status": "unknown", "message": "Cannot recognize"}

    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True,
        )
        live_embedding = results[0]["embedding"]
    except ValueError:
        return {"status": "unknown", "message": "Cannot recognize"}

    best_match = None
    lowest_distance = float("inf")

    for user in stored_users:
        dist = cosine_distance(live_embedding, user["embedding"])
        if dist < lowest_distance:
            lowest_distance = dist
            best_match = user["name"]

    if lowest_distance < THRESHOLD:
        return {
            "status": "recognized",
            "name": best_match,
            "distance": round(lowest_distance, 4),
        }

    return {"status": "unknown", "message": "Cannot recognize"}