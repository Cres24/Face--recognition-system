import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
from decouple import config
from deepface import DeepFace
from pymongo import MongoClient

# Load MongoDB connection from your .env file
MONGO_URI = config("MONGO_URI", default="mongodb://localhost:27017/")
DB_NAME = config("MONGO_DB_NAME", default="face_recognition_db")
COLLECTION_NAME = config("MONGO_COLLECTION_NAME", default="faces")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]


def extract_embedding(image_source) -> list[float]:
    """
    Accepts a file path (str/Path) or OpenCV BGR numpy array.
    Returns a 512-float facial embedding vector.
    """
    results = DeepFace.represent(
        img_path=str(image_source) if isinstance(image_source, Path) else image_source,
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=True,
    )
    if not results:
        raise ValueError("No face detected in the provided image.")
    return results[0]["embedding"]


def save_person_to_mongo(name: str, image_source, person_id: str | None = None) -> dict:
    """Extracts the face embedding and upserts/inserts the record into MongoDB."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Person's name cannot be empty.")

    embedding = extract_embedding(image_source)

    document = {
        "name": clean_name,
        "person_id": person_id or clean_name.lower().replace(" ", "_"),
        "embedding": embedding,  # Stored natively as a BSON array of floats
        "model": "Facenet512",
        "created_at": datetime.now(timezone.utc),
    }

    result = collection.insert_one(document)
    print(f"[SAVED] '{clean_name}' stored in MongoDB (id: {result.inserted_id})")
    return document


def enroll_from_webcam():
    """Opens webcam; press SPACE to capture and enroll a face, Q to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam active -> Press [SPACE] to capture face | Press [Q] to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Enroll Face to MongoDB", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            name = input("Enter person's full name: ").strip()
            person_id = input("Enter ID/USN (optional, press Enter to skip): ").strip() or None
            try:
                save_person_to_mongo(name, frame, person_id=person_id)
            except ValueError as exc:
                print(f"[ERROR] {exc} Try again with better lighting.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Enroll from webcam")
    print("2. Enroll from an image file")
    choice = input("Select option (1 or 2): ").strip()

    if choice == "1":
        enroll_from_webcam()
    elif choice == "2":
        name = input("Enter person's name: ").strip()
        img_path = input("Enter path to image file: ").strip().strip('"')
        save_person_to_mongo(name, Path(img_path))
    else:
        print("Invalid choice.")
        