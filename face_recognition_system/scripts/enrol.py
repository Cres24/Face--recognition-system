"""Enrol a face from an image: python scripts/enrol.py --name Alice --image alice.jpg"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_recognition_system.settings")

import django  # noqa: E402

django.setup()

from recognition.services.face_service import enrol, read_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrol a face into MongoDB")
    parser.add_argument("--name", required=True, help="Name to associate with the face")
    parser.add_argument("--image", required=True, help="Path to a photo containing a face")
    args = parser.parse_args()

    result = enrol(args.name, read_image(args.image))
    print(f"Enrolled {result.name}: {result.image_path} ({result.embedding_size}-D embedding)")


if __name__ == "__main__":
    main()
