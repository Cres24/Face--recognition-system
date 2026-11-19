"""Recognize faces: python scripts/recognize.py --image test.jpg [--webcam]"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_recognition_system.settings")

import cv2  # noqa: E402
import django  # noqa: E402

django.setup()

from recognition.services.face_service import draw_matches, read_image, recognize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize faces against enrolled people")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to an image to recognize")
    source.add_argument("--webcam", action="store_true", help="Use the default webcam")
    args = parser.parse_args()

    if args.image:
        matches = recognize(read_image(args.image))
        if not matches:
            print("No faces found (or no people enrolled yet).")
        for match in matches:
            print(f"{match.name}: {match.score:.4f} at {match.box}")
        return

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open webcam", file=sys.stderr)
        sys.exit(1)
    try:
        print("Press 'q' to quit.")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output = draw_matches(frame, recognize(frame))
            cv2.imshow("Face recognition", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
