"""Open the webcam and show a live feed: python scripts/recognize.py"""

import cv2


def main() -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise SystemExit("Could not open the default webcam")

    try:
        print("Press 'q' to quit.")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
