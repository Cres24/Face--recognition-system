import cv2
import requests

BASE_URL = "http://127.0.0.1:8000/api"
cap = cv2.VideoCapture(0)

print("Controls: [E] Enroll face | [R] Recognize face | [Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    cv2.imshow("Face Recognition System", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        name = input("Enter person's name to enroll: ").strip()
        if not name:
            print("Name cannot be empty.")
            continue
        _, img_encoded = cv2.imencode(".jpg", frame)
        res = requests.post(
            f"{BASE_URL}/faces/",
            data={"name": name},
            files={"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
        )
        print("Enroll Response:", res.status_code, res.json())

    elif key == ord("r"):
        _, img_encoded = cv2.imencode(".jpg", frame)
        res = requests.post(
            f"{BASE_URL}/recognize/",
            files={"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
        )
        data = res.json()
        print("Recognize Response:", data.get("message", data))

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()