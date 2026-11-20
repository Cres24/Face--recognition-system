# Face Recognition System

Baseline project setup: a Django skeleton, dependency/ignore configuration,
MongoDB Atlas configuration, and a script that opens the webcam. Face
detection/recognition is not implemented yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

`SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `MONGODB_URI`, and `MONGODB_DB` are read
from `.env` via `python-decouple`. The `.env` file is gitignored.

For MongoDB Atlas, use your cluster's SRV connection string with the real
URL-encoded password (encode special characters such as `@`, `#`, `/`), and
make sure your current IP is in the cluster's Network Access list. Atlas
creates the database on first write.

## Verify Django

From `face_recognition_system/`:

```bash
python manage.py check
python manage.py runserver
```

Django itself still uses SQLite (`db.sqlite3`) for admin/auth/sessions.
`db/mongo.py` holds the shared MongoDB client and `people` collection helper
for when face storage is added later.

## Webcam script

```bash
cd face_recognition_system
python scripts/recognize.py
```

This only opens the default webcam and displays the live feed; press `q` to
quit. It uses OpenCV but does no detection or recognition yet.
