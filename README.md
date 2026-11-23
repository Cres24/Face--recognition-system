# Face Recognition System

A small Django + React app: capture a frame from the webcam, enrol a name, then
recognise that person by name. The backend uses OpenCV YuNet for detection, a
shared 112x112 alignment, and a selectable embedding backend (SFace by default).
Embeddings are stored in MongoDB Atlas via `db.mongo`; Django itself keeps SQLite
for admin/auth/sessions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
python scripts/download_models.py  # downloads the models (~40 MB)
```

`SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `MONGODB_URI` and `MONGODB_DB` are read from
`.env` via `python-decouple`. For Atlas, use your cluster's SRV connection string with
the real URL-encoded password (encode `@`, `#`, `/`, etc.) and allow your current IP
in Network Access. Atlas creates the database on first write.

The download script installs the pinned models into `face_recognition_system/ml_models/`
(gitignored) and verifies SHA256 checksums. OpenCV Zoo stores these files in Git LFS,
so the script uses the `media.githubusercontent.com` endpoints. Licences: YuNet (MIT),
SFace (Apache-2.0), Haar cascade (Apache-2.0). Re-run with `--force` to replace a bad copy.

For Vercel, the Django backend service declares
`"buildCommand": "python scripts/download_models.py"` in the root `vercel.json`.
A self-contained copy of the downloader lives at
`face_recognition_system/scripts/download_models.py` so the models are fetched during
the backend build even when Vercel builds only that service root. They are gitignored,
so they must not be committed.

## Verify

```bash
# From the repo root:
python -m unittest discover -s tests -t . -v

# Then from face_recognition_system/:
python manage.py check
python manage.py runserver
```

Then, from `frontend/`, run `npm install && npm run dev` and open the Vite URL. Enrol a
face with the name field, then press **Recognize Face**. The API contract is unchanged:
`POST /api/faces/` (`name`, `image`) and `POST /api/recognize/` (`image`).

## Model pipeline

```
frame -> YuNet detection -> shared 112x112 alignment -> embedding -> cosine match -> name
```

- `shared/face_align.py` is the single source of truth for detection and alignment. It
  uses the standard ArcFace 5-point template, matching OpenCV's `alignCrop`. Data prep
  for any future training run must use this same module.
- `recognition/services/embeddings.py` owns feature extraction. Embeddings are stored
  with `model_id` and `embedding_size` so vectors from different models can never be
  mixed. Older records without `model_id` are ignored — re-enrol those people.
  `FACE_EMBEDDING_BACKEND` selects the backend; `FACE_MATCH_THRESHOLD` is the cosine
  threshold (0.363 is SFace's documented default).
- `FACE_DETECTOR_SCORE` controls YuNet's confidence threshold, shared by app and data prep.

The webcam script at `face_recognition_system/scripts/recognize.py` still only opens
the camera; the recognition flow lives in the React frontend via the API.
