"""Download the pinned ONNX models used by the web app.

    python scripts/download_models.py             # download anything missing
    python scripts/download_models.py --force     # re-download and verify everything
    python scripts/download_models.py --models-dir /some/other/dir

Files land in ``face_recognition_system/ml_models/`` by default (gitignored).
OpenCV Zoo stores ONNX files in Git LFS, hence the ``media.githubusercontent.com``
URLs -- ``raw.githubusercontent.com`` only returns a pointer file. Hashes are from
the pinned upstream files; a mismatched download is deleted and reported as an error.

Licences: YuNet is MIT, SFace is Apache-2.0 (see the model cards in opencv/opencv_zoo),
and the Haar cascade ships with OpenCV (Apache-2.0).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "face_recognition_system" / "ml_models"
LFS_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
OPENCV_DATA = "https://raw.githubusercontent.com/opencv/opencv/4.x/data"


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str
    size: int
    licence: str


MODELS = (
    Model(
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{LFS_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232589,
        licence="MIT",
    ),
    Model(
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{LFS_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size=38696353,
        licence="Apache-2.0",
    ),
    Model(
        filename="haarcascade_frontalface_default.xml",
        url=f"{OPENCV_DATA}/haarcascades/haarcascade_frontalface_default.xml",
        sha256="0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
        size=930127,
        licence="Apache-2.0",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, model: Model) -> str | None:
    actual = file_sha256(path)
    if path.stat().st_size != model.size or actual != model.sha256:
        return f"expected sha256={model.sha256} size={model.size}, got sha256={actual} size={path.stat().st_size}"
    return None


def download(model: Model, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        print(f"downloading {model.filename} ({model.size / 1e6:.1f} MB, {model.licence}) ...", flush=True)
        with urllib.request.urlopen(model.url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        problem = verify(temporary, model)
        if problem:
            raise ValueError(f"checksum failed for {model.filename}: {problem}")
        os.replace(temporary, destination)
        print(f"  ok -> {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = parser.parse_args()

    failures = 0
    for model in MODELS:
        destination = args.models_dir / model.filename
        if destination.is_file() and not args.force:
            problem = verify(destination, model)
            if problem is None:
                print(f"already present: {destination}")
                continue
            print(f"existing file is invalid ({problem}); re-downloading")
        try:
            download(model, destination)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} model(s) could not be installed.", file=sys.stderr)
        return 1
    print("All models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
