from django.core.exceptions import ImproperlyConfigured
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pymongo.errors import PyMongoError

from .services.face_service import decode_image, enrol, recognize


@csrf_exempt  # Local development only; add authentication/CSRF before exposing publicly.
@require_POST
def register_face(request):
    """POST /api/faces/ with multipart fields: name, image."""
    name = (request.POST.get("name") or "").strip()
    upload = request.FILES.get("image")
    if not name or upload is None:
        return JsonResponse({"error": "Both 'name' and 'image' are required."}, status=400)

    try:
        result = enrol(name, decode_image(upload.read()))
    except (ValueError, OSError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except (ImproperlyConfigured, PyMongoError) as exc:
        return JsonResponse({"error": f"Database error: {exc}"}, status=503)

    return JsonResponse(
        {"status": "ok", "name": result.name, "image_path": result.image_path},
        status=201,
    )


@csrf_exempt  # Local development only.
@require_POST
def recognize_face(request):
    """POST /api/recognize/ with multipart field: image."""
    upload = request.FILES.get("image")
    if upload is None:
        return JsonResponse({"error": "An 'image' file is required."}, status=400)

    try:
        matches = recognize(decode_image(upload.read()))
    except ValueError:
        return JsonResponse(
            {"status": "unknown", "message": "Cannot recognize", "faces": []},
            status=200,
        )
    except (ImproperlyConfigured, PyMongoError) as exc:
        return JsonResponse({"error": f"Database error: {exc}"}, status=503)

    faces_data = [
        {
            "name": match.name,
            "recognized": match.name.lower() not in ("unknown", "unrecognized", ""),
            "message": match.name if match.name.lower() not in ("unknown", "unrecognized", "") else "Cannot recognize",
            "score": match.score,
            "box": list(match.box),
        }
        for match in matches
    ]

    # Check if at least one face was positively identified in the database
    any_recognized = any(face["recognized"] for face in faces_data)

    if not faces_data or not any_recognized:
        return JsonResponse(
            {
                "status": "unknown",
                "message": "Cannot recognize",
                "faces": faces_data,
            },
            status=200,
        )

    return JsonResponse(
        {
            "status": "ok",
            "message": f"Recognized: {', '.join(f['name'] for f in faces_data if f['recognized'])}",
            "faces": faces_data,
        },
        status=200,
    )