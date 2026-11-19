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
    name = request.POST.get("name")
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
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except (ImproperlyConfigured, PyMongoError) as exc:
        return JsonResponse({"error": f"Database error: {exc}"}, status=503)

    return JsonResponse({
        "status": "ok",
        "faces": [
            {"name": match.name, "score": match.score, "box": list(match.box)}
            for match in matches
        ],
    })
