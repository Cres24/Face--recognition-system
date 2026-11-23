import unittest
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

from shared.face_align import TEMPLATE, align_face, similarity_transform


class SimilarityTransformTests(unittest.TestCase):
    def test_maps_landmarks_to_template(self):
        angle, scale, shift = 0.3, 1.8, np.array([40.0, 25.0])
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]]) * scale
        source = TEMPLATE @ rotation.T + shift

        matrix = similarity_transform(source)
        projected = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        np.testing.assert_allclose(projected, TEMPLATE, atol=0.01)


@unittest.skipUnless(Path(settings.FACE_SFACE_MODEL).is_file(),
                     "SFace model not installed (run scripts/download_models.py)")
class OpenCVAlignmentParityTests(unittest.TestCase):
    def test_align_matches_sface_aligncrop(self):
        recognizer = cv2.FaceRecognizerSF_create(str(settings.FACE_SFACE_MODEL), "")
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        landmarks = TEMPLATE * 2.0 + np.array([60.0, 40.0])
        face_row = np.array(
            [60, 40, 224, 224, *landmarks.ravel().tolist(), 0.99], dtype=np.float32
        ).reshape(1, 15)

        ours = align_face(image, face_row[0])
        theirs = recognizer.alignCrop(image, face_row)

        self.assertEqual(ours.shape, (112, 112, 3))
        self.assertLess(np.abs(ours.astype(np.int16) - theirs.astype(np.int16)).mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
