import unittest
from pathlib import Path

import numpy as np
from django.conf import settings

from recognition.services.embeddings import SFaceEmbedder, get_embedder


@unittest.skipUnless(Path(settings.FACE_SFACE_MODEL).is_file(),
                     "SFace model not installed (run scripts/download_models.py)")
class SFaceEmbedderTests(unittest.TestCase):
    def test_embedding_is_unit_length_128d(self):
        embedder = SFaceEmbedder(settings.FACE_SFACE_MODEL)
        image = np.full((112, 112, 3), 128, dtype=np.uint8)
        vector = embedder.embed(image)
        self.assertEqual(vector.shape, (embedder.embedding_size,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_identical_crops_score_one(self):
        embedder = SFaceEmbedder(settings.FACE_SFACE_MODEL)
        image = np.full((112, 112, 3), 128, dtype=np.uint8)
        a, b = embedder.embed(image), embedder.embed(image)
        self.assertAlmostEqual(embedder.similarity(a, b), 1.0, places=5)

    def test_default_backend_is_sface(self):
        embedder = get_embedder()
        self.assertIsInstance(embedder, SFaceEmbedder)
        self.assertEqual(embedder.model_id, "sface-2021dec")


if __name__ == "__main__":
    unittest.main()
