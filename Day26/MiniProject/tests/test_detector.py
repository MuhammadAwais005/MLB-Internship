"""Meaningful checks for geometry recovery, safe failure and input validation."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_detector import detect_document, load_image, process_image, save_image


class DocumentDetectorTests(unittest.TestCase):
    def test_recovers_known_document_at_original_resolution(self):
        image = np.full((1200, 1800, 3), 40, dtype=np.uint8)
        expected = np.array([[360, 190], [1450, 260], [1320, 1030], [240, 950]], np.int32)
        cv.fillPoly(image, [expected], (240, 240, 240))
        cv.putText(image, "TEST PAGE", (500, 620), cv.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 4)
        output, result, _ = detect_document(image, max_size=600)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(result["status"], "boundary_candidate")
        actual = np.array(result["boundary_points"], np.int32)
        expected_mask = np.zeros(image.shape[:2], np.uint8)
        actual_mask = np.zeros_like(expected_mask)
        cv.fillPoly(expected_mask, [expected], 1)
        cv.fillPoly(actual_mask, [actual], 1)
        iou = np.count_nonzero(expected_mask & actual_mask) / np.count_nonzero(expected_mask | actual_mask)
        self.assertGreater(iou, 0.95)

    def test_blank_image_has_no_invented_boundary(self):
        image = np.full((240, 320, 3), 220, np.uint8)
        output, result, _ = detect_document(image)
        self.assertEqual(result["status"], "no_boundary")
        self.assertIsNone(result["boundary_points"])
        np.testing.assert_array_equal(output, image)

    def test_saves_required_images_and_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((300, 400, 3), 20, np.uint8)
            cv.rectangle(image, (65, 40), (330, 260), (240, 240, 240), -1)
            save_image(root / "source.png", image)
            result = process_image(root / "source.png", root / "results")
            self.assertEqual(result["status"], "boundary_candidate")
            self.assertTrue((root / "results" / "status.json").is_file())
            self.assertEqual(load_image(root / "results" / "document_boundary.png").shape, image.shape)
            self.assertTrue((root / "results" / "06_cleaned_edges.png").is_file())
            with self.assertRaises(FileNotFoundError):
                load_image(root / "missing.png")
            (root / "corrupt.jpg").write_bytes(b"invalid image")
            with self.assertRaises(ValueError):
                load_image(root / "corrupt.jpg")

    def test_invalid_parameters(self):
        image = np.zeros((100, 100, 3), np.uint8)
        for options in ({"low": 150, "high": 50}, {"kernel_size": 4},
                        {"min_area_ratio": 1.2}, {"low": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                detect_document(image, **options)


if __name__ == "__main__":
    unittest.main()
