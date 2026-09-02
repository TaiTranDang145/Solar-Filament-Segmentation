import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solar_filament.masks import connected_components, decode_mask, encode_mask, rasterize_instances
from solar_filament.submission import build_submission_rows, validate_submission, write_submission


class MaskPipelineTests(unittest.TestCase):
    def test_polygon_rasterization_returns_one_binary_layer_per_instance(self):
        annotations = [
            {"segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]},
            {"segmentation": [[5, 5, 7, 5, 7, 7, 5, 7]]},
        ]

        masks = rasterize_instances(annotations, height=8, width=8)

        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].shape, (8, 8))
        self.assertEqual(masks[0].dtype, np.uint8)
        self.assertEqual(set(np.unique(masks[0])), {0, 1})
        self.assertGreater(int(masks[0].sum()), int(masks[1].sum()))

    def test_connected_components_keeps_exact_minimum_area(self):
        probabilities = np.zeros((5, 5), dtype=np.float32)
        probabilities[1, 1:3] = 0.5
        probabilities[4, 4] = 0.9

        masks = connected_components(probabilities, threshold=0.5, min_area=2)

        self.assertEqual(len(masks), 1)
        self.assertEqual(int(masks[0].sum()), 2)

    def test_connected_components_can_close_a_small_gap(self):
        probabilities = np.zeros((9, 9), dtype=np.float32)
        probabilities[3:6, 1:3] = 1
        probabilities[3:6, 4:6] = 1

        separate = connected_components(probabilities, min_area=1)
        joined = connected_components(probabilities, min_area=1, close_kernel=3)

        self.assertEqual(len(separate), 2)
        self.assertEqual(len(joined), 1)

    def test_coco_rle_round_trip_preserves_asymmetric_mask(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[0, 3] = 1
        mask[2:, 1] = 1

        encoded = encode_mask(mask)
        decoded = decode_mask(encoded, height=4, width=5)

        self.assertIsInstance(encoded, str)
        np.testing.assert_array_equal(decoded, mask)


class SubmissionTests(unittest.TestCase):
    def test_submission_rows_use_one_based_unique_suffixes(self):
        first = np.eye(3, dtype=np.uint8)
        second = np.fliplr(first).copy()

        rows = build_submission_rows({"20110101000000Bh": [first, second]})

        self.assertEqual([row["filament_id"] for row in rows], ["20110101000000Bh_1", "20110101000000Bh_2"])

    def test_submission_rows_remove_overlaps_and_empty_remainders(self):
        first = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        overlapping = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        duplicate = first.copy()

        rows = build_submission_rows(
            {"20110101000000Bh": [first, overlapping, duplicate]}
        )

        masks = [
            decode_mask(row["segmentation_rle"], height=2, width=2)
            for row in rows
        ]
        self.assertEqual([row["filament_id"] for row in rows], ["20110101000000Bh_1", "20110101000000Bh_2"])
        self.assertFalse(np.logical_and(masks[0], masks[1]).any())

    def test_written_submission_validates_and_decodes(self):
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2] = 1
        rows = build_submission_rows({"20110101000000Bh": [mask]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission(path, rows)

            report = validate_submission(
                path,
                expected_stems={"20110101000000Bh"},
                height=4,
                width=5,
            )

        self.assertEqual(report.rows, 1)
        self.assertEqual(report.errors, ())

    def test_validator_rejects_duplicate_ids(self):
        mask = np.ones((2, 2), dtype=np.uint8)
        rle = encode_mask(mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filament_id", "segmentation_rle"])
                writer.writeheader()
                writer.writerow({"filament_id": "20110101000000Bh_1", "segmentation_rle": rle})
                writer.writerow({"filament_id": "20110101000000Bh_1", "segmentation_rle": rle})

            report = validate_submission(path, height=2, width=2)

        self.assertTrue(any("duplicate" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
