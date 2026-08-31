import json
import tempfile
import unittest
from pathlib import Path

from solar_filament.data import assign_folds, audit_dataset, build_manifest
from solar_filament.metrics import evaluate_annotation_sets, score_instances


class DataContractTests(unittest.TestCase):
    def test_manifest_groups_annotators_by_physical_image(self):
        coco = {
            "images": [
                {"id": "a-img1", "file_name": "20110101000000Bh.jpeg", "width": 4, "height": 4},
                {"id": "b-img1", "file_name": "20110101000000Bh.jpeg", "width": 4, "height": 4},
                {"id": "a-img2", "file_name": "20120101000000Ch.jpeg", "width": 4, "height": 4},
            ],
            "annotations": [
                {"id": "1", "image_id": "a-img1", "segmentation": [[0, 0, 1, 0, 1, 1]], "area": 1},
                {"id": "2", "image_id": "b-img1", "segmentation": [[0, 0, 2, 0, 2, 1]], "area": 2},
                {"id": "3", "image_id": "a-img2", "segmentation": [[0, 0, 1, 0, 1, 1]], "area": 1},
            ],
        }

        manifest = build_manifest(coco)

        self.assertEqual(
            [row.file_name for row in manifest],
            ["20110101000000Bh.jpeg", "20120101000000Ch.jpeg"],
        )
        self.assertEqual([item.image_id for item in manifest[0].annotation_sets], ["a-img1", "b-img1"])

    def test_folds_are_reproducible_and_never_split_annotators(self):
        coco = {
            "images": [
                {"id": f"a-{year}{i:02d}Bh", "file_name": f"{year}{i:02d}Bh.jpeg", "width": 4, "height": 4}
                for year in (2011, 2012)
                for i in range(6)
            ],
            "annotations": [],
        }
        for image in coco["images"]:
            coco["annotations"].append(
                {"id": image["id"], "image_id": image["id"], "segmentation": [[0, 0, 1, 0, 1, 1]], "area": 1}
            )
        manifest = build_manifest(coco)

        first = assign_folds(manifest, n_folds=3, seed=17)
        second = assign_folds(manifest, n_folds=3, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(row.file_name for row in manifest))
        self.assertEqual(set(first.values()), {0, 1, 2})

    def test_real_dataset_passes_contract_audit(self):
        root = Path("filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
        if not root.exists():
            self.skipTest("competition data is not attached")

        report = audit_dataset(root)

        self.assertEqual(report.train_files, 707)
        self.assertEqual(report.test_files, 180)
        self.assertEqual(report.image_records, 1154)
        self.assertEqual(report.annotations, 8199)
        self.assertEqual(report.errors, ())


class OfficialMetricTests(unittest.TestCase):
    def test_perfect_match_scores_one(self):
        gt = [{0, 1, 4}]
        report = score_instances(gt, [{0, 1, 4}])
        self.assertEqual(report.pq, 1.0)
        self.assertEqual((report.tp, report.fp, report.fn), (1, 0, 0))

    def test_official_threshold_is_strictly_greater_than_half(self):
        report = score_instances([{0, 1}], [{0, 1, 2, 3}])
        self.assertEqual(report.pq, 0.0)
        self.assertEqual((report.tp, report.fp, report.fn), (0, 1, 1))

    def test_fragmentation_penalizes_the_unmatched_fragment(self):
        report = score_instances([{0, 1, 2, 3}], [{0, 1, 2}, {3}])
        self.assertAlmostEqual(report.pq, 0.5)
        self.assertEqual((report.tp, report.fp, report.fn), (1, 1, 0))
        self.assertEqual(report.one_to_many, 1)

    def test_official_aggregation_weights_each_annotator_set(self):
        perfect = ([{0, 1}], [{0, 1}])
        missed = ([{2, 3}], [])

        report = evaluate_annotation_sets([perfect, perfect, missed])

        self.assertAlmostEqual(report.official_pq, 2 / 2.5)
        self.assertEqual((report.tp, report.fp, report.fn), (2, 0, 1))
        self.assertAlmostEqual(report.macro_pq, 2 / 3)


if __name__ == "__main__":
    unittest.main()
