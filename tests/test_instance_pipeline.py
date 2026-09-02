import unittest
import tempfile
from pathlib import Path

import numpy as np

from solar_filament.data import AnnotationSet, ImageRecord
from solar_filament.instance_pipeline import (
    InstanceConfig,
    select_complete_annotation_set,
    prepare_yolo_dataset,
    square_bounds,
    threshold_instances,
    yolo_label,
)


class LegalInstanceDatasetTests(unittest.TestCase):
    def test_instance_config_allows_skipping_the_refiner(self):
        self.assertEqual(InstanceConfig(data_root="unused", refiner_epochs=0).refiner_epochs, 0)

    def test_selects_the_most_complete_segmentation_set_deterministically(self):
        record = ImageRecord(
            file_name="20110101000000Bh.jpeg",
            width=100,
            height=100,
            year=2011,
            observatory="Bh",
            annotation_sets=(
                AnnotationSet("few", ({"area": 900},)),
                AnnotationSet("small", ({"area": 10}, {"area": 20})),
                AnnotationSet("complete", ({"area": 30}, {"area": 40})),
            ),
        )

        selected = select_complete_annotation_set(record)

        self.assertEqual(selected.image_id, "complete")

    def test_yolo_label_is_always_class_agnostic_and_clipped(self):
        annotation = {
            "category_id": 3,
            "bbox": [1, 2, 3, 4],
            "spine": [5, 6],
            "segmentation": [[-10, 10, 50, 20, 120, 90]],
        }

        label = yolo_label(annotation, width=100, height=100)

        values = label.split()
        self.assertEqual(values[0], "0")
        self.assertEqual(
            [float(value) for value in values[1:]],
            [0.0, 0.1, 0.5, 0.2, 1.0, 0.9],
        )

    def test_square_crop_stays_inside_the_image_at_the_edge(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[:10, :20] = 1

        bounds = square_bounds(mask, context=2.0, minimum=32)

        self.assertEqual(bounds, (0, 0, 40, 40))

    def test_prepared_fold_uses_one_class_and_keeps_physical_images_grouped(self):
        records = [
            ImageRecord(
                file_name=f"201{i}0101000000Bh.jpeg",
                width=10,
                height=10,
                year=2010 + i,
                observatory="Bh",
                annotation_sets=(
                    AnnotationSet(
                        f"annotator-{i}",
                        (
                            {
                                "category_id": 2,
                                "area": 4,
                                "segmentation": [[1, 1, 3, 1, 3, 3]],
                            },
                        ),
                    ),
                ),
            )
            for i in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            for record in records:
                (image_dir / record.file_name).write_bytes(b"jpeg")

            yaml_path = prepare_yolo_dataset(
                records,
                folds={records[0].file_name: 0, records[1].file_name: 1},
                validation_fold=0,
                image_dir=image_dir,
                output_dir=root / "yolo",
            )

            validation_label = (
                root / "yolo" / "labels" / "val" / f"{Path(records[0].file_name).stem}.txt"
            ).read_text()
            training_label = (
                root / "yolo" / "labels" / "train" / f"{Path(records[1].file_name).stem}.txt"
            ).read_text()

        self.assertTrue(yaml_path.name.endswith(".yaml"))
        self.assertTrue(validation_label.startswith("0 "))
        self.assertTrue(training_label.startswith("0 "))

    def test_instance_thresholds_keep_exact_boundaries_and_drop_noise(self):
        masks = []
        for start in (0, 3, 6):
            mask = np.zeros((3, 3), dtype=np.uint8)
            mask.flat[start : start + 3] = 1
            masks.append(mask)
        probabilities = [mask.astype(np.float32) for mask in masks]

        instances = threshold_instances(
            confidences=[0.25, 0.249, 0.8],
            yolo_masks=masks,
            refined_probabilities=probabilities,
            confidence_threshold=0.25,
            mask_threshold=0.5,
            min_area=3,
        )

        self.assertEqual(len(instances), 2)
        np.testing.assert_array_equal(instances[0], masks[2])
        np.testing.assert_array_equal(instances[1], masks[0])

    def test_instance_thresholds_resolve_overlap_by_confidence_before_scoring(self):
        low = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        high = np.array([[0, 1], [0, 1]], dtype=np.uint8)

        instances = threshold_instances(
            confidences=[0.4, 0.9],
            yolo_masks=[low, high],
            refined_probabilities=[low, high],
            confidence_threshold=0.1,
            mask_threshold=0.5,
            min_area=1,
        )

        np.testing.assert_array_equal(instances[0], high)
        np.testing.assert_array_equal(instances[1], np.array([[1, 0], [0, 0]]))


if __name__ == "__main__":
    unittest.main()
