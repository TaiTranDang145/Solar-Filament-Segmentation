import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from solar_filament.data import AnnotationSet, ImageRecord
from solar_filament.inference import predict_image, predict_probability
from solar_filament.model import build_model
from solar_filament.training import (
    TrainConfig,
    FilamentDataset,
    gradient_divisor,
    optimizer_step_batches,
    build_training_loader,
    segmentation_loss,
    select_annotation_set,
)


class MultiAnnotatorTrainingTests(unittest.TestCase):
    def test_annotation_selection_cycles_without_unioning_annotators(self):
        record = ImageRecord(
            file_name="20110101000000Bh.jpeg",
            width=8,
            height=8,
            year=2011,
            observatory="Bh",
            annotation_sets=(
                AnnotationSet("annotator-a", ({"id": "a"},)),
                AnnotationSet("annotator-b", ({"id": "b"},)),
                AnnotationSet("annotator-c", ({"id": "c"},)),
            ),
        )

        selected = [select_annotation_set(record, epoch=epoch, seed=2026).image_id for epoch in range(3)]

        self.assertEqual(len(set(selected)), 3)

    def test_selection_is_reproducible(self):
        record = ImageRecord(
            file_name="20110101000000Bh.jpeg",
            width=8,
            height=8,
            year=2011,
            observatory="Bh",
            annotation_sets=(AnnotationSet("a", ()), AnnotationSet("b", ())),
        )

        self.assertEqual(
            select_annotation_set(record, epoch=5, seed=11),
            select_annotation_set(record, epoch=5, seed=11),
        )


class ModelContractTests(unittest.TestCase):
    def test_training_loader_drops_a_single_item_tail_batch(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(5))
        config = TrainConfig(data_root="unused", batch_size=2, num_workers=0)

        batch_sizes = [len(batch[0]) for batch in build_training_loader(dataset, config)]

        self.assertEqual(batch_sizes, [2, 2])

    def test_deeplab_training_rejects_batch_size_one(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 2"):
            TrainConfig(data_root="unused", batch_size=1)

    def test_native_training_accepts_batch_size_one(self):
        config = TrainConfig(
            data_root="unused",
            model_name="native_unet",
            batch_size=1,
            gradient_accumulation=4,
            tta=8,
            close_kernel=5,
        )

        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation, 4)

    def test_training_rejects_an_unsupported_tta_count(self):
        with self.assertRaisesRegex(ValueError, "tta must be 1, 4, or 8"):
            TrainConfig(data_root="unused", tta=2)

    def test_native_dataset_keeps_a_center_crop_without_rgb_expansion(self):
        record = ImageRecord(
            file_name="20110101000000Bh.jpeg",
            width=8,
            height=8,
            year=2011,
            observatory="Bh",
            annotation_sets=(
                AnnotationSet(
                    "annotator-a",
                    ({"segmentation": [[3, 3, 5, 3, 5, 5, 3, 5]]},),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            pixels = np.full((8, 8), 128, dtype=np.uint8)
            pixels[2, 2] = 255
            Image.fromarray(pixels).save(image_dir / record.file_name, quality=100)
            dataset = FilamentDataset(
                [record], image_dir, image_size=4, seed=1, augment=False, native=True
            )

            image, target, name = dataset[0]

        self.assertEqual(name, record.file_name)
        self.assertEqual(tuple(image.shape), (1, 4, 4))
        self.assertEqual(tuple(target.shape), (1, 4, 4))
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(int(image[0].argmax()), 0)
        self.assertGreater(int(target.sum()), 0)

    def test_native_inference_restores_the_crop_to_the_original_canvas(self):
        class ForegroundModel(torch.nn.Module):
            def forward(self, image):
                return {"out": torch.full_like(image, 10)}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpeg"
            Image.new("L", (8, 8), color=128).save(path)
            instances, _ = predict_image(
                ForegroundModel(),
                path,
                image_size=4,
                threshold=0.5,
                min_area=1,
                device=torch.device("cpu"),
                native=True,
            )

        self.assertEqual(len(instances), 1)
        self.assertEqual(tuple(instances[0].shape), (8, 8))
        self.assertEqual(int(instances[0].sum()), 16)
        np.testing.assert_array_equal(instances[0][2:6, 2:6], np.ones((4, 4)))
        self.assertEqual(int(instances[0][:2].sum() + instances[0][6:].sum()), 0)

    def test_dihedral_tta_restores_each_prediction_before_averaging(self):
        class IdentityModel(torch.nn.Module):
            def forward(self, image):
                return {"out": image}

        pixels = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpeg"
            Image.fromarray(pixels).save(path, quality=100)
            plain = predict_probability(
                IdentityModel(), path, 4, torch.device("cpu"), native=True, tta=1
            )
            augmented = predict_probability(
                IdentityModel(), path, 4, torch.device("cpu"), native=True, tta=8
            )

        np.testing.assert_allclose(augmented, plain, atol=1e-6)

    def test_gradient_accumulation_keeps_the_final_partial_step(self):
        self.assertEqual(optimizer_step_batches(10, 4), (3, 7, 9))
        self.assertEqual(
            tuple(gradient_divisor(index, 10, 4) for index in range(10)),
            (4, 4, 4, 4, 4, 4, 4, 4, 2, 2),
        )

    def test_model_returns_one_logit_per_input_pixel(self):
        model = build_model(pretrained_backbone=False).eval()

        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 64, 64))["out"]

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))

    def test_native_model_returns_one_logit_per_input_pixel(self):
        model = build_model(pretrained_backbone=False, model_name="native_unet").eval()

        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 64, 64))["out"]

        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))

    def test_loss_prefers_correct_logits(self):
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        correct = torch.tensor([[[[-8.0, 8.0], [8.0, -8.0]]]])
        inverse = -correct

        self.assertLess(
            float(segmentation_loss(correct, target, positive_weight=3.0)),
            float(segmentation_loss(inverse, target, positive_weight=3.0)),
        )


if __name__ == "__main__":
    unittest.main()
