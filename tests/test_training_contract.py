import unittest

import torch

from solar_filament.data import AnnotationSet, ImageRecord
from solar_filament.model import build_model
from solar_filament.training import (
    TrainConfig,
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

    def test_model_returns_one_logit_per_input_pixel(self):
        model = build_model(pretrained_backbone=False).eval()

        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 64, 64))["out"]

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
