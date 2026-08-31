# Kaggle Competition

This project focuses on solving a Kaggle competition for the automated segmentation of solar filaments in H-Alpha images.

## Competition Objective

The goal is to develop a computer-vision or deep-learning solution that identifies every solar filament and produces an accurate segmentation mask for each one. The predicted masks should preserve the complete filament structure, including fine-scale features, while avoiding background regions.

The competition uses the MAGFiLO dataset, which contains manually annotated GONG H-Alpha observations. The images are grayscale JPEG files with a resolution of `2048 x 2048` pixels, and the annotations follow a COCO-style JSON format.

## Evaluation

Submissions are evaluated mainly with the Panoptic Quality (PQ) metric. Dice score, IoU score, the quality of one-to-many and many-to-one matches, and the quality of the complete solution are also considered.

## Work Required

The task is to solve the complete segmentation problem:

1. Inspect and understand the MAGFiLO images and annotations.
2. Prepare the data and convert polygon annotations into training masks.
3. Design and train a segmentation model.
4. Validate the model using suitable image-level and instance-level metrics.
5. Generate predictions for the test images.
6. Convert the predicted masks into the required RLE format.
7. Create the final CSV submission and upload it to Kaggle.

## Training on Kaggle GPU

The training notebook will be uploaded to Kaggle and executed with a GPU accelerator. The workflow is:

1. Create a Kaggle notebook and attach the competition dataset.
2. Enable a GPU accelerator in the notebook settings.
3. Load the training images and COCO-style annotation JSON file.
4. Train the segmentation model using the available Kaggle GPU resources.
5. Save the best model checkpoint and validation results.
6. Run inference on the test set.
7. Encode the predicted masks as RLE counts and save the submission CSV.
8. Submit the CSV file to the Kaggle competition for evaluation.

The final notebook should contain the complete, reproducible pipeline, from data loading and preprocessing to model training, inference, RLE encoding, and submission generation.
