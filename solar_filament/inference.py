from __future__ import annotations

import json
import time
from pathlib import Path

from .masks import connected_components
from .submission import build_submission_rows, validate_submission, write_submission


def predict_probability(
    model,
    image_path: Path | str,
    image_size: int,
    device,
    native: bool = False,
    tta: int = 1,
):
    import torch
    from PIL import Image
    from torch.nn import functional as functional
    from torchvision.transforms import functional as transform

    if tta not in (1, 4, 8):
        raise ValueError("tta must be 1, 4, or 8")
    with Image.open(image_path) as source:
        width, height = source.size
        source = source.convert("L")
        if native:
            import numpy as np

            from .preprocessing import native_crop

            image = torch.from_numpy(native_crop(np.asarray(source), image_size)).unsqueeze(0)
        else:
            image = transform.pil_to_tensor(source).float() / 255.0
            image = transform.resize(image, [image_size, image_size], antialias=True)
            image = image.repeat(3, 1, 1)
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = []
        transforms = [(False, 0)] if tta == 1 else [(False, k) for k in range(4)]
        if tta == 8:
            transforms += [(True, k) for k in range(4)]
        for flip, rotations in transforms:
            transformed = image.flip(-1) if flip else image
            transformed = torch.rot90(transformed, rotations, (-2, -1))
            logits = model(transformed)["out"]
            logits = torch.rot90(logits, -rotations, (-2, -1))
            logits = logits.flip(-1) if flip else logits
            probabilities.append(logits.sigmoid())
        probability = torch.stack(probabilities).mean(0)[0, 0]
    if native:
        canvas = torch.zeros((height, width), device=probability.device)
        top, left = (height - image_size) // 2, (width - image_size) // 2
        canvas[top : top + image_size, left : left + image_size] = probability
        return canvas.cpu().numpy()
    return functional.interpolate(
        probability[None, None], size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()


def predict_image(
    model,
    image_path: Path | str,
    image_size: int,
    threshold: float,
    min_area: int,
    device,
    native: bool = False,
    tta: int = 1,
    close_kernel: int = 0,
):
    started = time.perf_counter()
    probabilities = predict_probability(
        model, image_path, image_size, device, native=native, tta=tta
    )
    instances = connected_components(
        probabilities,
        threshold=threshold,
        min_area=min_area,
        close_kernel=close_kernel,
    )
    return instances, time.perf_counter() - started


def load_model(checkpoint_path: Path | str, device=None):
    import torch

    from .model import build_model

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(
        pretrained_backbone=False,
        model_name=config.get("model_name", "deeplabv3_resnet50"),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device), config, device


def infer_directory(
    checkpoint_path: Path | str,
    image_dir: Path | str,
    output_csv: Path | str,
    run_manifest: Path | str | None = None,
):
    model, config, device = load_model(checkpoint_path)
    image_dir = Path(image_dir)
    rows: list[dict[str, str]] = []
    run_images = []
    image_paths = sorted(image_dir.glob("*.jpeg"))
    for image_path in image_paths:
        instances, elapsed = predict_image(
            model,
            image_path,
            image_size=int(config["image_size"]),
            threshold=float(config["threshold"]),
            min_area=int(config["min_area"]),
            device=device,
            native=config.get("model_name") == "native_unet",
            tta=int(config.get("tta", 1)),
            close_kernel=int(config.get("close_kernel", 0)),
        )
        rows.extend(build_submission_rows({image_path.stem: instances}))
        run_images.append(
            {
                "image": image_path.name,
                "instances": len(instances),
                "areas": [int(mask.sum()) for mask in instances],
                "seconds": elapsed,
            }
        )
    write_submission(output_csv, rows)
    report = validate_submission(
        output_csv, expected_stems={path.stem for path in image_paths}
    )
    if report.errors:
        raise ValueError("invalid submission: " + "; ".join(report.errors))
    manifest_path = Path(run_manifest) if run_manifest else Path(output_csv).with_suffix(".run.json")
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "config": config,
                "processed_images": len(image_paths),
                "submission_rows": len(rows),
                "images": run_images,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
