"""
Custom dataset.

Mostly copy-paste from coco.py
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} to custom dataset does not exist'
    training_json_file = 'train_annotations.json'
    validation_json_file = 'test_annotations.json'
    PATHS = {
        "train": (root / "train_images", root / "annotations" / training_json_file),
        "val": (root / "test_images", root / "annotations" / validation_json_file),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
