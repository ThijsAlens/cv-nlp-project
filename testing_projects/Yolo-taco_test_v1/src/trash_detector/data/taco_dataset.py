"""Dataset download and conversion utilities for the TACO dataset."""

from __future__ import annotations

import os
import shutil
from io import BytesIO

import concurrent.futures
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

from trash_detector.data.label_maps import build_identity_map, load_explicit_label_map
from trash_detector.utils.io import ensure_dir, read_json, write_json, write_yaml


DEFAULT_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"


@dataclass(slots=True)
class TacoPaths:
    """Resolved project paths for raw and prepared TACO data."""

    root: Path
    raw_dir: Path
    prepared_dir: Path
    annotations_path: Path
    images_dir: Path


@dataclass(slots=True)
class TacoCategory:
    """Representation of one TACO category entry."""

    id: int
    name: str
    supercategory: str


class TacoDatasetManager:
    """Download, inspect, and convert TACO data into Ultralytics YOLO format."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        raw_dir = project_root / "data" / "raw" / "taco"
        prepared_dir = project_root / "data" / "prepared"
        self.paths = TacoPaths(
            root=project_root,
            raw_dir=raw_dir,
            prepared_dir=prepared_dir,
            annotations_path=raw_dir / "annotations.json",
            images_dir=raw_dir / "images",
        )

    def download_annotations(self, force: bool = False) -> Path:
        """Download the official TACO annotations JSON if needed."""
        ensure_dir(self.paths.raw_dir)
        if self.paths.annotations_path.exists() and not force:
            return self.paths.annotations_path

        response = requests.get(DEFAULT_ANNOTATIONS_URL, timeout=60)
        response.raise_for_status()
        self.paths.annotations_path.write_bytes(response.content)
        return self.paths.annotations_path

    def load_annotations(self) -> dict:
        """Load the TACO COCO-format annotations file."""
        if not self.paths.annotations_path.exists():
            raise FileNotFoundError(
                f"Missing annotations file at {self.paths.annotations_path}. "
                "Run the download step first."
            )
        return read_json(self.paths.annotations_path)

    def categories(self) -> List[TacoCategory]:
        """Return all categories defined in the dataset."""
        payload = self.load_annotations()
        return [
            TacoCategory(id=item["id"], name=item["name"], supercategory=item["supercategory"])
            for item in payload["categories"]
        ]

    def write_category_summary(self) -> Path:
        """Create a small JSON summary with class frequencies.

        This is useful when deciding whether to train on all 60 classes or a focused subset.
        """
        payload = self.load_annotations()
        cat_by_id = {cat["id"]: cat for cat in payload["categories"]}
        counts = Counter(ann["category_id"] for ann in payload["annotations"])
        rows = []
        for cat_id, cat in sorted(cat_by_id.items(), key=lambda item: counts[item[0]], reverse=True):
            rows.append(
                {
                    "id": cat_id,
                    "name": cat["name"],
                    "supercategory": cat["supercategory"],
                    "instances": counts[cat_id],
                }
            )

        out_path = self.project_root / "assets" / "taco_category_summary.json"
        write_json(out_path, rows)
        return out_path

    def download_images(
        self,
        max_images: Optional[int] = None,
        num_workers: int = 8,
        use_resized_fallback: bool = True,
    ) -> Path:
        """Download TACO images referenced by the annotations file.

        The original TACO repo downloads images serially. This implementation keeps the same
        source URLs but adds concurrency and a fallback to the 640px Flickr version.
        """
        ensure_dir(self.paths.images_dir)
        payload = self.load_annotations()
        images = payload["images"][:max_images] if max_images else payload["images"]

        def _download_one(image_info: dict) -> Tuple[str, bool, str]:
            file_name = image_info["file_name"]
            target_path = self.paths.images_dir / file_name
            ensure_dir(target_path.parent)

            if target_path.exists():
                return file_name, True, "cached"

            urls = [image_info.get("flickr_url")]
            if use_resized_fallback:
                urls.append(image_info.get("flickr_640_url"))

            last_error = "unknown error"
            for url in [u for u in urls if u]:
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    with Image.open(BytesIO(response.content)) as image:
                        # Save through Pillow to normalize partially inconsistent content-types.
                        if image.getexif():
                            image.save(target_path, exif=image.info.get("exif"))
                        else:
                            image.save(target_path)
                    return file_name, True, "downloaded"
                except Exception as exc:  # noqa: BLE001 - download failures should not abort all jobs.
                    last_error = str(exc)

            return file_name, False, last_error

        failures: List[Tuple[str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_download_one, image_info) for image_info in images]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading TACO"):
                file_name, ok, message = future.result()
                if not ok:
                    failures.append((file_name, message))

        if failures:
            write_json(self.project_root / "assets" / "taco_download_failures.json", failures)
        return self.paths.images_dir

    def prepare_yolo_dataset(
        self,
        output_name: str,
        label_map_path: Optional[Path] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        min_box_pixels: int = 8,
        copy_images: bool = False,
    ) -> Path:
        """Convert TACO COCO annotations into Ultralytics YOLO detection format.

        Images are split at the image level, then label files are written in YOLO's normalized
        x_center, y_center, width, height format.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        payload = self.load_annotations()
        categories = sorted(payload["categories"], key=lambda item: item["id"])
        category_names = [category["name"] for category in categories]
        category_id_to_name = {category["id"]: category["name"] for category in categories}

        if label_map_path:
            selected_map = load_explicit_label_map(label_map_path, category_names)
        else:
            selected_map = build_identity_map(category_names)

        selected_classes = list(selected_map.keys())
        selected_class_ids = {
            category_id for category_id, name in category_id_to_name.items() if name in selected_map
        }

        image_by_id = {image["id"]: image for image in payload["images"]}
        annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            if annotation["category_id"] in selected_class_ids:
                annotations_by_image[annotation["image_id"]].append(annotation)

        # Keep only images that actually contain at least one selected annotation.
        candidate_image_ids = [image_id for image_id, anns in annotations_by_image.items() if anns]
        random.Random(seed).shuffle(candidate_image_ids)

        total = len(candidate_image_ids)
        train_cut = int(total * train_ratio)
        val_cut = train_cut + int(total * val_ratio)
        split_to_ids = {
            "train": candidate_image_ids[:train_cut],
            "val": candidate_image_ids[train_cut:val_cut],
            "test": candidate_image_ids[val_cut:],
        }

        dataset_root = self.paths.prepared_dir / output_name
        for split in ("train", "val", "test"):
            ensure_dir(dataset_root / "images" / split)
            ensure_dir(dataset_root / "labels" / split)

        split_counts = {}
        skipped_missing_images = []
        skipped_small_boxes = 0

        for split, image_ids in split_to_ids.items():
            written_images = 0
            written_annotations = 0
            for image_id in image_ids:
                image_info = image_by_id[image_id]
                source_image = self.paths.images_dir / image_info["file_name"]
                if not source_image.exists():
                    skipped_missing_images.append(image_info["file_name"])
                    continue

                width = float(image_info["width"])
                height = float(image_info["height"])
                lines: List[str] = []
                for annotation in annotations_by_image[image_id]:
                    x_min, y_min, box_width, box_height = annotation["bbox"]
                    if box_width < min_box_pixels or box_height < min_box_pixels:
                        skipped_small_boxes += 1
                        continue

                    class_name = category_id_to_name[annotation["category_id"]]
                    class_id = selected_map[class_name]
                    x_center = (x_min + box_width / 2.0) / width
                    y_center = (y_min + box_height / 2.0) / height
                    norm_width = box_width / width
                    norm_height = box_height / height
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

                if not lines:
                    continue

                label_path = dataset_root / "labels" / split / (Path(image_info["file_name"]).stem + ".txt")
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                target_image = dataset_root / "images" / split / Path(image_info["file_name"]).name
                if target_image.exists() or target_image.is_symlink():
                    target_image.unlink()

                # Windows often blocks symlink creation unless Developer Mode or elevated privileges are enabled.
                # Fall back to copying in that case so dataset preparation remains portable.
                use_copy = copy_images or os.name == "nt"
                if use_copy:
                    shutil.copy2(source_image, target_image)
                else:
                    try:
                        target_image.symlink_to(source_image.resolve())
                    except OSError:
                        shutil.copy2(source_image, target_image)

                written_images += 1
                written_annotations += len(lines)

            split_counts[split] = {
                "images": written_images,
                "annotations": written_annotations,
            }

        data_yaml = {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for name, index in selected_map.items()},
        }
        write_yaml(dataset_root / "dataset.yaml", data_yaml)

        metadata = {
            "selected_classes": selected_classes,
            "split_counts": split_counts,
            "skipped_missing_images": sorted(set(skipped_missing_images)),
            "skipped_small_boxes": skipped_small_boxes,
            "used_symlinks": not copy_images,
            "source_annotations": str(self.paths.annotations_path),
        }
        write_json(dataset_root / "metadata.json", metadata)
        return dataset_root
