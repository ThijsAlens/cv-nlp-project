#!/usr/bin/env python3
"""Count how many images contain each TACO class."""

import json
from collections import defaultdict
from pathlib import Path

ann_file = Path("data/raw/taco/annotations.json")

data = json.loads(ann_file.read_text())

# category_id -> name
categories = {c["id"]: c["name"] for c in data["categories"]}

# category_id -> set(image_id)
images_per_class = defaultdict(set)

for ann in data["annotations"]:
    images_per_class[ann["category_id"]].add(ann["image_id"])

results = [
    (categories[cid], len(imgs))
    for cid, imgs in images_per_class.items()
]

results.sort(key=lambda x: x[1], reverse=True)

print("\nImages containing each class:\n")

for name, count in results:
    print(f"{name:30} {count}")
